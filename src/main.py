from PyQt5 import QtGui, QtWidgets, uic, QtCore
from scipy.io.wavfile import write
from geometry import Geometry
from datetime import datetime
from fpdf import FPDF
import sounddevice as sd
import pyqtgraph.exporters
import pyqtgraph as pg
import soundfile as sf
import numpy as np
import rfda, os, re, shutil

def audio_callback(indata, frames, time, status):
    '''
    This is called (from a separate thread) for each audio block. It
    consumes the audio (indata) from the microphone input stream, and
    rolls it into a global array for access outside the stream thread.

    By rolling like this, the program only stores that last few seconds
    to reduce plotting complexity.

    Additionally, if actively recording, buffers data into a list of 
    ndarrays which will later be FFT'd etc.

    The start and stop of a recording is controlled by handle_recording()
    '''

    global audio_wfrm_data
    global audio_list
    global recording_flag

    # Roll the data into array from right to left for plotting
    shift = len(indata)
    audio_wfrm_data = np.roll(audio_wfrm_data, -shift, axis=0)
    audio_wfrm_data[-shift:] = np.concatenate(indata)

    if recording_flag == True:
        audio_list.append(np.concatenate(indata))

class TableModel(QtCore.QAbstractTableModel):
    '''
    In Qt, a QTableView requires a model for how to display
    the data in the table. This is user defined for flexibility.

    Taken from:
    https://www.pythonguis.com/tutorials/qtableview-modelviews
    -numpy-pandas/#introduction-to-qtableview
    '''

    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == QtCore.Qt.DisplayRole:
            return self._data[index.row()][index.column()]

    def rowCount(self, index):
        return len(self._data)

    def columnCount(self, index):
        return len(self._data[0])

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):

        # Load external ui file. Edit this ui file in 'Qt Designer'
        super(MainWindow, self).__init__(*args, **kwargs)
        uic.loadUi('RFDA.ui', self)

        # To be populated with plotLine objects
        self.audio_wfrm_line     = None
        self.audio_wfrm_fft_line = None
        self.audio_envelope_line = None
        self.audio_exp_fit_line  = None

        # Movable vertical line for selecting fft peak
        self.fft_peak_vline = pg.InfiniteLine(movable=True,
                                           pos=0.1)

        # Movable horizontal line for selecting audio impulse threshold
        self.audio_trig_hline = pg.InfiniteLine(movable=True,
                                           angle=0,
                                           pos=0.125)

        # Text object to show equation of fitted exponential on graph
        self.exp_fit_eqn = pg.TextItem(color=(255,255,255))

        # Values of the sample are populated in the ui
        self.sample_geometry = Geometry()

        # Input stream from sounddevice. Initialized in select_input_device()
        self.stream = None

        # Welcome message
        self.msg("Welcome to pyRFDA companion!")
        self.msg("Start by defining sample type and parameters," + \
                " then click the \"Create Sample\" button.")

        # Prepare the ui
        self.setup_graphs()
        self.setup_inputs()
        self.hide_sample_params()

        # Timer to update live waveform graph
        self.update_wfrm_timer = QtCore.QTimer()
        self.update_wfrm_timer.setInterval(int(1000/30))
        self.update_wfrm_timer.timeout.connect(self.update_audio_wfrm)
        self.update_wfrm_timer.start()

        # Timer to check for impulses in incoming audio
        self.check_impulse_timer = QtCore.QTimer()
        self.check_impulse_timer.setInterval(int(1000/30))
        self.check_impulse_timer.timeout.connect(self.handle_recording)
        self.check_impulse_timer.start()

        # List of lists stores data of runs_table. DO NOT change this
        # order without also updating the order in process_run() and
        # update_run_avgs()
        header = [ "Freq [Hz] ",
                   "Time [s] ",
                   "E [GPa] ",
                   "G [GPa] ",
                   "Q^-1 ",
                   "Damping ",
                   "MSE",
                   "Notes "]
        self.runs_table_data = [header, ["-"]*len(header)]
        self.runs_table_model = TableModel(self.runs_table_data)
        self.runs_table.setModel(self.runs_table_model)

        # List of lists stores data of avgs_table. DO NOT change this
        # order without also updating the order in save_run()
        header = [ "F_f (E) [Hz] ",
                   "F_t (G) [Hz] ",
                   "E [GPa] ",
                   "G [GPa] ",
                   "Poisson R ",
                   "Q^-1 "]
        self.avgs_table_data = [header, ["-"]*len(header)]
        self.avgs_table_model = TableModel(self.avgs_table_data)
        self.avgs_table.setModel(self.avgs_table_model)

        # Dictionary that stores measured and computed values across
        # multiple runs when collecting E and G. Converted to a list
        # and appended to the runs_table model when the save_run_btn 
        # is clicked.
        self.run_data = {}

        self.sample_rate = 44100 # Updated later based on input dev

        self.save_dir = None # Updated later by save dialog

    def setup_graphs(self):

        # Live microphone audio graph
        self.audio_wfrm_line = \
                self.audio_wfrm.plot(audio_wfrm_data, 
                                     autodownsample=True)
        self.audio_wfrm.setYRange(min=-0.2, max=0.2)
        self.audio_wfrm.showGrid(x=True, y=True)
        self.audio_wfrm.setTitle("Live Microphone Audio", size="12pt")
        self.audio_wfrm.setLabel("left", "Intensity")
        self.audio_wfrm.setLabel("right", "") # For padding
        self.audio_wfrm.setLabel("bottom", "Sample")
        self.audio_wfrm.addItem(self.audio_trig_hline)

        self.audio_wfrm.addItem(self.exp_fit_eqn)
        self.audio_envelope_line = self.audio_wfrm.plot(
                pen={'color': 'red'}, downsample=100)

        self.audio_exp_fit_line = self.audio_wfrm.plot(pen={'color': 'blue'})

        # FFT of audio recording after impulse is detected
        self.audio_wfrm_fft_line = self.audio_wfrm_fft.plot()
        self.audio_wfrm_fft.showGrid(x=True, y=True)
        self.audio_wfrm_fft.setTitle("FFT of Recorded Waveform", size="12pt")
        self.audio_wfrm_fft.setLabel("left", "Amplitude")
        self.audio_wfrm_fft.setLabel("right", "") # For padding
        self.audio_wfrm_fft.setLabel("bottom", "Frequency [Hz]")
        self.audio_wfrm_fft.addItem(self.fft_peak_vline)

    def update_audio_wfrm(self):
        # Called on QT timer to update graphs with latest audio
        global audio_wfrm_data
        self.audio_wfrm_line.setData(audio_wfrm_data)

    def handle_recording(self):
        '''
        Called on QT timer to set recording_flag causing audio_callback()
        to start storing data in the audio_queue.

        The recording_flag is set True when audio above the trigger threshold
        (hline on audio graph), and the "Recording" button is pressed.

        Once the input audio crosses the trigger threshold again, the
        recording_flag is set False, and the audio_list is written to a file.
        '''

        global audio_wfrm_data
        global audio_list
        global recording_flag

        # Only consider the most recent few samples. This acts as a slight
        # filter on triggering.
        max_sample = np.max(audio_wfrm_data[-10000:])
        trig = self.audio_trig_hline.value()

        # Start of recording
        if max_sample > trig and self.record_btn.isChecked():
            if recording_flag == False:
                self.msg("Detected impulse. Recording...")
                recording_flag = True

        # End of recording
        elif max_sample < trig and recording_flag == True:

            recording_flag = False
            self.record_btn.setEnabled(False)
            self.record_btn.setText("Saving...")

            # Flatten list to single ndarray
            audio_arr = np.concatenate(audio_list)

            # Compute sample data from run and plot it
            self.process_run(audio_arr)

            self.record_btn.setChecked(False)
            self.record_btn.setText("Record")
            self.record_btn.setEnabled(True)
            self.msg("Finished recording")

    def process_run(self, audio_arr):
        '''
        Compute E, G, damping etc from the recorded run and pass them 
        to stage_run() so they show up on the table.
        '''

        # Clip the first 5% of audio to avoid recording the sound
        # of the impulse itself.
        # TODO make this adjustable by the ui?
        audio_arr = audio_arr[int(0.05*audio_arr.size):]

        fft_arr = np.abs(np.fft.rfft(audio_arr))
        freqs_arr = np.fft.rfftfreq(audio_arr.size, 1.0/self.sample_rate)

        # Graph the FFT
        self.audio_wfrm_fft_line.setData(freqs_arr, fft_arr)

        # Crop the range to [0, 20kHz] if necessary
        if np.max(freqs_arr) > 20000:
            self.audio_wfrm_fft.setXRange(0, 20000)

        # Find the peak frequency based on FFT magnitude by cropping
        # to freq/values near the vline and searching for the largest.
        peak_est = self.fft_peak_vline.value()
        indices = np.where(
                (freqs_arr > peak_est - 100) & (freqs_arr < peak_est + 100))
        cropped_fft = fft_arr[indices]
        cropped_freqs = freqs_arr[indices]
        resonant_freq = cropped_freqs[np.argmax(cropped_fft)]
        self.run_data["freq"] = resonant_freq

        # Compute modulus from peak based on run type. Q^1 is only recorded
        # for E mod runs because the support wires are less interfering.
        match self.run_type_dropdown.currentText():

            case "Elastic Mod":
                E = rfda.elastic_modulus(self.sample_geometry, resonant_freq)
                self.run_data["E"] = E
                self.run_data.pop("G", "")

                # Upper envelope in red
                ue = rfda.upper_envelope(audio_arr)
                self.audio_envelope_line.setData(ue[0], ue[1])

                # Fit the exponential decay
                try:
                    a, b, c = rfda.exponential_fit(audio_arr)
                    x = np.linspace(0, audio_arr.size, 5000)
                    exp_fit = rfda.damped_exp(x, a, b, c)
                    self.audio_exp_fit_line.setData(x, exp_fit)
                    
                    # Display equation on plot just above max value
                    self.exp_fit_eqn.setText(text=
                                           "Fit: " + "{:.3E}".format(a) \
                                         + " * exp(-" + "{:.3E}".format(b) \
                                         + "*t) + " + "{:.3E}".format(c))
                    self.exp_fit_eqn.setPos(0,1.1*np.max(audio_arr))

                    # b coeff of exponential fit is the damping coeff
                    self.run_data["damping"] = b

                    # Inverse quality factor
                    self.run_data["Q-1"] = rfda.inv_Q_factor(b, resonant_freq)

                    # Means squared error
                    # TODO add MSE here


                except RuntimeError:
                    self.msg("Error: Could not fit exponential to data.")
                    return

            case "Shear Mod":
                G = rfda.shear_modulus(self.sample_geometry, resonant_freq)
                self.run_data["G"] = G
                self.run_data.pop("E", "")

            case _:
                self.msg("Error: Unknown run type.")

        # Graph the recorded audio, and envelope/exp if needed
        self.update_wfrm_timer.stop()
        self.check_impulse_timer.stop()

        self.audio_wfrm_line.setData(audio_arr)
        self.audio_wfrm.autoRange()

        # Time of the run in seconds
        self.run_data["time"] = audio_arr.size/self.sample_rate

        # Show the run on the runs_table
        self.stage_run()

    def stage_run(self):
        '''
        Convert the run_data dictionary to a list in the correct
        order for adding to the runs_table model. This data is
        always inserted at the top row of the runs_table. To save
        a run, a new row is simply inserted at the top of the table.
        '''

        run_list = []

        # TODO iterate over the header list to streamline this?

        run_list.append( "{:.2f}".format(self.run_data["freq"]))
        run_list.append( "{:.2f}".format(self.run_data["time"]))

        # Only E or G exists in a given run
        try:
            run_list.append( "{:.1f}".format(self.run_data["E"]/1E9))
        except KeyError:
            run_list.append("-")

        try:
            run_list.append( "{:.1f}".format(self.run_data["G"]/1E9))
        except KeyError:
            run_list.append("-")

        try:
            run_list.append( "{:.3E}".format(self.run_data["Q-1"]))
        except KeyError:
            run_list.append("-")

        try:
            run_list.append( "{:.3E}".format(self.run_data["damping"]))
        except KeyError:
                run_list.append("-")

        try:
            run_list.append( "{:.3f}".format(self.run_data["mse"]))
        except KeyError:
            run_list.append("-")

        run_list.append(self.notes_box.toPlainText())

        # Overwrite the last row
        self.runs_table_data[-1] = run_list

        # Force the table to update
        self.runs_table_model.layoutChanged.emit()
        self.save_run_btn.setEnabled(True)

    def update_run_avgs(self):
        '''
        Updates the run_avgs table with the data from a new run.
        Called when a run is saved or deleted.
        '''

        # For averaging
        sum_f_f = 0.0
        sum_t_f = 0.0
        sum_E = 0.0
        sum_G = 0.0
        sum_Q_inv = 0.0

        n_E = 0
        n_G = 0

        # Skip first and last rows (header and staged run)
        for row in self.runs_table_data[1:-1]:
            
            # Check what type of measurement it was
            if row[3] == "-":
                # E measurment
                sum_f_f += float(row[0])
                sum_E += float(row[2])
                sum_Q_inv += float(row[4])
                n_E += 1
            else:
                # G measurment
                sum_t_f += float(row[0])
                sum_G += float(row[3])
                n_G += 1

        avg_f_f = '-'
        avg_t_f = '-'
        avg_E = '-'
        avg_G = '-'
        avg_Q_inv = '-'

        # Avoid dividing by zero
        if n_E:
            avg_f_f   = "{:.1f}".format(sum_f_f/n_E)
            avg_E     = "{:.1f}".format(sum_E/n_E)
            avg_Q_inv = "{:.3E}".format(sum_Q_inv/n_E)

        if n_G:
            avg_t_f = "{:.2f}".format(sum_t_f/n_G)
            avg_G   = "{:.1f}".format(sum_G/n_G)

        # Poisson ratio using E and G
        poisson_r = '-'
        if (n_E and n_G):
            poisson_r = "{:.3f}".format(rfda.poisson(sum_E/n_E, sum_G/n_G))
            
        self.avgs_table_data[-1] = [ avg_f_f,
                                     avg_t_f,
                                     avg_E,
                                     avg_G,
                                     poisson_r,
                                     avg_Q_inv ]

        # Force table to update
        self.avgs_table_model.layoutChanged.emit()

    def setup_inputs(self):
        '''
        # Certain inputs like dropdown menus can't be configured in
        # Qt Designer so they're configured here.
        '''

        # Add input devices from sd to the devices dropdown
        devs = sd.query_devices()
        dev_names = [ dev['name'] \
                for dev in devs if dev['max_input_channels'] > 0]
        self.input_dropdown.addItems(dev_names)

        # Set to the default device since it should always be present
        default_idx = self.input_dropdown.findText("default")
        if default_idx != -1:
            self.input_dropdown.setCurrentIndex(default_idx)

        # Connect to allow changing device
        self.input_dropdown.activated.connect(
                self.select_input_device)

        # Call once to start the default device
        self.select_input_device()

        # Options for sample geometry
        self.geometry_dropdown.addItems(["Rectangle", "Rod", "Disc"])
        self.geometry_dropdown.activated.connect(
                self.hide_sample_params)
        self.create_geo_btn.clicked.connect(self.create_geometry)

        # Run type (E or G)
        self.run_type_dropdown.addItems(["Elastic Mod", "Shear Mod"])

        # Save run button
        self.save_run_btn.clicked.connect(self.save_run)

        # Delete run from runs table button
        self.delete_run_btn.clicked.connect(self.delete_run)

        # Record button
        self.record_btn.clicked.connect(self.record)

    def msg(self, s):
        # Shorter wrapper for sending status message to the text box
        s = "> " + str(s)
        self.status_msg_box.appendPlainText(s)

    ## SIGNALS ##

    def delete_run(self):
        '''
        When the delete_run_btn is clicked, any selected rows in the
        table are deleted from the model data. Also delete the corresponding
        directory from the save location, and rename the other directories
        to have ascending numeric IDs.
        '''

        # selectedRows() returns a QModelIndex
        selected = self.runs_table.selectionModel().selectedRows()
        rows = [idx.row() for idx in selected]

        # Iterate in reverse to avoid indices changing while deleting
        for row in reversed(rows):
            # Don't remove the header or staged (last) run
            if (row != 0) and (row != len(self.runs_table_data) - 1):
                del self.runs_table_data[row]

            # Delete saved run directory
            shutil.rmtree(os.path.join(self.save_dir, str(row)),
                          ignore_errors=True)

        # List all folders in the base directory
        folders = [f for f in os.listdir(self.save_dir) \
                  if os.path.isdir(os.path.join(self.save_dir, f))]
        
        # Sort folders by their numerical names
        folders.sort(key=lambda x: int(x))
        
        # Rename folders to have consecutive numbers starting from 1
        for index, folder in enumerate(folders):
            new_name = str(index + 1)
            current_path = os.path.join(self.save_dir, folder)
            new_path = os.path.join(self.save_dir, new_name)
            
            # Rename the folder
            os.rename(current_path, new_path)

        # Force table to update
        self.runs_table_model.layoutChanged.emit()
        self.update_run_avgs()

    def create_report(self):
        '''
        Creates and exports a PDF report of the measurements collected 
        using FPDF for formatting.
        '''

        date = datetime.now().strftime('%Y-%m-%d %H:%M')
        file_name = os.path.join(self.save_dir, "Report.pdf")

        # Title
        title = "RFDA Report - \"" + \
                str.lstrip(self.name_box.text()) + \
                "\" - " + date
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 20)
        pdf.cell(40,10, title)

        table_cell_width = 25
        table_cell_height = 6

        header = []
        params = []
        match self.sample_geometry.shape():
            case "rect":
                header = ["Type",
                         "Length [mm]",
                         "Width [mm]",
                         "Thickness [mm]",
                         "Mass [g]"]

                params = [self.sample_geometry.shape(),
                          str(self.sample_geometry._L),
                          str(self.sample_geometry._b),
                          str(self.sample_geometry._t),
                          str(self.sample_geometry._m)]

            case "rod":
                header = ["Type",
                         "Length [mm]",
                         "Diameter [mm]",
                         "Mass [g]"]

                params = [self.sample_geometry.shape(),
                          str(self.sample_geometry._L),
                          str(self.sample_geometry._d),
                          str(self.sample_geometry._m)]
            case "disc":
                header = ["Type",
                         "Diameter [mm]",
                         "Thickness [mm]",
                         "Mass [g]"]

                params = [self.sample_geometry.shape(),
                          str(self.sample_geometry._d),
                          str(self.sample_geometry._t),
                          str(self.sample_geometry._m)]

        ## Print sample parameters table

        # Subheading
        pdf.ln(15)
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(40,10, "Sample Parameters:")

        # Header
        pdf.ln(10)
        pdf.set_font('Arial', 'B', 10)
        for col in header:
            pdf.cell(table_cell_width,
                     table_cell_height,
                     col,
                     align='C',
                     border=2)

        # Values
        pdf.ln(table_cell_height)
        pdf.set_font('Arial', '', 10)
        for col in params:
            pdf.cell(table_cell_width,
                     table_cell_height,
                     col,
                     align='C',
                     border=1)

        ## Print runs table minus the notes because they run off the page

        # Subheading
        pdf.ln(10)
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(40,10, "Recorded Runs:")
        pdf.ln(10)

        # Header
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(15, 6, "Run", align='C', border=2)
        for col in self.runs_table_data[0][:-1]:
            pdf.cell(table_cell_width,
                     table_cell_height,
                     col,
                     align='C',
                     border=2)

        pdf.ln(table_cell_height)
        pdf.set_font('Arial', '', 10)
        for i,row in enumerate(self.runs_table_data[1:-1]):

            # Print row number
            pdf.cell(15,
                     table_cell_height,
                     str(i),
                     align='C',
                     border=1)

            # Print values in that row
            for val in row[:-1]:
                pdf.cell(table_cell_width,
                         table_cell_height,
                         str(val),
                         align='C',
                         border=1)
            pdf.ln(table_cell_height)
        pdf.ln(10)

        ## Print averages table

        # Subtitle
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(40,10, "Computed Averages and Values:")
        pdf.ln(10)

        # Header
        header = ["Flex. F [Hz]",
                 "Tors. F [Hz]",
                 "E [GPa]",
                 "G [GPa]",
                 "Poisson R [-]",
                 "Q^1 [-]"]

        pdf.set_font('Arial', 'B', 10)
        for col in header:
            pdf.cell(table_cell_width,
                     table_cell_height,
                     col,
                     align='C',
                     border=2)

        # Values
        pdf.ln(table_cell_height)
        pdf.set_font('Arial', '', 10)
        for val in self.avgs_table_data[-1]:

            if type(val) is str:
                pdf.cell(table_cell_width,
                         table_cell_height,
                         str(val),
                         align='C',
                         border=1)
            else:
                pdf.cell(table_cell_width,
                         table_cell_height,
                         "{:.2E}".format(val),
                         align='C',
                         border=1)

        ## Print notes from runs table

        # Subtitle
        pdf.ln(10)
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(40,10, "Run Notes:")
        pdf.ln(10)

        # Header
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(15, table_cell_height, "Run", align='C', border=2)
        pdf.cell(180, 6, "Note", align='C', border=2)

        # Values
        pdf.ln(table_cell_height)
        pdf.set_font('Arial', '', 10)
        for i,row in enumerate(self.runs_table_data[1:-1]):

            # Print row number
            pdf.cell(15,
                     table_cell_height,
                     str(i),
                     align='C',
                     border=1)

            # Note is the last value in the TableModel
            pdf.cell(180,
                     table_cell_height,
                     row[-1],
                     align='L',
                     border=1)
            pdf.ln(table_cell_height)
        pdf.ln(10)

        # Write the file
        pdf.output(file_name, 'F')

    def record(self):
        '''
        Prepare the waveform plot and flush old data from arrays.
        '''

        global audio_list
        audio_list = []
        
        self.audio_wfrm_line.setData([0],[0])
        self.audio_envelope_line.setData([0],[0])
        self.audio_exp_fit_line.setData([0],[0])
        self.audio_wfrm.setYRange(min=-0.2, max=0.2)
        self.audio_wfrm.setXRange(min=0, max=100000)
        self.update_wfrm_timer.start()
        self.check_impulse_timer.start()

    def save_run(self):
        '''
        The current run is "staged" in the last row of the table.
        This allows it to be overwritten as many times as desired until
        the save_run_btn is clicked triggering this method which adds
        a new row so that the run is "saved" from being overwritten.

        This also exports the graphs and audio collected to the path
        specified by the user in the Save As dialog.
        '''

        self.record_btn.setEnabled(False)

        # Prompt user to create save location
        if self.save_dir is None:
            self.save_as()
        if self.save_dir is None:
            self.msg("No save location specified.")
            return

        directory = os.path.join(
                self.save_dir, str(len(self.runs_table_data) - 1))
        os.mkdir(directory)

        # Export graphs to png
        self.audio_wfrm.autoRange()
        file_path = os.path.join(directory, "audio_graph.png")
        exporter = pg.exporters.ImageExporter(
                self.audio_wfrm.plotItem)
        exporter.parameters()['width'] = 1500
        exporter.parameters()['height'] = 1000
        exporter.export(file_path)

        self.audio_wfrm_fft.autoRange()
        file_path = os.path.join(directory, "audio_fft_graph.png")
        exporter = pg.exporters.ImageExporter(
                self.audio_wfrm_fft.plotItem)
        exporter.parameters()['width'] = 1500
        exporter.parameters()['height'] = 1000
        exporter.export(file_path)

        # Save recorded audio to .wav and .csv
        global audio_list
        audio_arr = np.concatenate(audio_list)

        file_path = os.path.join(directory, "recorded_audio.wav")
        sf.write(file_path, audio_arr, self.sample_rate)

        file_path = os.path.join(directory, "recorded_audio.csv")
        np.savetxt(file_path, audio_arr, delimiter=",")

        # Update/reset the ui elements
        self.update_run_avgs()
        self.runs_table_data.append(
                ['-']*len(self.runs_table_data[0]))
        self.runs_table_model.layoutChanged.emit()
        self.notes_box.clear()
        self.save_run_btn.setEnabled(False)
        self.record_btn.setEnabled(True)
        self.exp_fit_eqn.setText("")

        # Generate the report. This is overwritten each time.
        self.create_report()

    def save_as(self):
        '''
        Standard "Save As" dialog box for selecting the directory
        to save the generated files to.

        Directory is saved as follows:
            "<user/selected/path>/<name_box text current-time>/"
        '''

        dialog = QtWidgets.QFileDialog(self)
        dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        dialog.ShowDirsOnly = True

        if dialog.exec_():
            directory = dialog.selectedFiles()[0]

            directory = os.path.join(
                    directory, 
                    datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) \
                    + "_" + str.lstrip(self.name_box.text())
            
            os.mkdir(directory)

            self.msg("Saving files to: " + directory)
            self.save_dir = directory

    def select_input_device(self):
        '''
        When the "Input Device" dropdown is clicked, change to the new
        input device. This includes handling the sounddevice stream to
        gracefully point it to the new device.
        '''

        if self.stream is not None:
            self.stream.abort()

        # Set the default device to be current text
        sd.default.device = self.input_dropdown.currentText()
            
        # This occasionally throws an error on windows due to duplicate
        # devices showing up. It doesn't actually cause problems
        try:
            self.sample_rate = \
                int(sd.query_devices(sd.default.device)['default_samplerate'])
        except ValueError:
            pass

        # Global ndarry to plot live recorded audio. Adjust the window for
        # for a given sample rate to always plot 2 seconds of audio.
        global audio_wfrm_data
        audio_wfrm_data = np.zeros(self.sample_rate*2)

        dev = self.input_dropdown.currentText()

        try:
            self.stream = sd.InputStream(device=dev,
                                         callback=audio_callback,
                                         channels=1)
            self.stream.start()
        except ValueError:
            pass

        self.msg("Set input device to " + dev + \
                ". Sample rate: " + str(self.sample_rate) + "Hz")

    def hide_sample_params(self):
        '''
        Enable or disable input boxes according to the geometry type
        so that users know which parameters are required.
        '''

        match self.geometry_dropdown.currentText():
            case "Rectangle":
                self.length_box.setEnabled(True)
                self.thickness_box.setEnabled(True)
                self.width_box.setEnabled(True)
                self.diameter_box.setEnabled(False)
            case "Rod":
                self.length_box.setEnabled(True)
                self.thickness_box.setEnabled(False)
                self.width_box.setEnabled(False)
                self.diameter_box.setEnabled(True)
            case "Disc":
                self.length_box.setEnabled(False)
                self.thickness_box.setEnabled(True)
                self.width_box.setEnabled(False)
                self.diameter_box.setEnabled(True)

    def create_geometry(self):
        '''
        Check the input values and handle exceptions thrown by
        by the geometry class.

        Estimated flexural and torsional frequencies are calculated
        if estimates for E and G are provided.

        Displays the nodal spacing to the ui.

        Finally, this also enables the "Record" button and moves
        the FFT vline to the estimated flexural freq.
        '''

        L = self.length_box.value()
        t = self.thickness_box.value()
        b = self.width_box.value()
        m = self.mass_box.value()
        d = self.diameter_box.value()

        match self.geometry_dropdown.currentText():
            case "Rectangle":
                if (L and t and b and m) == 0:
                    self.msg("Error: Missing sample parameter")
                    return
                try:
                    self.sample_geometry.rect(L, t, b, m)
                    self.msg("Created rectangular geometry")
                except ValueError as e:
                    self.msg(str(e))

            case "Rod":
                if (L and d and m) == 0:
                    self.msg("Error: Missing sample parameter")
                    return
                try:
                    self.sample_geometry.rod(L, t, d, m)
                    self.msg("Created rod geometry")
                except ValueError as e:
                    self.msg(str(e))

            case "Disc":
                if (t and d and m) == 0:
                    self.msg("Error: Missing sample parameter")
                    return
                try:
                    self.sample_geometry.disc(t, d, m)
                    self.msg("Warning, node spacing not validated")
                    self.msg("Created disc geometry")
                except ValueError as e:
                    self.msg(str(e))
                    return
            case _:
                # This shouldn't be possible but good to have
                self.msg("Error: Missing geometry type")
                return

        # Help user set spacing of supports
        fn, ns = self.sample_geometry.node_spacing()
        self.first_node_box.setValue(fn)
        self.node_spacing_box.setValue(ns)

        # Show estimates of resonant freq if moduli available
        # Input values are GPa
        E = 1E9*self.e_mod_est.value()
        G = 1E9*self.g_mod_est.value()

        if E != 0:
            f_f = self.sample_geometry.flexural_freq(E)
            self.flexural_f_est.setValue(int(f_f))

        if G != 0:
            t_f = self.sample_geometry.torsional_freq(G)
            self.torsional_f_est.setValue(int(t_f))

        # Disable all inputs in the sample parameters section so they
        # can't be changed. The user must reopen the program for a new sample.
        self.name_box.setEnabled(False)
        self.geometry_dropdown.setEnabled(False)
        self.length_box.setEnabled(False)
        self.width_box.setEnabled(False)
        self.thickness_box.setEnabled(False)
        self.diameter_box.setEnabled(False)
        self.mass_box.setEnabled(False)
        self.e_mod_est.setEnabled(False)
        self.g_mod_est.setEnabled(False)
        self.create_geo_btn.setEnabled(False)

        self.record_btn.setEnabled(True)

if __name__ == "__main__":

    # TODO can these globals be moved into the mainwindow
    # Or rather, can the callback be in the mainwindow
    global audio_wfrm_data
    audio_wfrm_data = [0]

    # Global list to buffer audio before it is written to file.
    global audio_list
    audio_list = []

    # Global flag to control recording of data to file.
    global recording_flag
    recording_flag = False

    app = QtWidgets.QApplication([])
    main = MainWindow()
    main.show()
    app.exec()
