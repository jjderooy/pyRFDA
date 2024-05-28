from PyQt5 import QtGui, QtWidgets, uic, QtCore
from geometry import Geometry
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import pyqtgraph as pg
import numpy as np
import rfda 

# TODO
# Export button that saves graphs as pngs and csv.
# Report button that creates a PDF of the sample parameters etc.
# ^ maybe html and then print that? Or LaTeX????
# Damping coeff (b coeff) and inverse Q factor
# Multiple successive runs in tabular form
# Briefly look at selecting peak in fft rather than estimating E,G
# ^ look at 'LinearRegionItem'
# Input device selection using sd.query_device()
# Measure length of time for exponential decay to background noise

def audio_callback(indata, frames, time, status):
    '''
    This is called (from a separate thread) for each audio block. It
    consumes the audio (indata) from the microphone input stream, and
    rolls it into a global array for access outside the stream thread.

    By rolling like this, the program only stores that last few seconds
    to reduce plotting complexity.

    Additionally, if actively recording, buffers data into a list of 
    ndarrays which will later be FFT'd etc.
    '''
    global audio_wfrm_data
    global audio_list
    global recording_flag

    # Roll the data into array from right to left for plotting
    shift = len(indata)
    audio_wfrm_data = np.roll(audio_wfrm_data, -shift, axis=0)
    #audio_wfrm_data[-shift:] = indata[0] # Downsample (fast)
    audio_wfrm_data[-shift:] = np.concatenate(indata) # No downsample (slow)

    if recording_flag == True:
        audio_list.append(np.concatenate(indata))


class MainWindow(QtWidgets.QMainWindow):

    # To be populated with plotLine object in setup_graphs()
    audio_wfrm_line = None
    audio_wfrm_fft_line = None

    # Movable vertical line for selecting fft peak
    fft_peak_vline = pg.InfiniteLine(movable=True)

    # Movable horizontal line for selecting audio impulse threshold
    audio_trig_hline = pg.InfiniteLine(movable=True,
                                       angle=0,
                                       pos=0.1)

    # Values of the sample are populated in the ui
    sample_geometry = Geometry()

    # Input stream from sounddevice. Initialized in select_input_device()
    stream = None

    def __init__(self, *args, **kwargs):

        # Load external ui file. Edit this ui file in 'Qt Designer'
        super(MainWindow, self).__init__(*args, **kwargs)
        uic.loadUi('RFDA.ui', self)

        # Prepare the ui
        self.setup_graphs()
        self.setup_inputs()
        self.hide_sample_params()

        # Timer to update live waveform graph
        self.update_wfrm_timer = QtCore.QTimer()
        self.update_wfrm_timer.setInterval(int(1000/30)) # 30 Hz
        self.update_wfrm_timer.timeout.connect(self.update_audio_wfrm)
        self.update_wfrm_timer.start()

        # Timer to check for impulses in incoming audio
        self.check_impulse_timer = QtCore.QTimer()
        self.check_impulse_timer.setInterval(200) # 5 Hz
        self.check_impulse_timer.timeout.connect(self.handle_recording)
        self.check_impulse_timer.start()
        
        # Welcome message
        self.msg("Welcome to pyRFDA companion!")
        self.msg("Start by defining sample type and parameters," + \
                 " then click the \"Create Sample\" button.")


    def setup_graphs(self):
        # Live microphone audio graph
        self.audio_wfrm_line = \
            self.audio_wfrm.plot(audio_wfrm_data, 
                                 autodownsample=True)
        self.audio_wfrm.setYRange(min=-0.3, max=0.3)
        self.audio_wfrm.showGrid(x=True, y=True)
        self.audio_wfrm.setTitle("Live Microphone Audio", size="12pt")
        self.audio_wfrm.setLabel("left", "Intensity")
        self.audio_wfrm.setLabel("right", "") # For padding
        self.audio_wfrm.setLabel("bottom", "Sample")
        self.audio_wfrm.addItem(self.audio_trig_hline)

        # FFT of audio recording after impulse is detected
        self.audio_wfrm_fft_line = self.audio_wfrm.plot()
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

        max_sample = np.max(audio_wfrm_data)
        trig = self.audio_trig_hline.value()

        # TODO setting max_sample like this acts as a sort of box filter
        # on the trigger. Is this desirable?

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

            # Write to file
            with sf.SoundFile("test.wav",
                              mode='w',
                              samplerate=44100,
                              channels=1) as file:
                file.write(audio_arr)

            # Compute sample data from run and plot it
            self.process_run(audio_arr)

            self.record_btn.setChecked(False)
            self.record_btn.setText("Record")
            self.record_btn.setEnabled(True)
            self.msg("Finished recording")

    def process_run(self, audio_arr):
        '''
        Compute E, G, damping etc from the recorded run and pass them 
        to append_run() so they show up on the table.
        '''

        # TODO don't always want to immediately add everything to the table
        # since it takes two setups to get all the data (E and G)

        fft_arr = np.abs(np.fft.rfft(audio_arr))
        freqs_arr = np.fft.rfftfreq(audio_arr.size, 1.0/44100)
        
        # Graph the FFT
        # TODO This call to clear removes the vline
        #self.audio_wfrm_fft.clear()
        self.audio_wfrm_fft.plot(freqs_arr,
                                 fft_arr,
                                 autodownsample=True,
                                 downsampleMethod='peak')

        # Crop the range to [0, 20kHz] if necessary
        if np.max(freqs_arr) > 20000:
            self.audio_wfrm_fft.setXRange(0, 20000)
        
        # Estimate the peak value
        peak_est = self.fft_peak_vline.value()
        indices = np.where(
                (freqs_arr > peak_est - 100) & (freqs_arr < peak_est + 100))
        resonant_freq = freqs_arr[np.argmax(fft_arr)]

        # TODO one of these will be wrong for a given run type
        # Compute elastic and shear mod from peak
        E = rfda.elastic_modulus(self.sample_geometry, resonant_freq)
        G = rfda.shear_modulus(self.sample_geometry, resonant_freq)

        # b coeff of exponential fit
        damping_coeff = rfda.damping_coeff(audio_arr)

        # Inverse quality factor
        Q_inv = damping_coeff/(np.pi*resonant_freq)

        run_dict = {}
        run_dict["loss_rate"] = 0.0 # TODO implement
        run_dict["damping"]   = damping_coeff
        run_dict["freq"] = resonant_freq
        run_dict["time"] = audio_arr.size/44100
        run_dict["Q-1"]  = Q_inv
        run_dict["E"]    = E/1E9
        run_dict["G"]    = G/1E9

        self.append_run(run_dict)

    def append_run(self, row_data):
        '''
        Create a new row in the runs_table, and add row_data to it.
        row_data is a dict with the following keys:
        "freq", "time", "E", "G", "damping", "loss_rate", "Q-1"
        '''

        # It's a bit messy to do it this way but it allows better
        # control over the formatting.

        row_pos = self.runs_table.rowCount()
        self.runs_table.insertRow(row_pos)
        items = []

        # Create items for the table. Each cell requires a QTableWidgetItem
        items.append(QtWidgets.QTableWidgetItem(
                                self.name_box.text()))
        items.append(QtWidgets.QTableWidgetItem(
                                self.sample_geometry.shape()))
        items.append(QtWidgets.QTableWidgetItem(
                                "{:.2f}".format(row_data["freq"])))
        items.append(QtWidgets.QTableWidgetItem(
                                "{:.2f}".format(row_data["time"])))
        items.append(QtWidgets.QTableWidgetItem(
                                "{:.2f}".format(row_data["E"])))
        items.append(QtWidgets.QTableWidgetItem(
                                "{:.2f}".format(row_data["G"])))
        items.append(QtWidgets.QTableWidgetItem(
                                "{:.5f}".format(row_data["damping"])))
        items.append(QtWidgets.QTableWidgetItem(
                                "{:.5}".format(row_data["Q-1"])))

        # Add each item to the table
        for i in range(0, len(items)):
            self.runs_table.setItem(row_pos, i, items[i])

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

        # Options for sample geometry
        self.geometry_dropdown.addItems(["Rectangle", "Rod", "Disc"])
        self.geometry_dropdown.activated.connect(
                self.hide_sample_params)
        self.create_geo_btn.clicked.connect(self.create_geometry)

        # Configure the runs table
        header_list = [
                "Name ",
                "Type ",
                "Freq [Hz] ",
                "Time [s] ",
                "E [GPa] ",
                "G [GPa] ",
                "Q^-1 ",
                "Damping ",
                "Notes "]
        self.runs_table.setColumnCount(len(header_list))
        self.runs_table.setHorizontalHeaderLabels(header_list)
        header = self.runs_table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)

    def msg(self, s):
        # Shorter wrapper for sending status message to the text box
        s = "> " + s
        self.status_msg_box.appendPlainText(s)

    ## SIGNALS ##

    def select_input_device(self):
        '''
        When the "Input Device" dropdown is clicked, change to the new
        input device. This includes handling the sounddevice stream to
        gracefully point it to the new device.
        '''

        if self.stream is None:
            self.stream = sd.InputStream(callback=audio_callback, channels=1)

        self.stream.stop()

        # Set the default device to be current text
        sd.default.device = self.input_dropdown.currentText()
        dev = self.input_dropdown.currentText()
        self.msg("Set dev to " + dev)

        self.stream.start()

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

        Additionally, the estimated flexural and torsional
        frequencies are calculated if estimates for E and G
        are provided.

        Finally, this also enable the "Record" button
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
                    self.msg("Created disc geometry")
                except ValueError as e:
                    self.msg(str(e))
            case _:
                # This shouldn't be possible but good to have
                self.msg("Error: Missing geometry type")
        
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

        self.record_btn.setEnabled(True)
            

if __name__ == "__main__":

    # TODO can these globals be moved into the mainwindow
    # Global ndarry to plot live recorded audio.
    global audio_wfrm_data
    audio_wfrm_data = np.zeros(100000) # TODO adjust to sample rate

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

'''
  Audacity recording at 384000 sample rate.
  Sample file has a resonance at 1281 Hz according to audacity,
  and this steel should have an elastic modulus of around 200 GPa (ASTM)
s, data = sf.read('114x25.4x3.12mm Hot Rolled Mild Steel Bar.wav')

damping = rfda.damping_coeff(s)

# Estimate f_f then search for that peak to get true f_f
f_f = bar.flexural_freq(E=200E9)
print("Estimated flexural freq: %4.0f" % f_f)

# Search for peak near est
f_f, prom = rfda.find_peak(s, 384000, f_f*0.8, f_f*1.2)
print("Found peak at %.0f Hz with prominence %.0f on specified range." % (f_f, prom)

# Use f_f to get the elastic modulus
E = rfda.elastic_modulus(bar, f_f)

print("E: % 3.1f GPa" % (E/1E9))
print("Damping: % 1.5f" % damping)
'''
