from pyRFDA import RFDA
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf # For testing
from scipy.io.wavfile import write

# Define geometry
rfda = RFDA()
rfda.rect(L=(114.36/1000), \
          t=(3.12/1000),   \
          b=(25.44/1000),  \
          m=(71.3/1000))

'''
# Record sample vibration using microphone and wait until complete
fs = 384000
rec_time = 5
vibration = sd.rec(int(fs*rec_time), samplerate=fs, channels=1)
sd.wait()

# TODO detect tap and trim silence at start of recording
'''

'''
  Audacity recording at 384000 sample rate.
  Sample file has a resonance at 1281 Hz according to audacity,
  and this steel should have an elastic modulus of around 200 GPa (ASTM)
'''
s, data = sf.read('114x25.4x3.12mm Hot Rolled Mild Steel Bar.wav')

damping = rfda.damping_coeff(s)

# Estimate f_f then search for that peak to get true f_f
f_f = rfda.flexural_freq(E=200E9)
print("Estimated flexural freq: %4.0f" % f_f)

f_f = rfda.find_peak(s, 384000, f_f*0.8, f_f*1.2)
print("Found peak at %4.0f Hz" % f_f)

# Use f_f to get the elastic modulus
E = rfda.elastic_modulus(f_f)

print("E: % 3.1f GPa" % (E/1E9))
print("Damping: % 1.5f" % damping)
