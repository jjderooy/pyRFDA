import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf # For testing
from scipy.io.wavfile import write
from scipy.optimize import curve_fit

# From https://stackoverflow.com/a/60402647
def h_envelopes_idx(s, dmin=1, dmax=1, split=False):
    '''
    Input:
      s: 1d-array, data signal from which to extract high envelopes

      dmin, dmax: int, optional, size of chunks, use this if the size
      of the input signal is too big

      split: bool, optional, if True, split the signal in half along
      its mean, might help to generate the envelope in some cases

    Output:
      lmax : high envelope idx of input signal s

    Desc:
      The amplitude of a damped vibration decays exponentially, where
      the coeff in the exponential is related to the damping. This
      function computes the indices of the envelope points of the input
      array (vibration signal) which are used later to fit an exponential
      curve to obtain this coeff.
    ''' 

    # Big ballin one liner to compute local maximums
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1

    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s)
        # pre-sorting of local max based on relative position with respect to s_mid
        lmax = lmax[s[lmax]>s_mid]

    # Global max of dmax-chunks of locals max
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]

    return lmax

# Definition of curve fit for scipy optimization
# TODO Might have to remove c term or translate the signal down by 1/2 the median
def envelope_fit(x, a, b, c):
  return a*np.exp(-b*x) + c

fs = 44100
rec_time = 5

# Record sample vibration using microphone and wait until complete
#vibration = sd.rec(int(fs*rec_time), samplerate=fs, channels=1)
#sd.wait()

# Testing
vibration, data = sf.read('sample_test.wav')

# Normalize the data
vibration /= np.max(vibration)

# Guess to help optimizer converge
fit_guess  = [1, 0.01, 0]

# Bounds on parameters a, b, c in fit
fit_bounds = (0, [1, 0.1, 0.5])

# Get envelope curve points
envelope_x = h_envelopes_idx(vibration)
envelope_y = vibration[envelope_x]

# Obtain envelope parameters and fit envelope to points
if len(envelope_x) == 0:
  print("Error: Unable to construct envelope. Array size is 0")
else:
  popt, pocv = curve_fit(\
      envelope_fit,      \
      envelope_x,        \
      envelope_y,        \
      p0=fit_guess,      \
      bounds=fit_bounds)

  fit = envelope_fit(envelope_x, *popt)

  print('fit: a=%1.5f, b=%1.5f, c=%1.5f' % tuple(popt))

  plt.plot(vibration)
  plt.plot(envelope_x, envelope_y)
  plt.plot(envelope_x, fit)
  plt.show()
