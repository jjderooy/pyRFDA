import numpy as np
from scipy.optimize import curve_fit

def elastic_modulus(geo, f_f):
    '''
    Input:
      geo: Instance of geometry class for the sample under analysis
      f_f: [Hz] Resonant frequency of the flexural mode of vibration

    Outputs:
      E: [Pa] Elastic modulus of the sample's material

    Desc:
      The elastic modulus E of a material can be computed from the
      resonant frequency of the flexural mode of vibration. More info:
      https://en.wikipedia.org/wiki/Impulse_excitation_technique
    '''

    # Correction factor
    T = 1 + 6.585*np.power((geo._t/geo._L), 2)
    E = 0.9465*(geo._m*np.power(f_f,2)/geo._b)*(np.power(geo._L/geo._t,3))*T
    return E

def upper_envelope(s):
    '''
    Input:
      s: 1d audio array from which to extract upper envelope

    Output:
      lmax : Tuple (x,y) of 1d-arrays of the upper envelope points of s

    Desc:
      The amplitude of a damped vibration decays exponentially, where
      the coeff in the exponential is related to the damping. This
      function computes the upper envelope points of the input array
      (vibration signal).

      Modified from https://stackoverflow.com/a/60402647
    ''' 

    # Big ballin one liner to compute points of local maximums
    x = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1
    y = s[x]

    return (x,y)

def damped_exp(x, b, c):
    '''
    Inputs:
      x: Independent variable
      b: Damping coeff
      c: DC bias

    Outputs:
      y: Computation of y = exp(-b*x) + c

    Desc:
      For use with curve_fit from scipy
    '''
    return np.exp(-b*x) + c

def damping_coeff(s, guess=0.005):
    '''
    Inputs:
      s: 1d audio array from which to compute the damping coeff
      guess: (optional) Estimate of the damping coeff for solver

    Outputs:
      damping: [-] b coeff of exponential decay of audio signal

    Desc:
      The damping coeff of a material describes how a quickly its
      vibrations are suppressed by internal friction. Mathematically,
      the amplitude of the vibrations of a damped material decay
      with the form 'a*exp(-b*x) + c' where a,b,c are coefficients
      describing the material. The coefficient 'b' is the damping coeff.

      The input audio is normalized so that 'a' = 1 and is thus ommited.
      Due to the nature of microphones, it appears that c is always close
      to 0, however the parameter is left in anyways.
    '''

    # Normalize the array
    s /= np.max(s)
    envelope = upper_envelope(s)

    # TODO compute some kind of error value here to warn if the data is noisy

    # Fit curve using scipy
    popt, pocv = curve_fit(damped_exp,   \
                           envelope[0],  \
                           envelope[1],  \
                           p0=[guess, 0])

    damping = popt[0] # b coeff
    return damping
  
def find_peak(s, samp_rate, min_freq, max_freq):
    '''
    Inputs:
      s: Input audio signal of which to analyze
      samp_rate: [/s] Sample rate audio signal was recorded at
      min_freq:  [Hz] Minimum of frequency range to seach for peak
      max_freq:  [Hz] Maximum of frequency range to seach for peak

    Outputs:
      peak_freq:  [Hz] Frequncy of max amplitude in range.
      prominence: [-]  Amplitude of peak_freq divided by the average
                       over the range of frequencies. If prominence
                       is small ~2, the peak may just be the largest
                       value in some noisy data.
    '''


    '''
      Take fft then correct frequencies using sample rate
      Just a note since the numpy docs aren't super clear. rfftfreq()
      returns an array that corresponds to the [Hz] frequency value
      for each magnitude in the array returned by rfft().
    '''
    fourier = np.abs(np.fft.rfft(s))
    freq    = np.fft.rfftfreq(s.size, 1.0/samp_rate)

    # Look up index corresponding to the freq min and max range
    indices = np.where((freq > min_freq) & (freq < max_freq))[0]

    # Extract freq and magnitudes in that range
    cropped_freq = freq[indices]
    cropped_fourier = fourier[indices] 

    # Maximum peak in the expected range is the resonant peak of the sample
    peak_freq = cropped_freq[np.argmax(cropped_fourier)]

    # Check that the peak really is a peak and not just the largest value
    # in a sea of similarly sized peaks.
    prominence = peak_freq/np.mean(cropped_fourier)
    if prominence < 2:
        print("Warning: Peak frequency is not prominent on given range.")

    return (peak_freq, prominence)

