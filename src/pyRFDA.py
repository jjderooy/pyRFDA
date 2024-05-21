import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class RFDA:

  def rect(self, L, t, b, m):
      '''
      Inputs:
        L:  [m]  Length
        t:  [m]  Thickness
        b:  [m]  Width
        m:  [kg] Mass

      Desc:
        Define a rectangular geometry for RFDA. Error checks geometry
        dimensions to ensure it meets criteria for RFDA.
      '''

      if L/t < 20:
          raise ValueError("For rect geometry L/t must be greater than 20")

      self._shape = 'rect'
      self._L = L 
      self._t = t 
      self._b = b 
      self._m = m 

  def rod(self, L, t, d, m):
      '''
      Inputs:
        L:  [m]  Length
        t:  [m]  Thickness
        d:  [m]  Diameter
        m:  [kg] Mass

      Desc:
        Define a cylindrical geometry for RFDA. Error checks geometry
        dimensions to ensure it meets criteria for RFDA.
      '''

      if L/d < 20:
          raise ValueError("For rect geometry L/t must be greater than 20")

      self._shape = 'rod'
      self._L = L # [m]  Length
      self._t = t # [m]  Thickness
      self._d = d # [m]  Diameter
      self._m = m # [kg] Mass

  def disc(self, t, d, m):
      '''
      Inputs:
        t:  [m]  Thickness
        d:  [m]  Diameter
        m:  [kg] Mass

      Desc:
        Define a disc (like a coin) geometry for RFDA. Error checks
        geometry dimensions to ensure it meets criteria for RFDA.
      '''

      raise ValueError("Not implemented! Add error check on disk!")

      self._shape = 'disc'
      self._t = t # [m]  Thickness
      self._d = d # [m]  Diameter
      self._m = m # [kg] Mass

  def flexural_freq(self, E):
      '''
      Inputs:
        E:   [Pa] Elastic modulus of material.

      Outputs:
        f_f: [Hz] Resonant frequency of the flexural mode of vibration

      Desc:
        The flexural resonant frequency f_f of a sample can be computed from
        the elastic modulus E of the  material and its geometry. More info:
        https://en.wikipedia.org/wiki/Impulse_excitation_technique
      '''
      
      if self._shape == 'rect':

          # Correction factor
          T = 1 + 6.585*np.power((self._t/self._L), 2)

          f_f = np.sqrt(E/(T*0.9465)* \
              (self._b/self._m)*np.power(self._t/self._L,3))
          return f_f

      elif sample_geometry._shape == 'rod':
          raise NotImplementedError("Not implemented for rod!")
      elif sample_geometry._shape == 'disc':
          raise NotImplementedError("Not implemented for disk!")

  def torsional_freq(self, G):
      '''
      Inputs:
        G:   [Pa] Shear modulus of material.

      Outputs:
        f_f: [Hz] Resonant frequency of the torsional mode of vibration

      Desc:
        The torsional resonant frequency f_t of a sample can be computed from
        the shear modulus G of the  material and its geometry. More info:
        https://en.wikipedia.org/wiki/Impulse_excitation_technique
      '''

      if self._shape == 'rect':
          
          # Correction factor
          R = ((1+np.power(self._b/self._t, 2)) / \
              (4-2.521*(self._t/self._b) * \
              (1-(1.991/(np.exp(np.pi*self._b/self._t) + 1))))) * \
              (1+0.00851*np.power(self._b/self._L, 2)) - \
              (0.060*np.power(self._b/self._L, 3/2)*np.power(self.b/self.t-1,2))

          f_t = np.sqrt((G/R)*(self._b*self._t/(4*self._L*self._m)))
          return f_t

      elif sample_geometry._shape == 'rod':
          raise NotImplementedError("Not implemented for rod!")
      elif sample_geometry._shape == 'disc':
          raise NotImplementedError("Not implemented for disk!")
      else:
          raise ValueError("_shape must be one of 'rect', 'rod', 'disc'")

  def elastic_modulus(self, f_f):
      '''
      Input:
        f_f: [Hz] Resonant frequency of the flexural mode of vibration

      Outputs:
        E: [Pa] Elastic modulus of the sample's material

      Desc:
        The elastic modulus E of a material can be computed from the
        resonant frequency of the flexural mode of vibration. More info:
        https://en.wikipedia.org/wiki/Impulse_excitation_technique
      '''

      # Correction factor
      T = 1 + 6.585*np.power((self._t/self._L), 2)
      E = 0.9465*(self._m*np.power(f_f,2)/self._b)*(np.power(self._L/self._t,3))*T
      return E

  def upper_envelope(self, s):
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
      #x = y[[i+np.argmax(s[y[i:i+1]]) for i in range(0,len(y),1)]]

      return (x,y)

  def damped_exp(self, x, b, c):
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

  def damping_coeff(self, s, guess=0.005):
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
      envelope = self.upper_envelope(s)

      # Fit curve using scipy TODO don't pollute namespace like this
      popt, pocv = curve_fit(self.damped_exp, \
                             envelope[0],     \
                             envelope[1],     \
                             p0=[guess, 0])

      damping = popt[0] # b coeff
      return damping
    
  def find_peak(self, s, samp_rate, min_freq, max_freq):
      '''
      Inputs:
        s: Input audio signal of which to analyze
        samp_rate: [/s] Sample rate audio signal was recorded at
        min_freq:  [Hz] Minimum of frequency range to seach for peak
        max_freq:  [Hz] Maximum of frequency range to seach for peak

      Outputs:
        peak_freq: [Hz] Frequncy of max amplitude in range.
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

      plt.plot(cropped_freq, cropped_fourier)
      plt.show()

      # Check that the peak really is a peak and not just the largest value
      # in a sea of similarly sized peaks.
      if peak_freq < 5*np.mean(cropped_fourier):
          print("Warning: Peak frequency is not dominant on given range.")

      return peak_freq
