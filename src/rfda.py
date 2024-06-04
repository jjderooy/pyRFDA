import numpy as np
from scipy.optimize import curve_fit
from geometry import Geometry

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
        https://en.wikipedia.org/wiki/
        Impulse_excitation_technique#Young's_modulus
    '''

    match geo.shape():

        case 'rect':
            # Correction factor
            T = 1 + 6.585*np.power((geo._t/geo._L), 2)
            E = 0.9465*(geo._m*np.power(f_f,2)/geo._b)*(np.power(geo._L/geo._t,3))*T
            return E

        case 'rod':
            # Correction factor
            T = 1+4.939*np.power(geo._d/geo._L,2)
            E = 1.6067*(np.power(geo._L,3)/np.power(geo._d,4)) * \
                    geo._m*np.power(f_f,2)*T
            return E

        case 'disc':
            raise NotImplementedError()
        case _:
            raise ValueError("Unknown shape. Must be one of rect, rod, disc.")

def shear_modulus(geo, f_t):
    '''
    Input:
        geo: Instance of geometry class for the sample under analysis
        f_t: [Hz] Resonant frequency of the torsional mode of vibration

    Outputs:
        G: [Pa] Shear modulus of the sample's material

    Desc:
        The shear modulus G of a material can be computed from the
        resonant frequency of the torsional mode of vibration. More info:
        https://en.wikipedia.org/wiki/
        Impulse_excitation_technique#Shear_modulus
    '''

    match geo.shape():

        case 'rect':
            # Correction factor
            R = ((1+np.power((geo._b/geo._t),2) / \
                   (4-2.521*(geo._t/geo._b) * \
                    (1-(1.991/(np.exp(np.pi*(geo._b/geo._t))+1)))))) * \
                 (1+0.00851*np.power(geo._b/geo._L,2)) - \
                 0.060*np.power(geo._b/geo._L,1.5)*np.power((geo._b/geo._t)-1,2) \

            G = (4*geo._L*geo._m*np.power(f_t,2)*R)/(geo._b*geo._t)
            return G

        case 'rod':
            # No correction factor needed
            G = 16*geo._L*geo._m*np.power(f_t,2)/(np.pi*np.power(geo._d,2))
            return G

        case 'disc':
            raise NotImplementedError()
        case _:
            raise ValueError("Unknown shape. Must be one of rect, rod, disc.")
    
def poisson(E, G):

    if (E and G) == 0:
        raise ValueError("E or G cannot be zero.")
    else:
        poisson_ratio = 0.5*(E/G)-1
        return poisson_ratio

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

    # Big ballin one liner to compute indices of local maximums
    x = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1

    # For each box_size interval on x, take the largest corresponding
    # value in s # and add that to filtered_x. 
    # This removes local maxiumums of the noise present at the start 
    # of the waveform due to unwanted harmonics that eventually die out.
    filtered_x = []
    box_size = 20
    for i in range(box_size, x.size, box_size):
        interval = x[i-box_size:i]
        filtered_x.append(interval[np.argmax(s[interval])])

    y = s[filtered_x]
    #y = s[x]

    return (filtered_x,y)
    #return (x,y)

def damped_exp(x, a, b, c):
    '''
    Inputs:
        x: Independent variable
        a: Amplitude
        b: Damping coeff
        c: DC bias

    Outputs:
        y: Computation of y = a*exp(-b*x) + c

    Desc:
        For use with curve_fit from scipy
    '''
    return a*np.exp(-b*x) + c

def exponential_fit(s, guess=0.005):
    '''
    Inputs:
        s: 1d audio array from which to compute the damping coeff
        guess: (optional) Estimate of the damping coeff for solver

    Outputs:
        popt: [-] a, b, c, coefficients of exponential decay envelope

    Desc:
        The damping coeff of a material describes how quickly its
        vibrations are suppressed by internal friction. Mathematically,
        the amplitude of the vibrations of a damped material decay
        with the form 'a*exp(-b*x) + c' where a, b, c are coefficients
        describing the material. The coefficient 'b' is the damping coeff.
    '''

    envelope = upper_envelope(s)

    # Fit curve using scipy
    popt, pocv = curve_fit(damped_exp,   \
                           envelope[0],  \
                           envelope[1],  \
                           p0=[guess, 0, 0])

    return popt
  
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

def inv_Q_factor(b, resonant_freq):
    '''
    Input:
        b: [-] Damping coeff of exponential fit
        resonant_freq: [Hz] Peak freq. of the sample
    Outputs:
        q_inv: [-] Inverse Q factor. See:
        https://en.wikipedia.org/wiki/
        Impulse_excitation_technique#Damping_coefficient
    '''

    q_inv = b/(np.pi*resonant_freq)
    return q_inv

