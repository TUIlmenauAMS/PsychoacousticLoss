# Imports
import numpy as np
from numpy.fft import fft, ifft


# Noise mT in Silence
def noisefromdBSpectrum(spec, fs):
    # produces noise according to the dB spectrum given in spec
    # Spectrum goes from frequency 0 up to Nyquist frequency
    specvoltage = 10.0**(spec/20.0)
    # produce 40 blocks of sound:
    # Noise in the range of -1...+1, and Multiply noise with spectrum:
    noisespec = specvoltage*(np.random.rand(len(specvoltage))-0.5)*2
    # make spectrum symmetric for ifft:
    # trick: Append zeros to fill up negative frequencies in upper half of DFT, then take real part of inverse transform:
    noisespec = np.concatenate((noisespec, np.zeros(len(noisespec))))
    noise_ifft = np.real(ifft(noisespec, norm='ortho'))
    noise = noise_ifft
    return (noise, fs)
