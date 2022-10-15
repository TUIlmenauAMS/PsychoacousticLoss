# Imports
import torch
from demo_aux_functions import noisefromdBSpectrum
import numpy as np
import librosa
import math
torch.pi = math.pi

# Parameters
fs = 48000  # [Hz]
maxfreq = fs//2
duration = 2  # [s]
n_fft = 2**17

# Sinusoids
x1 = torch.sin(2*torch.pi/n_fft*2404*torch.arange(n_fft))
x2 = torch.sin(2*torch.pi/n_fft*(2404+1000)*torch.arange(n_fft))*0.004
x3 = torch.sin(2*torch.pi/n_fft*(2404+2000)*torch.arange(n_fft))*0.004
x4 = torch.sin(2*torch.pi/n_fft*(2404+4000)*torch.arange(n_fft))*0.001

# Quantization
factor = 0.95
x_quant = torch.round(x1/factor)*factor

# Demo signals

# Shaped Noise
f = np.linspace(0.0001, fs//2, n_fft//2)
LTQ = np.clip((3.64*(f/1000.)**-0.8 - 6.5*np.exp(-0.6 *
              (f/1000. - 3.3)**2.)+1e-3*((f/1000.)**4.)), -20, 60)
LTQ = LTQ-60
shaped_noise_mT, fs = noisefromdBSpectrum(LTQ, fs)
shaped_noise_mT /= np.abs(shaped_noise_mT).max()

x = x1.float()
x_quant = x_quant.float()
y = (x1 + x2 + x3 + +x4 + shaped_noise_mT).float()

# Signals in Frequency
freqs = librosa.fft_frequencies(sr=fs, n_fft=n_fft)
x_fft_mag = 20*torch.log(torch.abs(torch.fft.rfft(x, norm='ortho'))+1e-12)
x_quant_fft_mag = 20 * \
    torch.log(torch.abs(torch.fft.rfft(x_quant, norm='ortho'))+1e-12)
y_fft_mag = 20*torch.log(torch.abs(torch.fft.rfft(y, norm='ortho'))+1e-12)
