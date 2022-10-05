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

# Demo signals

# Shaped Noise
f = np.linspace(0.0001, fs//2, n_fft//2)
LTQ = np.clip((3.64*(f/1000.)**-0.8 - 6.5*np.exp(-0.6 *
              (f/1000. - 3.3)**2.)+1e-3*((f/1000.)**4.)), -20, 60)
LTQ = LTQ-60
shaped_noise_mT, fs = noisefromdBSpectrum(LTQ, fs)
shaped_noise_mT /= np.abs(shaped_noise_mT).max()

x = x1.float()
y = (x1 + x2 + x3 + +x4 + shaped_noise_mT).float()

# Signals X and Y in Frequency
freqs = librosa.fft_frequencies(sr=fs, n_fft=n_fft)
x_fft_mag = 20*torch.log(torch.abs(torch.fft.rfft(x, norm='ortho'))+1e-12)
y_fft_mag = 20*torch.log(torch.abs(torch.fft.rfft(y, norm='ortho'))+1e-12)

# Signal
audio_wav, sr_wav = librosa.load(
    'Slash_Anastasia.wav', sr=None, duration=20, offset=36)
audio_aac, sr_aac = librosa.load(
    'Slash_Anastasia.aac', sr=None, duration=20, offset=36)
audio_mp3, sr_mp3 = librosa.load(
    'Slash_Anastasia.mp3', sr=None, duration=20, offset=36)
audio_aac = torch.from_numpy(audio_aac[:audio_wav.shape[-1], ])
audio_mp3 = torch.from_numpy(audio_mp3[:audio_wav.shape[-1], ])
audio_wav = torch.from_numpy(audio_wav)
audio_wav.requires_grad = True
audio_mp3.requires_grad = True
audio_aac.requires_grad = True
wav_fft_mag = 20 * \
    torch.log(torch.abs(torch.fft.rfft(audio_wav, norm='ortho', n=2**17))+1e-12)
mp3_fft_mag = 20 * \
    torch.log(torch.abs(torch.fft.rfft(audio_mp3, norm='ortho', n=2**17))+1e-12)
aac_fft_mag = 20 * \
    torch.log(torch.abs(torch.fft.rfft(audio_aac, norm='ortho', n=2**17))+1e-12)
