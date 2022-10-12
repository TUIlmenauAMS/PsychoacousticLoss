# Imports
import torch
import torch.nn as nn
import math


class WMSELoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_fft=2048, hop_length=None, eps=1e-7):
        super(WMSELoss, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.eps = eps

    def forward(self, x, y):
        n_blocks = math.floor(y.shape[0]/1024)
        size = 1024*n_blocks
        x_blocks = torch.reshape(x[:size], (n_blocks, 1024))
        y_blocks = torch.reshape(y[:size], (n_blocks, 1024))
        y_blocks = y_blocks.numpy()
        Weighted_matrix = []
        sr = 44100
        gamma_1 = 0.92
        gamma_2 = 0.60
        Coef_num = 12
        for bt in range(y_blocks.shape[0]):
            audio_np = y_blocks[bt, :]
            import librosa
            import numpy as np
            lpc_coeff = librosa.lpc(audio_np, order=Coef_num)
            a_gamma1 = np.zeros((1, Coef_num))
            a_gamma1[0, 0] = 1
            a_gamma2 = np.zeros((1, Coef_num))
            a_gamma2[0, 0] = 1
            worn = 1024
            for k1 in range(1, Coef_num):
                a_gamma1[0, k1] = lpc_coeff[k1] * np.power(gamma_1, k1)
                a_gamma2[0, k1] = lpc_coeff[k1] * np.power(gamma_2, k1)
                import scipy.signal as signal
                w1, freqz_amp_values = signal.freqz(
                    a_gamma2[0, :], a_gamma1[0, :], worN=worn, fs=sr)
                Weighted_matrix.append(
                    torch.from_numpy(np.abs(freqz_amp_values)))
        STFT_Original = torch.stft(
            x.squeeze(), n_fft=2046, return_complex=True)
        STFT_Reconstructed = torch.stft(
            y.squeeze(), n_fft=2046, return_complex=True)
        criterion = nn.MSELoss(reduction='none')
        loss = criterion(torch.abs(STFT_Original),
                         torch.abs(STFT_Reconstructed))
        loss = torch.mean(loss, dim=1)
        loss = torch.div(loss, torch.stack(Weighted_matrix))
        loss = torch.mean(loss)
        return loss


if __name__ == "__main__":
    import librosa
    audio_wav, sr = librosa.load('Slash_Anastasia.wav', sr=None)
    audio_aac, sr = librosa.load('Slash_Anastasia.aac', sr=None)
    audio_mp3, sr = librosa.load('Slash_Anastasia.mp3', sr=None)
    audio_aac = torch.from_numpy(audio_aac[:audio_wav.shape[-1], ])
    audio_mp3 = torch.from_numpy(audio_mp3[:audio_wav.shape[-1], ])
    audio_wav = torch.from_numpy(audio_wav)
    wmse_loss = WMSELoss()
    print(wmse_loss(audio_mp3, audio_wav))
