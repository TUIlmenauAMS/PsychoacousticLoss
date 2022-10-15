# Imports
import torch
import torch.nn as nn


class LSDLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_fft=2048, hop_length=None, eps=1e-7):
        super(LSDLoss, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.eps = eps

    def forward(self, x, y):
        x_stft = torch.stft(x, n_fft=self.n_fft,
                            hop_length=self.hop_length, return_complex=True)
        y_stft = torch.stft(y, n_fft=self.n_fft,
                            hop_length=self.hop_length, return_complex=True)
        loss = torch.sqrt(((torch.log10(torch.abs(x_stft)**2 + self.eps) - torch.log10(
            torch.abs(y_stft)**2 + self.eps))**2).mean(dim=1)).mean(dim=0)
        return loss


if __name__ == "__main__":
    import librosa
    audio_wav, sr = librosa.load('Slash_Anastasia.wav', sr=None)
    audio_aac, sr = librosa.load('Slash_Anastasia.aac', sr=None)
    audio_mp3, sr = librosa.load('Slash_Anastasia.mp3', sr=None)
    audio_aac = torch.from_numpy(audio_aac[:audio_wav.shape[-1], ])
    audio_mp3 = torch.from_numpy(audio_mp3[:audio_wav.shape[-1], ])
    audio_wav = torch.from_numpy(audio_wav)
    lsd_loss = LSDLoss()
    print(lsd_loss(audio_mp3, audio_wav))
