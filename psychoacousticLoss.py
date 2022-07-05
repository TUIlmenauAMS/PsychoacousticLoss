"""
PyTorch Implementation of a Psychoacoustic Perceptual Loss Function
based on Prof. Dr. -Ing. Gerald Schuller's book:
Filter Banks and Audio Coding: Compressing Audio Signals Using Python
Springer; 1st ed. 2020 edition (24 Sept. 2020), ISBN-10: 3030512487
---------------------------------------------------------------------
Author: renato.profeta@tu-ilmenau.de
Rev: July, 2022
"""

# Imports
import torch
import numpy as np
from numpy.fft import fft, ifft
import torch.nn as nn
from asteroid.losses import SingleSrcMultiScaleSpectral, SingleSrcPMSQE


class PsychoAcousticLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_fft=2048, return_complex=False,
                 n_filts=64, fs=48000, alpha=0.8, mode="mse_time"):
        super(PsychoAcousticLoss, self).__init__()
        self.n_fft = n_fft
        self.return_complex = return_complex
        self.n_filts = n_filts
        self.fs = fs
        self.maxfreq = fs//2
        self.alpha = alpha
        self.mode = mode

    def hz2bark_torch(self, f):
        """ Usage: Bark=hz2bark(f)
            f    : (ndarray)    Array containing frequencies in Hz.
        Returns  :
            Brk  : (ndarray)    Array containing Bark scaled values.
        """
        if not torch.is_tensor(f):
            f = torch.tensor(f)

        Brk = 6. * torch.arcsinh(f/600.)
        return Brk

    def bark2hz_torch(self, Brk):
        """ Usage:
        Hz=bark2hs(Brk)
        Args     :
            Brk  : (ndarray)    Array containing Bark scaled values.
        Returns  :
            Fhz  : (ndarray)    Array containing frequencies in Hz.
        """
        if not torch.is_tensor(Brk):
            Brk = torch.tensor(Brk)
        Fhz = 600. * torch.sinh(Brk/6.)
        return Fhz

    def mapping2barkmat_torch(self, fs, nfilts, nfft):
        # Constructing mapping matrix W which has 1's for each Bark subband, and 0's else
        #usage: W=mapping2barkmat(fs, nfilts,nfft)
        # arguments: fs: sampling frequency
        # nfilts: number of subbands in Bark domain
        # nfft: number of subbands in fft
        # upper end of our Bark scale:22 Bark at 16 kHz
        maxbark = self.hz2bark_torch(fs/2)
        nfreqs = nfft/2
        step_bark = maxbark/(nfilts-1)
        binbark = self.hz2bark_torch(
            torch.linspace(0, (nfft/2), (nfft//2)+1)*fs/nfft)
        W = torch.zeros((nfilts, nfft))
        for i in range(nfilts):
            W[i, 0:int(nfft/2)+1] = (torch.round(binbark/step_bark) == i)
        return W

    def mapping2bark_torch(self, mX, W, nfft):
        # Maps (warps) magnitude spectrum vector mX from DFT to the Bark scale
        # arguments: mX: magnitude spectrum from fft
        # W: mapping matrix from function mapping2barkmat
        # nfft: : number of subbands in fft
        # returns: mXbark, magnitude mapped to the Bark scale
        nfreqs = int(nfft/2)
        # Here is the actual mapping, suming up powers and conv. back to Voltages:
        mXbark = (torch.matmul(
            torch.abs(mX[:nfreqs])**2.0, W[:, :nfreqs].T))**(0.5)
        return mXbark

    def mappingfrombarkmat_torch(serlf, W, nfft):
        # Constructing inverse mapping matrix W_inv from matrix W for mapping back from bark scale
        #usuage: W_inv=mappingfrombarkmat(Wnfft)
        # argument: W: mapping matrix from function mapping2barkmat
        # nfft: : number of subbands in fft
        nfreqs = int(nfft/2)
        W_inv = torch.matmul(torch.diag(
            (1.0/(torch.sum(W, 1)+1e-6))**0.5), W[:, 0:nfreqs + 1]).T
        return W_inv

    def mappingfrombark_torch(self, mTbark, W_inv, nfft):
        #usage: mT=mappingfrombark(mTbark,W_inv,nfft)
        # Maps (warps) magnitude spectrum vector mTbark in the Bark scale
        # back to the linear scale
        # arguments:
        # mTbark: masking threshold in the Bark domain
        # W_inv : inverse mapping matrix W_inv from matrix W for mapping back from bark scale
        # nfft: : number of subbands in fft
        # returns: mT, masking threshold in the linear scale
        nfreqs = int(nfft/2)
        mT = torch.matmul(mTbark, W_inv[:, :nfreqs].T.float())
        return mT

    def f_SP_dB_torch(self, maxfreq, nfilts):
        #usage: spreadingfunctionmatdB=f_SP_dB(maxfreq,nfilts)
        # computes the spreading function protoype, in the Bark scale.
        # Arguments: maxfreq: half the sampling freqency
        # nfilts: Number of subbands in the Bark domain, for instance 64
        # upper end of our Bark scale:22 Bark at 16 kHz
        maxbark = self.hz2bark_torch(maxfreq)
        # Number of our Bark scale bands over this range: nfilts=64
        spreadingfunctionBarkdB = torch.zeros(2*nfilts)
        # Spreading function prototype, "nfilts" bands for lower slope
        spreadingfunctionBarkdB[0:nfilts] = torch.linspace(
            -maxbark*27, -8, nfilts)-23.5
        # "nfilts" bands for upper slope:
        spreadingfunctionBarkdB[nfilts:2 *
                                nfilts] = torch.linspace(0, -maxbark*12.0, nfilts)-23.5
        return spreadingfunctionBarkdB

    def spreadingfunctionmat_torch(self, spreadingfunctionBarkdB, alpha, nfilts):
        # Turns the spreading prototype function into a matrix of shifted versions.
        # Convert from dB to "voltage" and include alpha exponent
        # nfilts: Number of subbands in the Bark domain, for instance 64
        spreadingfunctionBarkVoltage = 10.0**(
            spreadingfunctionBarkdB/20.0*alpha)
        # Spreading functions for all bark scale bands in a matrix:
        spreadingfuncmatrix = torch.zeros((nfilts, nfilts))
        for k in range(nfilts):
            spreadingfuncmatrix[k, :] = spreadingfunctionBarkVoltage[(
                nfilts-k):(2*nfilts-k)]
        return spreadingfuncmatrix

    def maskingThresholdBark_torch(self, mXbark, spreadingfuncmatrix, alpha, fs, nfilts):
        # Computes the masking threshold on the Bark scale with non-linear superposition
        #usage: mTbark=maskingThresholdBark(mXbark,spreadingfuncmatrix,alpha)
        # Arg: mXbark: magnitude of FFT spectrum, on the Bark scale
        # spreadingfuncmatrix: spreading function matrix from function spreadingfunctionmat
        # alpha: exponent for non-linear superposition (eg. 0.6),
        # fs: sampling freq., nfilts: number of Bark subbands
        # nfilts: Number of subbands in the Bark domain, for instance 64
        # Returns: mTbark: the resulting Masking Threshold on the Bark scale

        # Compute the non-linear superposition:
        mTbark = torch.matmul(mXbark**alpha, spreadingfuncmatrix**alpha)
        # apply the inverse exponent to the result:
        mTbark = mTbark**(1.0/alpha)
        # Threshold in quiet:
        maxfreq = fs/2.0
        maxbark = self.hz2bark_torch(maxfreq)
        step_bark = maxbark/(nfilts-1)
        barks = torch.arange(0, nfilts)*step_bark
        # convert the bark subband frequencies to Hz:
        f = self.bark2hz_torch(barks)+1e-6
        # Threshold of quiet in the Bark subbands in dB:
        LTQ = torch.clip((3.64*(f/1000.)**-0.8 - 6.5*torch.exp(-0.6*(f/1000.-3.3)**2.)
                          + 1e-3*((f/1000.)**4.)), -20, 120)
        # Maximum of spreading functions and hearing threshold in quiet:
        a = mTbark
        b = 10.0**((LTQ-60)/20)
        mTbark = torch.max(a, b)
        return mTbark

    def forward(self, x, y):
        # Compute STFT of x
        x_stft = torch.stft(x, self.n_fft, return_complex=self.return_complex)
        x_stft_mt = torch.zeros_like(x_stft)
        # Compute mT for every Block
        n_frames = x_stft.shape[1]
        W = self.mapping2barkmat_torch(self.fs, self.n_filts, self.n_fft)
        spreadingfunctionBarkdB = self.f_SP_dB_torch(
            self.maxfreq, self.n_filts)
        spreadingfuncmatrix = self.spreadingfunctionmat_torch(
            spreadingfunctionBarkdB, self.alpha, self.n_filts)
        for frame in range(n_frames):
            stft_frame = x_stft[:, frame, 0]
            mXbark = self.mapping2bark_torch(stft_frame, W, self.n_fft)
            mTbark = self.maskingThresholdBark_torch(
                mXbark, spreadingfuncmatrix, self.alpha, self.fs, self.n_filts)
            W_inv = self.mappingfrombarkmat_torch(W, self.n_fft)
            mT = self.mappingfrombark_torch(mTbark, W_inv, self.n_fft)
            x_stft_mt[:, frame, 0] = stft_frame * mT**-1

        # Compute STFT of Y
        y_stft = torch.stft(y, self.n_fft, return_complex=self.return_complex)
        y_stft_mt = torch.zeros_like(y_stft)
        # Compute mT for every Block
        n_frames = y_stft.shape[1]
        W = self.mapping2barkmat_torch(self.fs, self.n_filts, self.n_fft)
        spreadingfunctionBarkdB = self.f_SP_dB_torch(
            self.maxfreq, self.n_filts)
        spreadingfuncmatrix = self.spreadingfunctionmat_torch(
            spreadingfunctionBarkdB, self.alpha, self.n_filts)
        for frame in range(n_frames):
            stft_frame = y_stft[:, frame, 0]
            mXbark = self.mapping2bark_torch(stft_frame, W, self.n_fft)
            mTbark = self.maskingThresholdBark_torch(
                mXbark, spreadingfuncmatrix, self.alpha, self.fs, self.n_filts)
            W_inv = self.mappingfrombarkmat_torch(W, self.n_fft)
            mT = self.mappingfrombark_torch(mTbark, W_inv, self.n_fft)
            y_stft_mt[:, frame, 0] = stft_frame * mT**-1

        if self.mode == "mse_frequency":
            x = x_stft_mt.flatten()
            y = y_stft_mt.flatten()
            mse = torch.nn.MSELoss()
            return mse(x_stft_mt, y_stft_mt)

        # Compute ISTFT of x
        x_istft = torch.istft(x_stft_mt, self.n_fft,
                              return_complex=self.return_complex)
        # Compute ISTFT of y
        y_istft = torch.istft(y_stft_mt, self.n_fft,
                              return_complex=self.return_complex)

        # Compute Difference
        if self.mode == "mse_time":
            mse = torch.nn.MSELoss()
            return mse(x_istft, y_istft)
        if self.mode == "multiscale":
            loss_multiScaleSpectral = SingleSrcMultiScaleSpectral()
            return loss_multiScaleSpectral(x_istft.unsqueeze(dim=0), y_istft.unsqueeze(dim=0))
        if self.mode == "l1":
            l1_loss = nn.L1Loss()
            return l1_loss(x_istft, y_istft)


if __name__ == "__main__":
    # Signal Processing Parameters

    fs = 48000  # [Hz]
    maxfreq = fs/2
    duration = 2  # [s]
    n_fft = 2**17
    alpha = 0.8  # Exponent for non-linear superposition of spreading functions
    nfilts = 64  # number of subbands in the bark domain

    x1 = torch.sin(2*np.pi/n_fft*2404*torch.arange(n_fft))
    x2 = torch.sin(2*np.pi/n_fft*(2404+1000)*torch.arange(n_fft))*0.005
    y = x1 + x2
    psychoLoss_mse = PsychoAcousticLoss()
    print('psychoLoss mse:', psychoLoss_mse(x1, y))
    loss_mse = torch.nn.MSELoss()
    print('mse:', loss_mse(x1, y))
    loss_multiScaleSpectral = SingleSrcMultiScaleSpectral()
    print('Multi Scale Spectral:', loss_multiScaleSpectral(
        x1.unsqueeze(dim=0), y.unsqueeze(dim=0)))
    psychoLoss_multiscale = PsychoAcousticLoss(mode='multiscale')
    print('psychoLoss multiscale:', psychoLoss_multiscale(x1, y))
    print('ratio multiscale/psycho_multiscale:', loss_multiScaleSpectral(
        x1.unsqueeze(dim=0), y.unsqueeze(dim=0))/psychoLoss_multiscale(x1, y))
