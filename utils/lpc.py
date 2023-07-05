import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt

from torch.nn import functional as F
from scipy import signal
from rich.progress import track
from config import settings

fs = settings.AUDIO.SAMPLE_RATE


def lpc_filter_poles(ak, bw_threshold=200, fs=16000, min_poles=4):
    """
    Calcular polos y eliminar algunos según el criterio de la letra
    """

    def pole_bw(p):
        # Estima ancho de banda de un polo dada su magnitud (|p|<1)
        bw = -fs * np.log10(np.abs(p)) / np.pi
        # print(f"p={p} | bw={bw}")
        return bw

    poly_coeffs = np.concatenate([[1], -ak])
    poles = np.roots(poly_coeffs)
    filtered_poles = []
    for p in poles:
        # Sólo agrego polos imaginarios, en cuadrante superior
        if p.imag > 1e-3:
            # Sólo filtro por ancho de banda si hay más de min_poles
            if pole_bw(p) <= bw_threshold or len(poles) <= min_poles:
                filtered_poles.append(p)
    return filtered_poles, poles


def lpc_poles_frequencies(poles, fs, k=2):
    # Obtiene las frecuencias (en Hz) de los k polos de menor frecuencia

    # TODO: (improvement) discard poles too near in freq
    freqs = []
    for p in poles:
        omega = np.angle(p)
        freqs.append(fs * omega / (2 * np.pi))
    freqs = np.array(freqs)
    return freqs[np.argsort(freqs)[:k]]


def plot_spectrum_and_lpc(
    wave,
    n_win=1024,
    n_hop=512,
    extra_axes=1,
    fs=16000,
):
    # total length
    L = wave.shape[0]

    window = signal.windows.get_window("hann", n_win)
    num_frames = int(np.floor((L - n_win) / n_hop))

    # Inicializo STFT
    X_stft = np.zeros((n_win, num_frames), dtype=complex)

    # LPC lowest freq poles
    n_poles = 3
    X_lpc = np.zeros((n_poles, num_frames), dtype=np.float64)

    if type(wave) is torch.Tensor:
        wave = wave.numpy()

    for ind in track(range(num_frames)):
        n_ini = int(ind * n_hop)
        n_end = n_ini + n_win

        # Fragmento enventanado
        xr = wave[n_ini:n_end] * window

        # DFT del fragmento
        X_stft[:, ind] = np.fft.fft(xr, n_win)

        # Own implementation
        # ak, r = lpc_autocorr(xr, p=20)
        # Librosa implementation
        ak = -librosa.lpc(xr, order=20)[1:]
        filtered_poles, _ = lpc_filter_poles(ak, min_poles=n_poles)
        f_formants = lpc_poles_frequencies(filtered_poles, fs, k=n_poles)
        if len(f_formants):
            X_lpc[: len(f_formants), ind] = f_formants

    # Frecuencia de 0 a fs (intervalo [fs//2 a fs] son las negativas)
    f_s = fs * np.arange(n_win) / n_win
    k_s = np.int64(n_win / 2 + np.arange(num_frames) * n_hop)  # índices de cada ventana
    t_s = k_s / fs  # t de cada ventana

    total_axes = 2 + extra_axes
    fig, axes = plt.subplots(nrows=total_axes, ncols=1, figsize=(10, 3 * total_axes))
    fig.tight_layout(pad=3.0)
    if type(axes) is not np.ndarray:
        axes = [axes]  # single axis case

    axes[0].pcolormesh(t_s, f_s, 2 * np.log(np.abs(X_stft)))
    axes[0].set_xlabel("t [s]")
    axes[0].set_ylabel("Freq [Hz]")
    axes[0].set_ylim([0, fs // 4])
    axes[0].title.set_text("Espectrograma")

    axes[1].title.set_text("LPC polos de menos frecuencia")
    axes[1].plot(t_s, X_lpc.T)
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel("Pole freq [Hz]")

    return axes, X_lpc


def get_formants_lpc(wave, window, n_win=1024, n_hop=512, n_poly=20, n_formants=2):
    L = wave.shape[0]
    # num_frames = int(np.floor((L - n_win) / n_hop))
    num_frames = int(np.floor(L / n_hop)) + 1
    l_formants = []
    for ind in range(num_frames):
        n_ini = int(ind * n_hop)
        n_end = n_ini + n_win
        if n_end > len(wave):
            wave = F.pad(wave, (0, n_end - len(wave)))  # zero pad last window
        xr = wave[n_ini:n_end] * window

        # poly coefficients
        ak = -librosa.lpc(xr.numpy(), order=n_poly)[1:]
        filtered_poles, _ = lpc_filter_poles(ak, min_poles=n_formants)
        f_formants = lpc_poles_frequencies(filtered_poles, fs, k=n_formants)
        l_formants.append(f_formants)
    return l_formants
