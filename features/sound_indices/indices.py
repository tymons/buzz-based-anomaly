#!/usr/bin/env python
__author__ = "Patrice Guyot"
__version__ = "0.4"
__credits__ = ["Patrice Guyot", "Alice Eldridge", "Mika Peck"]
__email__ = ["guyot.patrice@gmail.com", "alicee@sussex.ac.uk", "m.r.peck@sussex.ac.uk"]
__status__ = "Development"

from scipy import signal, fftpack
import numpy as np


def compute_aci(spectro, j_bin):
    """
    Compute the Acoustic Complexity Index from the spectrogram of an audio signal.

    Reference: Pieretti N, Farina A, Morri FD (2011) A new methodology to infer the singing activity of an
    avian community: the Acoustic Complexity Index (ACI). Ecological Indicators, 11, 868-873.

    Ported from the soundecology R package.

    spectro: the spectrogram of the audio signal
    j_bin: temporal size of the frame (in samples)


    """

    times = range(0, spectro.shape[1], j_bin)  # relevant time indices
    j_specs = [np.array(spectro[:, i:i + j_bin]) for i in times]  # sub-spectros of temporal size j
    aci = [sum((np.sum(abs(np.diff(jspec)), axis=1) / np.sum(jspec, axis=1))) for jspec in j_specs]

    main_value = sum(aci)
    temporal_values = aci

    return main_value, temporal_values


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def compute_bi(spectro, frequencies, dbfs_max, min_freq=2000, max_freq=8000):
    """
    Compute the Bioacoustic Index from the spectrogram of an audio signal.
    In this code, the Bioacoustic Index correspond to the area under the mean spectre (in dB) minus
    the minimum frequency value of this mean spectre.

    Reference: Boelman NT, Asner GP, Hart PJ, Martin RE. 2007. Multi-trophic invasion resistance in Hawaii:
    bioacoustics, field surveys, and airborne remote sensing. Ecological Applications 17: 2137-2144.

    spectro: the spectrogram of the audio signal
    frequencies: list of the frequencies of the spectrogram
    dbfs_max: max value for dbfs
    min_freq: minimum frequency (in Hertz)
    max_freq: maximum frequency (in Hertz)

    Ported from the soundecology R package.
    """

    min_freq_bin = int(np.argmin([abs(e - min_freq) for e in frequencies]))
    max_freq_bin = int(np.ceil(np.argmin([abs(e - max_freq) for e in frequencies])))

    spectro_bi = 20 * np.log10(spectro / dbfs_max)
    spectre_bi_mean = 10 * np.log10(np.mean(10 ** (spectro_bi / 10), axis=1))
    spectre_bi_mean_segment = spectre_bi_mean[min_freq_bin:max_freq_bin]
    spectre_bi_mean_segment_normalized = spectre_bi_mean_segment - min(spectre_bi_mean_segment)
    area = np.sum(spectre_bi_mean_segment_normalized / (frequencies[1] - frequencies[0]))

    return area, spectre_bi_mean_segment_normalized


def compute_sh(spectro):
    """
    Compute Spectral Entropy of Shannon from the spectrogram of an audio signal.

    spectro: the spectrogram of the audio signal

    Ported from the seewave R package.
    """
    n_len = spectro.shape[0]
    spec = np.sum(spectro, axis=1)
    spec = spec / np.sum(spec)
    main_value = - sum([y * np.log2(y) for y in spec]) / np.log2(n_len)
    return main_value


def compute_th(sig):
    """
    Compute Temporal Entropy of Shannon from an audio signal.

    file: an instance of the AudioFile class.
    integer: if set as True, the Temporal Entropy will be compute on the Integer values of the signal. If not,
    the signal will be set between -1 and 1.

    Ported from the seewave R package.
    """
    env = abs(signal.hilbert(sig, fftpack.helper.next_fast_len(len(sig))))
    env = env / np.sum(env)

    n_len = len(env)
    return - sum([y * np.log2(y) for y in env]) / np.log2(n_len)


def gini(values):
    """
    Compute the Gini index of values.

    values: a list of values

    Inspired by http://mathworld.wolfram.com/GiniCoefficient.html and http://en.wikipedia.org/wiki/Gini_coefficient
    """
    y = sorted(values)
    n = len(y)
    gini_val = np.sum([i * j for i, j in zip(y, range(1, n + 1))])
    gini_val = 2 * gini_val / np.sum(y) - (n + 1)
    return gini_val / n


def compute_aei(pcm_spectro, dbfs_max, freq_band_hz, max_freq=10000, dbfs_threshold=-50, freq_step=1000):
    """
    Compute Acoustic Evenness Index of an audio signal.

    Reference: Villanueva-Rivera, L. J., B. C. Pijanowski, J. Doucette, and B. Pekin. 2011.
    A primer of acoustic analysis for landscape ecologists. Landscape Ecology 26: 1233-1246.

    spectro: spectrogram of the audio signal
    freq_band_Hz: frequency band size of one bin of the spectrogram (in Hertz)
    max_freq: the maximum frequency to consider to compute AEI (in Hertz)
    db_threshold: the minimum dB value to consider for the bins of the spectrogram
    freq_step: size of frequency bands to compute AEI (in Hertz)

    Ported from the soundecology R package.
    """

    bands_hz = range(0, max_freq, freq_step)
    bands_bin = [f / freq_band_hz for f in bands_hz]

    spec_aei = 20 * np.log10(pcm_spectro / dbfs_max)
    spec_aei_bands = [spec_aei[int(bands_bin[k]):int(bands_bin[k] + bands_bin[1]), ] for k in range(len(bands_bin))]

    values = [np.sum(spec_aei_bands[k] > dbfs_threshold) / float(spec_aei_bands[k].size) for k in range(len(bands_bin))]

    return gini(values)


def compute_adi(pcm_spectro, dbfs_max, freq_band_hz, max_freq=10000, dbfs_threshold=-50, freq_step=1000):
    """
    Compute Acoustic Diversity Index.

    Reference: Villanueva-Rivera, L. J., B. C. Pijanowski, J. Doucette, and B. Pekin. 2011.
    A primer of acoustic analysis for landscape ecologists. Landscape Ecology 26: 1233-1246.

    spectro: spectrogram of the audio signal
    dbfs_max: max value for decibel full scale
    freq_band_Hz: frequency band size of one bin of the spectrogram (in Hertz)
    max_freq: the maximum frequency to consider to compute ADI (in Hertz)
    db_threshold: the minimum dB value to consider for the bins of the spectrogram
    freq_step: size of frequency bands to compute ADI (in Hertz)


    Ported from the soundecology R package.
    """

    bands_hz = range(0, max_freq, freq_step)
    bands_bin = [f / freq_band_hz for f in bands_hz]
    spec_adi = 20 * np.log10(pcm_spectro / dbfs_max)
    spec_adi_bands = [spec_adi[int(bands_bin[k]):int(bands_bin[k] + bands_bin[1]), ] for k in range(len(bands_bin))]

    values = [np.sum(spec_adi_bands[k] > dbfs_threshold) / float(spec_adi_bands[k].size) for k in range(len(bands_bin))]
    values = [value for value in values if value != 0]

    return np.sum([-i / np.sum(values) * np.log(i / np.sum(values)) for i in values])


def compute_zcr(sig, win_len=512, hop_len=256):
    """
    Compute the Zero Crossing Rate of an audio signal.

    file: an instance of the AudioFile class.
    windowLength: size of the sliding window (samples)
    windowHop: size of the lag window (samples)

    return: a list of values (number of zero crossing for each window)
    """
    sig = sig - int(np.mean(sig))
    times = range(0, len(sig) - win_len + 1, hop_len)
    frames = [sig[i:i + win_len] for i in times]
    return [len(np.where(np.diff(np.signbit(x)))[0]) / float(win_len) for x in frames]


def compute_spectral_centroid(spectro, frequencies):
    """
    Compute the spectral centroid of an audio signal from its spectrogram.

    spectro: spectrogram of the audio signal
    frequencies: list of the frequencies of the spectrogram
    """

    centroid = [np.sum(magnitudes * frequencies) / np.sum(magnitudes) for magnitudes in spectro.T]
    return centroid
