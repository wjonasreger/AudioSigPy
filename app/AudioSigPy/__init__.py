from .src.array import (
    unsqueeze,
    align,
    validate,
    downsample,
    linseries,
    subset
)

from .src.plot import (
    plot_series,
    plot_image
)

from .src.spectral import (
    dft,
    fft,
    stft,
    istft,
    calc_spectra,
    calc_spectrogram
)

from .src.windows import (
    choose_window,
    linear,
    generalized_cosine,
    generalized_hann
)

from .src.transforms import (
    sinc,
    amplify,
    integrate,
    smooth,
    logn_transform,
    scalar_transform,
    index_transform,
    centre_clip,
    Quantiser
)

from .src.peak_detection import (
    find_local_maxima,
    find_peaks,
    find_first_peak
)

from .src.frequency import (
    val_sample_rate,
    calc_lowcut,
    calc_highcut,
    calc_step_bands,
    calc_erb,
    hz2erb_rate,
    erb_rate2hz,
    erb_centre_freqs,
    note2frequency,
    frequency2note,
    fft_frequencies
)

from .src.generators import (
    choose_generator,
    sine,
    square,
    triangle,
    sawtooth,
    pulse,
    white_noise,
    chirp,
    glottal
)

from .src.sound import (
    calc_rms,
    calc_spl,
    calc_sil,
    calc_loudness,
    calc_zcr,
    rms_transform,
    spl_transform,
    sil_transform
)

from .src.statistics import (
    instant_stats,
    calc_centre_gravity,
    calc_central_moment,
    calc_std,
    calc_skewness,
    calc_kurtosis
)

from .src.filter import (
    pole_mapping,
    gamma_tone_filter,
    convolve,
    linear_filter,
    fir_coefs,
    fir_filter,
    butter_filter
)

from .src.speech_analysis import (
    calc_fundamentals,
    pitch_contour,
    val_ffrequency,
    find_endpoints
)

# from .src.processors import (...)

from .src.waveform import (
    Waveform
)


# Package-wide variables
__version__ = '1.0.0'
__author__ = 'W. Jonas Reger'

# Initialization code
print("AudioSigPy package initialized")