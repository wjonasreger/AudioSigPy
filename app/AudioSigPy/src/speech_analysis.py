from typing import Optional, Tuple
import numpy as np
from . import array as Array
from . import transforms as Transforms
from . import frequency as Frequency
from . import peak_detection as PeakDetection
from . import statistics as Statistics
from . import windows as Windows
from . import sound as Sound


def calc_fundamentals(
    signal: np.ndarray, 
    sample_rate: Optional[float] = None, 
    step: int = 5, 
    method: str = "mse", 
    window: str = "rectangular",
    clip_thres: str = 0.3,
    clip_mode: str = "normal",
    pool: bool = True
) -> np.ndarray:
    """
    Compute the fundamental period and frequency of a signal.
    
    Math:
        F0 = 1 / T0
        T0 = (L * d + 1) / fs
        where:
            F0 is the fundamental frequency in Hz
            T0 is the fundamental period in seconds
            L is the index of the first peak detected in a statistic measure signal
            d is the delay between the base signal and the lagged signal that yielded the first peak
            fs is the sampling rate in Hz
            L * d + 1 is the number of samples to the first peak

    Args:
        signal (np.ndarray): The input signal
        sample_rate (float, optional): The sampling rate of the signal. Defaults to None.
        step (int, optional): The lag step for instantaneous signal statistics calculations. Defaults to 2.
        method (str, optional): The computation method. Defaults to "mse".
        window (str, optional): The window for instantaneous computations. Defaults to "rectangular".
        clip_thres (str, optional): The clipping threshold used for the autocorrelation method. Defaults to 0.3.
        clip_mode (str, optional): The clipping mode used for the autocorrelation method. Defaults to "normal".
        pool (bool, optional): Pool the channel data. Defaults to True.

    Returns:
        np.ndarray: The fundamental periods and frequencies of the signal (T0, F0).
    """
    
    if not isinstance(method, str):
        raise TypeError(f"[ERROR] Calculate Funadmentals: Method must be a string. Got {type(method).__name__}.")
    method = method.lower()
    if method not in ["mse", "ac"]:
        raise ValueError(f"[ERROR] Calculate Fundamentals: Method must be one of [mse, ac]. Got {method}.")
    
    signal = Array.validate(array=signal)
    sample_rate = Frequency.val_sample_rate(sample_rate=sample_rate)
    
    # Different methods to compute the fundamentals
    if method == "mse":
        # Get first peak of -MSE signal
        instants = -1 * Statistics.instant_stats(signal=signal, step=step, window=window, method="mse", pool=pool)
        peak_index = PeakDetection.find_first_peak(signal=instants, normalize=True, threshold=0.7, pool=pool)
    elif method == "ac":
        # Get first peak of centre clipped autocorrelation signal
        instants = Statistics.instant_stats(signal=signal, step=step, window=window, method="ac", pool=pool)
        instants = Transforms.centre_clip(signal=instants, threshold=clip_thres, mode=clip_mode, pool=pool)
        peak_index = PeakDetection.find_first_peak(signal=instants, normalize=True, threshold=0.7, pool=pool)
    
    # Compute the fundamental periods and frequencies (T0, F0)
    T0 = (peak_index * step + 1) / sample_rate
    F0 = np.round(1 / T0, 1)
    
    return T0, F0


def pitch_contour(
    signal: np.ndarray, 
    sample_rate: Optional[float] = None, 
    size: float = 0.02, 
    overlap: float = 0.5, 
    step: int = 2, 
    window: str = "rectangular", 
    instant_window: str = "rectangular", 
    clip_thres: float = 0.3, 
    clip_mode: str = "normal",
    detection_mode: str = "stable"
) -> np.ndarray:
    """
    Map out the pitch contour of a signal by computing fundamental frequencies over time.

    Args:
        signal (np.ndarray): The input signal.
        sample_rate (float, optional): The sample rate. Defaults to None.
        size (float, optional): The proportional size of the window. Defaults to 0.02.
        overlap (float, optional): The proportional window overlap. Defaults to 0.5.
        step (int, optional): The step size for instantaneous signal measures. Defaults to 2.
        window (str, optional): The window function for the window. Defaults to "rectangular".
        instant_window (str, optional): The window function for instant computations. Defaults to "rectangular".
        clip_thres (float, optional): The clipping threshold for centre clipping. Defaults to 0.3.
        clip_mode (str, optional): The clipping mode for centre clipping. Defaults to "normal".
        detection_mode (str, optional): The peak detection mode. Default to stable.

    Returns:
        np.ndarray: The pitch contour of the signal.
        
    Raises:
        TypeError: If size or overlap is not float.
        ValueError: If size or overlap is out of bounds for (0, 1)
    """
    
    if not isinstance(size, float):
        raise TypeError(f"[ERROR] Pitch Contour: Size must be a float. Got {type(size).__name__}.")
    if not isinstance(overlap, float):
        raise TypeError(f"[ERROR] Pitch Contour: Overlap must be a float. Got {type(overlap).__name__}.")
    
    if size <= 0 or size > 1:
        raise ValueError(f"[ERROR] Pitch Contour: Size must be in (0, 1]. Got {size}.")
    
    if overlap > 0.95:
        overlap = 0.95
        print(f"[WARNING] Pitch Contour: Overlap should be no greater than 0.95 to ensure stability. Got {overlap}. Resetting overlap to 0.95.")
        
    if overlap < 0 or overlap >= 1:
        raise ValueError(f"[ERROR] Pitch Contour: Overlap must be in [0, 1). Got {overlap}.")
    
    # Validate the signal and sample rate
    signal = Array.validate(array=signal)
    sample_rate = Frequency.val_sample_rate(sample_rate)

    # Compute the sample size of windows and shifts
    sample_count = int(size * sample_rate)
    shift = int(sample_count * (1 - overlap))

    # Generate the window function
    w = Windows.choose_window(length=sample_count, name=window, mode="run")

    # Allocated space for pitch contour
    pitch_contour = np.zeros(signal.shape)
    
    # Iterate over channels
    for c in range(signal.shape[1]):
        # Shift window over the signal
        for i in range(0, signal.shape[0] - sample_count, shift):
            # Apply window function on the window
            window = signal[i:i + sample_count, c] * w.reshape(-1)

            # Pad zeros to prior to applying instant autocorrelation
            shape = list(window.shape)
            shape[0] = sample_count
            window = np.concatenate([window, np.zeros(shape)], axis=0)
            
            # Compute instant autocorrelation of the window
            instant_ac = Statistics.instant_stats(window, step, instant_window, "ac")
            
            # Apply centre clipping to preprocess autocorrelated signal for peak detection
            instant_ac = Transforms.centre_clip(instant_ac, clip_thres, clip_mode)
            
            # Search for the first peak to compute the fundamental frequency of the signal
            peak_idx = PeakDetection.find_first_peak(instant_ac, normalize=False, threshold=0, mode=detection_mode)
            
            # Compute the fundamental frequency for pitch contour
            pitch_contour[i:i + sample_count, c] = sample_rate / (peak_idx * step + 1)

    return pitch_contour


def val_ffrequency(signal: np.ndarray, width: int = 2) -> np.ndarray:
    """
    Validate fundamental frequencies (F0) from pitch contour.

    Args:
        signal (np.ndarray): The input fundamental frequencies to be cleaned.
        width (int, optional): The width of a series of valid F0 values. Defaults to 2.

    Returns:
        np.ndarray: The cleaned pitch contour signal.
        
    Raises:
        TypeError: If width is not an integer.
        ValueError: If width is not positive.
    """
    
    if not isinstance(width, int):
        raise TypeError(f"[ERROR] Validate Fundamental Frequencies: Width must be an integer. Got {type(width).__name__}.")
    
    if width <= 0:
        raise ValueError(f"[ERROR] Validate Fundamental Frequencies: Width must be a positive number. Got {width}.")
    
    signal = Array.validate(array=signal)
    
    # Set F0 values outside of human range to 0
    signal[signal < 50] = 0
    signal[signal > 600] = 0
    
    # Iterate over channels
    for c in range(signal.shape[1]):
        count = 0
        for i in range(signal.shape[0]):
            # non-zero and non-end signal increases count
            if (signal[i, c] > 0) and (i < signal.shape[0] - 1):
                count += 1
            # non-zero and end signal increases count and checks length
            elif (signal[i, c] > 0) and (i == signal.shape[0] - 1):
                count += 1
                if count < width:
                    signal[i - count:i, c] = 0
                count = 0
            # check length of last non-zero signal when 0 is found
            else:
                if count < width:
                    signal[i - count:i, c] = 0
                count = 0
            
    return signal


def find_endpoints(
    signal: np.ndarray, 
    sample_rate: Optional[float] = None, 
    size: float = 0.02, 
    int_threshold: float = 60.0, 
    zcr_threshold: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect the endpoints of silence in a signal.

    Args:
        signal (np.ndarray): The input signal.
        sample_rate (float, optional): The sample rate of the signal. Defaults to None.
        size (float, optional): The window size. Defaults to 0.02.
        int_threshold (float, optional): The intensity threshold. Defaults to 60.0.
        zcr_threshold (float, optional): The ZCR threshold. Defaults to 0.05.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The indicator signal for silence.
        
    Raises:
        TypeError: If size, int_threshold, zcr_threshold is not float
        ValueError: If size or zcr_threshold is not in (0, 1)
    """
    
    if not isinstance(size, float):
        raise TypeError(f"[ERROR] Find Endpoints: Size must be a float. Got {type(size).__name__}.")
    if not isinstance(int_threshold, (int, float)):
        raise TypeError(f"[ERROR] Find Endpoints: Intensity threshold must be an integer or float. Got {type(int_threshold).__name__}.")
    if not isinstance(zcr_threshold, float):
        raise TypeError(f"[ERROR] Find Endpoints: ZCR threshold must be a float. Got {type(zcr_threshold).__name__}.")
    
    if size <= 0 or size >= 1:
        raise ValueError(f"[ERROR] Find Endpoints: Size must be in (0, 1). Got {size}.")
    if zcr_threshold <= 0 or zcr_threshold >= 1:
        raise ValueError(f"[ERROR] Find Endpoints: ZCR threshold must be in (0, 1). Got {zcr_threshold}.")
    
    # Validate signal and sample rate
    signal = Array.validate(array=signal)
    sample_rate = Frequency.val_sample_rate(sample_rate)
    
    # Compute windowing parameters
    sample_count = int(sample_rate * size)
    window_count = int(np.ceil(signal.shape[0] / sample_count))
    
    # Allocate lists for statistics
    intensities = np.zeros((window_count, signal.shape[1]))
    zcrs = np.zeros((window_count, signal.shape[1]))
    silences = np.zeros((window_count, signal.shape[1]))
    
    # Iterate over channels
    for c in range(signal.shape[1]):
        # Iterate over each window
        for i in range(window_count):
            # Subset window from signal
            start = int(i * sample_count)
            end = int((i + 1) * sample_count)
            window = signal[start:end, c]
            
            # Compute intensities and ZCRs to get silences
            intensities[i, c] = Sound.calc_sil(window, raw=True)
            zcrs[i, c] = Sound.calc_zcr(window)
            silences[i, c] = 1 if intensities[i, c] < -int_threshold and \
                (zcrs[i, c] < zcr_threshold or np.isnan(zcrs[i, c])) else 0
            
    return (intensities, zcrs, silences)