from typing import Tuple, Optional
import numpy as np
from . import array as Array
from . import windows as Windows
from . import frequency as Frequency
from . import spectral as Spectral


def instant_stats(
    signal: np.ndarray, 
    step: int = 2, 
    window: str = "hamming", 
    method: str = "ac", 
    pool: bool = True
) -> np.ndarray:
    """
    Compute instant statistics between base signal and each lagged signal of a given signal. 
    
    Methods:
        Instant Autocorrelation (ac)
            ac(n) = sum(x(n) * x(n + k))
            
        Mean Squared Error (mse)
            mse(n) = mean((x(n) - x(n + k))^2)
            
        Root Mean Squared Error (rmse)
            rmse(n) = sqrt(mean((x(n) - x(n + k))^2))
            
        Mean Absolute Error (mae)
            mae(n) = mean(|x(n) - x(n + k)|)
            
        Base signal and lagged signals (none)

    Args:
        signal (np.ndarray): A time series signal
        step (int, optional): The number of samples to delay each shift. Defaults to 2.
        window (str, optional): The window function. Defaults to "hamming".
        method (str, optional): The statistic to compute. Defaults to "ac".
        pool (bool, optional): To pool the channel data. Defaults to True.

    Raises:
        TypeError: If step is not an integer or method is not a string or pool is not bool.
        ValueError: If step is not positive, if window function is not available, or method is not valid.

    Returns:
        np.ndarray: instant stats for lagged signals, or the base and lagged signals if no stat measure is selected.
    """
    
    # Checking and validating inputs
    if not isinstance(pool, bool):
        raise TypeError(f"[ERROR] Instant stats: Pool must be a boolean. Got {type(pool).__name__}.")
    if not isinstance(step, int):
        raise TypeError(f"[ERROR] Instant stats: Step must be an integer. Got {type(step).__name__}.")
    elif step <= 0:
        raise ValueError(f"[ERROR] Instant stats: Step must be a positive integer. Got {step}.")
    
    if not Windows.choose_window(length=0, name=window, mode="check"):
        raise ValueError(f"[ERROR] Instant stats: Window must be a supported function. Got {window}.")
    
    if not isinstance(method, str):
        raise TypeError(f"[ERROR] Instant stats: Method must be a string. Got {type(method).__name__}.")
    
    method = method.lower()
    if method not in ["ac", "mse", "rmse", "mae", "none"]:
        raise ValueError(f"[ERROR] Instant stats: Method must be one of [ac, mse, rmse, mae, none]. Got {method}.")
    
    signal = Array.validate(signal)
    
    # Extend the signal as needed to ensure even numbered samples
    if signal.shape[0] % 2 == 1:
        shape = list(signal.shape)
        shape[0] = 1
        signal = np.concatenate([signal, np.zeros(shape)])
        
    # Base signal is the first half of the original signal
    base_signal = Array.subset(array=signal, limits=[0.5], axes=[0])
    
    # The lag index and lag signal storage
    index = Array.linseries(start=0, end=base_signal.shape[0]-1, endpoint=False, coerce="int")
    lag_index = index[index % step == 0].reshape((-1, 1))
    lag_signals = np.zeros((lag_index.shape[0], ) + base_signal.shape)
    
    # Get each lagged signal after the base signal
    for i, lag in enumerate(lag_index):
        lag_signals[i, :] = Array.subset(
                                array=signal,
                                limits=[[lag[0], lag[0] + base_signal.shape[0] - 1]],
                                axes=[0],
                                method="index",
                                how="inner"
                            )
        
    # Create the window array
    window_array = Windows.choose_window(length=base_signal.shape[0], name=window, mode="run")
    
    # Apply the windowing function to the base signal and each lagged signal
    base_signal = base_signal * window_array
    for i in range(lag_index.shape[0]):
        lag_signals[i, :] = lag_signals[i, :] * window_array
        
    if method != "none":
        # Comput the statistics on the lagged signals
        output = []
        for i in range(lag_index.shape[0]):
            axis = None if pool else 0
            if method == "ac":
                stat = np.sum(base_signal * lag_signals[i], axis=axis)
            elif method == "mse":
                stat = np.mean((base_signal - lag_signals[i, :])**2, axis=axis)
            elif method == "rmse":
                stat = np.sqrt(np.mean((base_signal - lag_signals[i, :])**2, axis=axis))
            elif method == "mae":
                stat = np.mean(np.abs(base_signal - lag_signals[i, :]), axis=axis)
                
            output.append(stat)
            
        # Validate and normalize the output array of statistics
        output = Array.validate(array=output)
        output = output / np.nanmax(output)

        return output
    
    else:
        return base_signal, lag_signals


def calc_centre_gravity(
    signal: np.ndarray, 
    sample_rate: Optional[float] = None, 
    power: float = 2, 
    pool: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the spectral centre of gravity of the signal in Hz.
    
    Formula:
        centre_gravity = sum(f(k) * X(k)^p) / sum(X(k)^p), where p is the power to the spectrum.

    Args:
        signal (np.ndarray): Time series signal.
        sample_rate (int, optional): Sampling rate. Defaults to None.
        power (int, optional): Power to the spectrum. Defaults to 2.
        pool (bool, optional): Pool channels together. Defaults to True.

    Returns:
        (np.ndarray, np.ndarray, np.ndarray): The centre of gravity for the signal.
        
    Raises:
        TypeError: If power is not float or integer, or pool is not boolean.
    """
    
    if not isinstance(power, (float, int)):
        raise TypeError(f"[ERROR] Calc Centre Gravity: Power must be an integer or float. Got {type(power).__name__}.")
    if not isinstance(pool, bool):
        raise TypeError(f"[ERROR] Calc Centre Gravity: Pool must be an boolean. Got {type(pool).__name__}.")
    
    if power <= 0:
        print(f"[WARNING] Calc Centre Gravity: Power should be a postive number. Got {power}. This may impact related operations.")
    
    signal = Array.validate(array=signal)
    sample_rate = Frequency.val_sample_rate(sample_rate=sample_rate)
    
    # Compute the spectral frequencies and magnitudes
    frequencies = Frequency.fft_frequencies(size=signal.shape[0])
    _, magnitudes, _ = Spectral.calc_spectra(signal=signal)
    
    # Subset the frequencies and magnitudes for half the spectrum
    frequencies = Array.subset(array=frequencies, limits=[0.5], axes=[0])
    magnitudes = Array.subset(array=magnitudes, limits=[0.5], axes=[0])
    
    # Compute the centre of gravity
    axis = None if pool else 0
    mag_power = magnitudes ** power
    centre_gravity = np.sum(frequencies * mag_power, axis=axis) / np.sum(mag_power, axis=axis)
    
    centre_gravity = Array.validate(array=centre_gravity, direction="up")
    
    return centre_gravity, frequencies, magnitudes


def calc_central_moment(
    signal: np.ndarray, 
    sample_rate: Optional[float] = None, 
    moment: float = 2, 
    power: float = 2, 
    pool: bool = True
) -> np.ndarray:
    """
    Calculate the m^th spectral central moment of the signal.
    
    Formula:
        central_moment = sum((f(k) - c)^m * X(k)^p) / sum(X(k)^p)
        
        where c is the centre of gravity, m is the moment, and p is the power to the spectrum.

    Args:
        signal (np.ndarray): Time series signal.
        sample_rate (float, optional): The sampling rate of the signal. Defaults to None.
        moment (int, optional): The m^th spectral moment. Defaults to 2.
        power (int, optional): The power of the spectrum. Defaults to 2.
        pool (bool, optional): To pool the channel data. Defaults to True.

    Returns:
        np.ndarray: The central moment for the signal.
        
    Raises:
        TypeError: If power or moment is not float or integer, or pool is not boolean.
    """
    
    if not isinstance(power, (float, int)):
        raise TypeError(f"[ERROR] Calc Central Moment: Power must be an integer or float. Got {type(power).__name__}.")
    if not isinstance(moment, (float, int)):
        raise TypeError(f"[ERROR] Calc Central Moment: Moment must be an integer or float. Got {type(moment).__name__}.")
    if not isinstance(pool, bool):
        raise TypeError(f"[ERROR] Calc Central Moment: Pool must be an boolean. Got {type(pool).__name__}.")
    
    if power <= 0:
        print(f"[WARNING] Calc Central Moment: Power should be a postive number. Got {power}. This may impact related operations.")
    if moment <= 0:
        print(f"[WARNING] Calc Central Moment: Moment should be a postive number. Got {moment}. This may impact related operations.")
        
    # Compute the central moment of the spectrum
    axis = None if pool else 0
    centre_gravity, frequencies, magnitudes = calc_centre_gravity(signal, sample_rate, pool=pool)
    mag_power = magnitudes ** power
    central_moment = np.sum((frequencies - centre_gravity) ** moment * mag_power, axis=axis) / np.sum(mag_power, axis=axis)
    
    central_moment = Array.validate(array=central_moment, direction="up")
    
    return central_moment


def calc_std(
    signal: np.ndarray, 
    sample_rate: Optional[float] = None, 
    power: float = 2, 
    pool: bool = True
) -> np.ndarray:
    """
    Calculate the spectral standard deviation of the signal.
    
    Formula:
        STD(X(k)) = sqrt(moment(X(k), 2))

    Args:
        signal (np.ndarray): Time series signal.
        sample_rate (float, optional): The smapling rate of the signal. Defaults to None.
        power (int, optional): The power of the spectrum. Defaults to 2.
        pool (bool, optional): To pool the channel data. Defaults to True.

    Returns:
        np.ndarray: The standard deviation of the spectrum
        
    Raises:
        TypeError: If power is not an integer or float.
    """
    
    if not isinstance(power, (float, int)):
        raise TypeError(f"[ERROR] Calc STD: Power must be an integer or float. Got {type(power).__name__}.")
    
    if power <= 0:
        print(f"[WARNING] Calc STD: Power should be a postive number. Got {power}. This may impact related operations.")
    
    # Compute the standard deviation of the spectrum
    moment_2 = calc_central_moment(signal, sample_rate, moment=2, power=power, pool=pool)
    std = np.sqrt(moment_2)
    
    std = Array.validate(array=std, direction="up")
    
    return std


def calc_skewness(
    signal: np.ndarray, 
    sample_rate: Optional[float] = None, 
    power: float = 2, 
    pool: bool = True
) -> np.ndarray:
    """
    Calculate the spectral skewness of the signal.
    
    Formula:
        skewness(X(k)) = moment(X(k), 3) / moment(X(k), 2) ^ 1.5

    Args:
        signal (np.ndarray): Time series signal.
        sample_rate (float, optional): The sampling rate of the signal. Defaults to None.
        power (int, optional): The power of the spectrum. Defaults to 2.
        pool (bool, optional): To pool the channel data. Defaults to True.

    Returns:
        np.ndarray: The skewness of the spectrum
        
    Raises:
        TypeError: If power is not an integer or float.
    """
    
    if not isinstance(power, (float, int)):
        raise TypeError(f"[ERROR] Calc Skewness: Power must be an integer or float. Got {type(power).__name__}.")
    
    if power <= 0:
        print(f"[WARNING] Calc Skewness: Power should be a postive number. Got {power}. This may impact related operations.")
    
    # Compute the skewness of the spectrum
    moment_2 = calc_central_moment(signal, sample_rate, moment=2, power=power, pool=pool)
    moment_3 = calc_central_moment(signal, sample_rate, moment=3, power=power, pool=pool)
    skewness = moment_3 / moment_2 ** 1.5
    
    skewness = Array.validate(array=skewness, direction="up")
    
    return skewness


def calc_kurtosis(
    signal: np.ndarray, 
    sample_rate: Optional[float] = None, 
    power: float = 2, 
    pool: bool = True
) -> np.ndarray:
    """
    Calculate the spectral kurtosis of the signal.
    
    Formula:
        kurtosis(X(k)) = moment(X(k), 4) / moment(X(k), 2) ^ 2 - 3

    Args:
        signal (np.ndarray): Time series signal.
        sample_rate (float, optional): The sampling rate of the signal. Defaults to None.
        power (int, optional): The power of the spectrum. Defaults to 2.
        pool (bool, optional): To pool the channel data. Defaults to True.

    Returns:
        np.ndarray: The kurtosis measure of the spectrum
        
    Raises:
        TypeError: If power is not an integer or float.
    """
    
    if not isinstance(power, (float, int)):
        raise TypeError(f"[ERROR] Calc Kurtosis: Power must be an integer or float. Got {type(power).__name__}.")
    
    if power <= 0:
        print(f"[WARNING] Calc Kurtosis: Power should be a postive number. Got {power}. This may impact related operations.")
    
    # Compute the kurtosis of the spectrum
    moment_2 = calc_central_moment(signal, sample_rate, moment=2, power=power, pool=pool)
    moment_4 = calc_central_moment(signal, sample_rate, moment=4, power=power, pool=pool)
    kurtosis = moment_4 / moment_2 ** 2 - 3
    
    kurtosis = Array.validate(array=kurtosis, direction="up")
    
    return kurtosis