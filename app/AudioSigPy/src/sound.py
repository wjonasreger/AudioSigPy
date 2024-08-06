from typing import Optional, Union, Tuple
import numpy as np
import pyloudnorm
from . import array as Array
from . import transforms as Transforms
from . import frequency as Frequency


def calc_rms(signal: np.ndarray, pool: bool = True) -> np.ndarray:
    """
    Calculate the root mean square (RMS) of the input signal.
    
    Formula:
        If pooled:
            RMS = sqrt((1 / N) * sum(x_i^2))
            
            where N is the total number of samples in the signal and x_i represents each sample value.
        
        If not pooled:
            RMS_c = sqrt((1 / N_c) * sum(x_{i,c}^2))
            
            where N_c is the number of samples in each channel c, and x_{i,c} represents each sample 
            value in channel c.

    Arguments:
        signal (np.ndarray): The input signal.
        pool (bool, optional): Whether to pool the output.

    Returns:
        np.ndarray: The RMS value of the signal.
    
    Raises:
        TypeError: If pool is not of type bool.
    """
    
    if not isinstance(pool, bool):
        raise TypeError(f"[ERROR] Culculate RMS: Pool must be bool. Got {type(pool).__name__}.")

    # Validate signal
    signal = Array.validate(array=signal)
    
    # Pooling allows calculation over all samples vs samples within each channel
    if pool:
        rms = np.sqrt(np.mean(signal ** 2, axis=None))
    else:
        rms = np.sqrt(np.mean(signal ** 2, axis=0))
        
    rms = Array.validate(array=rms, direction="up")
        
    return rms


def calc_spl(
    signal: np.ndarray, 
    ref_pressure: float = 2e-5, 
    pool: bool = True, 
    raw: bool = False
) -> np.ndarray:
    """
    Calculate the sound pressure level (SPL) of the input signal.
    
    Formula:
        SPL = 20 * log_10(RMS(x) / p)
        
        where p = 2e-5 (reference pressure)

    Arguments:
        signal (np.ndarray): The input signal.
        ref_pressure (float, optional): The reference pressure.
        pool (bool, optional): Whether to pool the output.
        raw (bool, optional): To compute raw intensity. Default to False.

    Returns:
        np.ndarray: The SPL value of the signal.
    
    Raises:
        TypeError: If pool or raw is not of type bool.
        ValueError: If ref_pressure is not positive.
    """
    
    if not isinstance(ref_pressure, (int, float)) or ref_pressure <= 0:
        raise ValueError(f"[ERROR] Calculate SPL: Reference pressure must be a positive number. Got {type(ref_pressure).__name__} {ref_pressure}.")
    if not isinstance(pool, bool):
        raise TypeError(f"[ERROR] Calculate SPL: Pool must be bool. Got {type(pool).__name__}.")
    if not isinstance(raw, bool):
        raise TypeError(f"[ERROR] Calculate SPL: Raw must be bool. Got {type(raw).__name__}.")

    # RMS of signal is the pressure
    pressure = calc_rms(signal=signal, pool=pool)
    # Compute the theoretical value of SPL
    ref_pressure = 1 if raw else ref_pressure
    spl = Transforms.logn_transform(signal=pressure / ref_pressure, coefficient=20, base=10)
    
    spl = Array.validate(array=spl, direction="up")
    
    return spl


def calc_sil(
    signal: np.ndarray, 
    ref_intensity: float = 1e-12, 
    pool: bool = True, 
    raw: bool = False
) -> np.ndarray:
    """
    Calculate the sound intensity level (SIL) of the input signal.
    
    Formula:
        SIL = 10 * log_10(RMS(x)^2 / (impedance * i))
        
        where i = 1e-12 (reference intensity), and impedance = 2e-5^2 / 1e-12 (i.e., p^2 / i)

    Arguments:
        signal (np.ndarray): The input signal.
        ref_intensity (float, optional): The reference intensity.
        pool (bool, optional): Whether to pool the output.
        raw (bool, optional): To compute raw intensity. Default to False.

    Returns:
        np.ndarray: The SIL value of the signal.
    
    Raises:
        TypeError: If pool or raw is not of type bool.
        ValueError: If ref_intensity is not positive.
    """
    
    if not isinstance(ref_intensity, (int, float)) or ref_intensity <= 0:
        raise ValueError(f"[ERROR] Calculate SIL: Reference intensity must be a positive number. Got {type(ref_intensity).__name__} {ref_intensity}.")
    if not isinstance(pool, bool):
        raise TypeError(f"[ERROR] Calculate SIL: Pool must be bool. Got {type(pool).__name__}.")
    if not isinstance(raw, bool):
        raise TypeError(f"[ERROR] Calculate SIL: Raw must be bool. Got {type(raw).__name__}.")

    # Compute impedance based on p_ref and I_ref classical constant values
    impedance = 1.0 if raw else (2e-5)**2 / (1e-12)
    ref_intensity = 1.0 if raw else ref_intensity
    intensity = calc_rms(signal=signal, pool=pool)**2 / impedance
    
    # Compute the theoretical value of SIL
    sil = Transforms.logn_transform(signal=intensity / ref_intensity, coefficient=10, base=10)
    
    sil = Array.validate(array=sil, direction="up")
    
    return sil


def calc_loudness(
    signal: np.ndarray, 
    sample_rate: Optional[Union[int, float]] = None, 
    scale: bool = True, 
    target_rms: float = 0.1, 
    pool: bool = True
) -> np.ndarray:
    """
    Calculate the loudness of the input signal in LUFS.

    Arguments:
        signal (np.ndarray): The input signal.
        sample_rate (int or float, optional): The sample rate.
        scale (bool, optional): Whether to scale the signal.
        target_rms (float, optional): The target RMS value.
        pool (bool, optional): Whether to pool the output.

    Returns:
        np.ndarray: The loudness value of the signal.
    
    Raises:
        TypeError: If pool is not of type bool.
                   If scale is not of type bool.
        ValueError: If target_rms is not positive.
    """
    
    if not isinstance(target_rms, (int, float)) or target_rms <= 0:
        raise ValueError(f"[ERROR] Calculate loudness: Target RMS must be a positive number. Got {type(target_rms).__name__} {target_rms}.")
    if not isinstance(pool, bool):
        raise TypeError(f"[ERROR] Calculate loudness: Pool must be bool. Got {type(pool).__name__}.")
    if not isinstance(scale, bool):
        raise TypeError(f"[ERROR] Calculate loudness: Scale must be bool. Got {type(scale).__name__}.")

    sample_rate = Frequency.val_sample_rate(sample_rate=sample_rate)
    
    # Validate signal
    signal = Array.validate(array=signal)
    
    # Set up the loudness meter
    meter = pyloudnorm.Meter(rate=sample_rate)
    
    if scale:
        signal, _ = rms_transform(signal=signal, target_rms=target_rms, pool=pool)
    
    # Pooling allows calculation over all samples vs samples within each channel
    if pool:
        loudness = meter.integrated_loudness(data=signal)
    else:
        loudness = []
        for i in range(signal.shape[1]):
            loudness.append(meter.integrated_loudness(data=signal[:, i]))
            
    loudness = Array.validate(array=loudness, direction="up")
    
    return loudness


def calc_zcr(signal: np.ndarray, pool: bool = True) -> float:
    """
    Calculate the zero-crossing rate (ZCR) of the input signal.

    Arguments:
        signal (np.ndarray): The input signal.
        pool (bool, optional): Whether to pool the output.

    Returns:
        float: The zero-crossing rate.
    
    Raises:
        TypeError: If pool is not of type bool.
    """
    
    if not isinstance(pool, bool):
        raise TypeError(f"[ERROR] Calculate ZCR: Pool must be bool. Got {type(pool).__name__}.")
    
    signal = Array.validate(array=signal)
    
    # Compute +/- directions between each sample
    directions = []
    for sample in range(1, signal.shape[0]):
        direction = np.abs(np.sign(signal[sample-1, :]) - np.sign(signal[sample, :]))
        directions.append(direction)
        
    if pool:
        zcr = np.mean(directions, axis=None) / 2
    else:
        zcr = np.mean(directions, axis=0) / 2
        
    zcr = Array.validate(array=zcr, direction="up")
    
    return zcr


def rms_transform(
    signal: np.ndarray, 
    target_rms: float = 0.1, 
    pool: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Adjusts the root mean square (RMS) of the input signal to the target RMS value.

    Arguments:
        signal (np.ndarray): input signal to be transformed
        target_rms (float, optional): target RMS value (default 0.1)
        pool (bool, optional): Whether to pool the output.

    Returns:
        (np.ndarray, float): Transformed signal and the scalar used for transformation.

    Raises:
        TypeError: If pool is not of type bool.
        ValueError: if target_rms is not a float or int
    """
    
    if not isinstance(pool, bool):
        raise TypeError(f"[ERROR] RMS transform: Pool must be bool. Got {type(pool).__name__}.")
    if not isinstance(target_rms, (int, float)):
        raise ValueError(f"[ERROR] RMS transform: Target RMS must be int or float. Got {type(target_rms).__name__}.")
    
    # Validate signal
    signal = Array.validate(array=signal)
    
    # Compute to scalar to transform signal
    if pool:
        scalar = target_rms * np.sqrt(signal.shape[0] * signal.shape[1] / np.sum(signal ** 2))
    else:
        scalar = target_rms * np.sqrt(signal.shape[0] / np.sum(signal ** 2, axis=0))
    
    # Transform signal
    signal = Transforms.scalar_transform(signal=signal, scalar=scalar, max_scale=False, pool=pool)
    
    return signal, scalar


def spl_transform(
    signal: np.ndarray, 
    target_spl: float, 
    ref_pressure: float = 2e-5, 
    pool: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Adjusts the sound pressure level (SPL) of the input signal to the target SPL value.

    Arguments:
        signal (np.ndarray): input signal array
        target_spl (float): target SPL value
        ref_pressure (float, optional): The reference pressure.
        pool (bool, optional): Whether to pool the output.

    Returns:
        (np.ndarray, float): Scaled signal and the scalar used for scaling.

    Raises:
        TypeError: If pool is not of type bool.
        ValueError: If ref_pressure is not positive.
        ValueError: if target_spl is not a float or int
    """
    
    if not isinstance(ref_pressure, (int, float)) or ref_pressure <= 0:
        raise ValueError(f"[ERROR] SPL transform: Reference pressure must be a positive number. Got {type(ref_pressure).__name__} {ref_pressure}.")
    if not isinstance(pool, bool):
        raise TypeError(f"[ERROR] SPL transform: Pool must be bool. Got {type(pool).__name__}.")
    if not isinstance(target_spl, (int, float)):
        raise ValueError(f"[ERROR] SPL transform: Target SPL must be int or float. Got {type(target_spl).__name__}.")

    signal = Array.validate(array=signal)
    
    # Compute pressure and scalar for scaling the signal
    pressure = calc_rms(signal=signal, pool=pool)
    scalar = 10 ** (target_spl / 20) * ref_pressure / pressure
    
    # Scale the signal
    signal = Transforms.scalar_transform(signal=signal, scalar=scalar, max_scale=False, pool=pool)
    
    return signal, scalar


def sil_transform(
    signal: np.ndarray, 
    target_sil: float, 
    ref_intensity: float = 1e-12, 
    pool: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Adjusts the sound intensity level (SIL) of the input signal to the target SIL value.

    Arguments:
        signal (np.ndarray): input signal array
        target_sil (float): target SIL value
        ref_intensity (float, optional): The reference intensity.
        pool (bool, optional): Whether to pool the output.

    Returns:
        (np.ndarray, float): Scaled signal and the scalar used for scaling.

    Raises:
        TypeError: If pool is not of type bool.
        ValueError: If ref_intensity is not positive.
        ValueError: if target_sil is not a float or int
    """
    
    if not isinstance(ref_intensity, (int, float)) or ref_intensity <= 0:
        raise ValueError(f"[ERROR] SIL transform: Reference intensity must be a positive number. Got {type(ref_intensity).__name__} {ref_intensity}.")
    if not isinstance(pool, bool):
        raise TypeError(f"[ERROR] SIL transform: Pool must be bool. Got {type(pool).__name__}.")
    if not isinstance(target_sil, (int, float)):
        raise ValueError(f"[ERROR] SIL transform: Target SIL must be int or float. Got {type(target_sil).__name__}.")

    signal = Array.validate(array=signal)
    
    # Compute impedance based on p_ref and I_ref classical constant values
    impedance = (2e-5)**2 / (1e-12)
    
    # Compute intensity and scalar for scaling the signal
    intensity = calc_rms(signal=signal, pool=pool)**2 / impedance
    scalar = 10 ** (target_sil / 10) * ref_intensity / intensity
    
    # Scale the signal
    signal = Transforms.scalar_transform(signal=signal, scalar=scalar, max_scale=False, pool=pool)
    
    return signal, scalar