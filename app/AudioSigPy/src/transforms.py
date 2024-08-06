from typing import Union, Tuple
import numpy as np
from . import array as Array


def sinc(signal: np.ndarray, filter: float = 1e-20, point: bool = False) -> np.ndarray:
    """
    Computes the normalized sinc function.

    Arguments:
        signal (np.ndarray): input signal
        filter (float, optional): threshold value to avoid division by zero (default 1e-20)
        point (bool, optional): boolean flag to apply filter at points (default False)

    Returns:
        np.ndarray: Computed sinc values.

    Raises:
        TypeError: if filter is not a float or int
        ValueError: if filter is not a positive value
    """
    
    if not isinstance(filter, (int, float)):
        raise ValueError(f"[ERROR] Sinc: Filter must be int or float. Got {type(filter).__name__}.")
    if filter <= 0:
        raise ValueError(f"[ERROR] Sinc: Filter must be positive. Got {filter}.")

    signal = Array.validate(signal)
    
    # Adjust only zeros
    if point:
        signal[np.abs(signal) == 0] = filter
    # Adjust a small interval around zeros
    else:
        signal[np.abs(signal) <= filter] = filter
        
    # Compute normalized sinc transform on signal
    signal = scalar_transform(signal=signal, scalar=np.pi, max_scale=False)
    signal = np.sin(signal) / signal
    
    return signal


def amplify(signal: np.ndarray, alpha: Union[int, float, np.ndarray]) -> np.ndarray:
    """
    Summary:
        Amplify the input signal by a factor alpha.
        
    Formula:
        y(t) = alpha * x(t)
        
    Arguments:
        signal (np.ndarray): The input signal array.
        alpha (int or float): The amplification factor.
        
    Returns:
        np.ndarray: The amplified signal.
        
    Raises:
        TypeError: If alpha is not an integer or float.
        ValueError: If alpha is not a positive number.
        
    Theory:
        The amplification of a signal is performed by multiplying the signal 
        by a constant factor alpha, resulting in y(t) = alpha * x(t).
    """
    
    if not isinstance(alpha, (int, float, np.ndarray)):
        raise TypeError(f"[ERROR] Amplify: Alpha must be an integer or float. Got {type(alpha).__name__}.")
    
    if isinstance(alpha, (int, float)):
        alpha = Array.validate(array=alpha, direction="up")
    else:
        alpha = Array.validate(array=alpha)
        
    if np.any(alpha <= 0):
        raise ValueError(f"[ERROR] Amplify: Alpha must be a non-zero positive real number. Got {alpha}.")

    signal = Array.validate(array=signal)
    signal = signal * alpha
    
    return signal


def integrate(signal: np.ndarray) -> np.ndarray:
    """
    Summary:
        Integrate the input signal.
        
    Formula:
        y(t) = x(t) + y(t-1)
        
    Arguments:
        signal (np.ndarray): The input signal array.
        
    Returns:
        np.ndarray: The integrated signal.
        
    Raises:
        TypeError: If signal is not a numpy ndarray.
        
    Theory:
        The integration of a signal is performed by summing the current value 
        with the previous integrated value, resulting in y(t) = x(t) + y(t-1).
    """
    
    signal = Array.validate(array=signal)
    
    y = [0]  # Initial condition: y(t-1) when t=0 is assumed to be 0.
    for x in signal:
        y.append(x + y[-1])
    new_signal = np.array(y[1:])
    
    return new_signal


def smooth(signal: np.ndarray, size: int = 3) -> np.ndarray:
    """
    Summary:
        Smooth the input signal using a moving average.
        
    Formula:
        y(t) = (x(t-T) + ... + x(t) + ... + x(t+T)) / (2T+1), where T is size
        
    Arguments:
        signal (np.ndarray): The input signal array.
        size (int, optional): The smoothing window size. Default is 3.
        
    Returns:
        np.ndarray: The smoothed signal.
        
    Raises:
        TypeError: If size is not an integer.
        ValueError: If size is negative or larger than half the length of the signal.
        
    Theory:
        The smoothing of a signal is performed by averaging over a window 
        centered at each point, resulting in y(t) = (x(t-T) + ... + x(t) + ... + x(t+T)) / (2T+1), where T is size.
    """
    
    signal = Array.validate(array=signal)
    if not isinstance(size, int):
        raise TypeError(f"[ERROR] Smooth: Size must be an integer. Got {type(size).__name__}.")
    if size < 0 or size >= signal.shape[0] // 2:
        raise ValueError(f"[ERROR] Smooth: Size must be a non-negative integer and smaller than half the input array length. Got {size} > {signal.shape[0] // 2}.")

    padding = np.zeros((size, signal.shape[1]))  # Zero padding
    signal = np.concatenate((padding, signal, padding), axis=0)
    
    new_signal = []
    for x in range(size, signal.shape[0] + size):
        sum_terms = sum(signal[x-size : x+size+1]) / (2 * size+1)
        new_signal.append(sum_terms)
        
    new_signal = Array.validate(array=new_signal)
    
    return new_signal


def logn_transform(
    signal: np.ndarray, 
    coefficient: float = 1, 
    base: float = 10, 
    filter: float = 1e-10,
    ref_axes=(0, 1)
) -> np.ndarray:
    """
    Transforms the input signal using logarithm base n and a coefficient.

    Arguments:
        signal (np.ndarray): input signal to be transformed
        coefficient (float, optional): scalar coefficient (default 1)
        base (float, optional): logarithmic base (default 10)
        filter (float, optional): filter level to handle 0 input. Default to 1e-10.
        reef_axes ((int, int), optional): Reference axes for alignment. Default to (0, 1)

    Returns:
        np.ndarray: Transformed signal.

    Raises:
        ValueError: if coefficient or base is not a float or int
    """
    
    if not isinstance(coefficient, (int, float)):
        raise ValueError(f"[ERROR] Log N Transform: Coefficient must be int or float. Got {type(coefficient).__name__}.")
    if not isinstance(base, (int, float)):
        raise ValueError(f"[ERROR] Log N Transform: Base must be int or float. Got {type(base).__name__}.")
    if not isinstance(filter, float):
        raise ValueError(f"[ERROR] Log N Transform: Filter must be float. Got {type(filter).__name__}.")
    
    # Validate signal
    signal = Array.validate(array=signal, ref_axes=ref_axes)
    
    # Set zeros to 1 for log calculation
    zero_mask = (np.abs(signal) <= filter)
    signal[zero_mask] = filter
    
    # Compute log transform
    signal = coefficient * np.emath.logn(n=base, x=signal)
    
    # signal[zero_mask] = 0
    
    return signal


def scalar_transform(
    signal: np.ndarray, 
    scalar: float = 0.9, 
    max_scale: bool = True, 
    pool: bool = True
) -> np.ndarray:
    """
    Transforms the input signal by scaling it with a scalar.

    Arguments:
        signal (np.ndarray): input signal to be transformed
        scalar (float, optional): scalar value (default 0.9)
        max_scale (bool, optional): scale with maximum value. Default to True.
        pool (bool, optional): Whether to pool the output.

    Returns:
        np.ndarray: Scaled signal.

    Raises:
        TypeError: If pool or max_scale is not of type bool.
        ValueError: if scalar is not a float or int.
    """
    
    if not isinstance(pool, bool):
        raise TypeError(f"[ERROR] Scalar transform: Pool must be bool. Got {type(pool).__name__}.")
    if not isinstance(scalar, (int, float)):
        if not isinstance(scalar, (list, np.ndarray)):
            raise ValueError(f"[ERROR] Scalar transform: Scalar must be int or float. Got {type(scalar).__name__}.")
    if not isinstance(max_scale, bool):
        raise ValueError(f"[ERROR] Scalar transform: Max scale must be bool. Got {type(max_scale).__name__}.")
    
    # Validate array
    signal = Array.validate(array=signal)
    
    # Scale the signal
    axis = None if pool else 0
    max_value = np.max(np.abs(signal), axis=axis) if max_scale else 1
    signal = signal * (scalar / max_value)
    
    return signal


def boxcox_transform(signal: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Applies Box-Cox power transformation to an array.

    Arguments:
        signal (int): The input signal.
        alpha (float): Power parameter, limited to [0, 1]. Default to 0.5.

    Returns:
        np.ndarray: The scaled array.

    Raises:
        TypeError: if alpha is not a float
        ValueError: if alpha is out of (0, 1] range
    """
    
    if not isinstance(alpha, (int, float)):
        raise ValueError(f"[ERROR] Box-Cox transform: Alpha must be float. Got {type(alpha).__name__}.")
    if alpha > 1 or alpha <= 0:
        raise ValueError(f"[ERROR] Box-Cox transform: Alpha must be in (0, 1]. Got {alpha}.")
    
    signal = Array.validate(array=signal)
    
    # Scale the index with Box-Cox power transformation
    signal = ((signal + 1) ** alpha - 1) / alpha
    
    return signal
    
    # # A linear series
    # index = Array.linseries(start=0, end=size, size=size, endpoint=False, coerce="int")
    
    # # Scale the index with Box-Cox power transformation
    # index = ((index + 1) ** alpha - 1) / alpha
    # index = np.floor((index) / np.max(index) * size).astype(int)
    # index = np.clip(index, 0, size - 1)
    
    # return index


def index_transform(size: int, alpha: float) -> np.ndarray:
    """
    Applies Box-Cox power transformation to linear indices.

    Arguments:
        size (int): Size of the index array
        alpha (float): Power parameter, limited to [0, 1]

    Returns:
        np.ndarray: Scaled index array.

    Raises:
        TypeError: if size is not an int
        ValueError: if size is non-positive
    """
    
    if not isinstance(size, int):
        raise TypeError(f"[ERROR] Index transform: Size must be int. Got {type(size).__name__}.")
    if size <= 0:
        raise ValueError(f"[ERROR] Index transform: Size must be positive. Got {size}.")
    
    # A linear series
    index = Array.linseries(start=0, end=size, size=size, endpoint=False, coerce="int")
    
    # Scale the index with Box-Cox power transformation
    index = boxcox_transform(signal=index, alpha=alpha)
    
    # Validate indices
    index = np.floor((index) / np.max(index) * size).astype(int)
    index = np.clip(index, 0, size - 1)
    
    return index


def centre_clip(
    signal: np.ndarray, 
    threshold: float = 0.3, 
    mode: str = "normal", 
    pool: bool = True
) -> np.ndarray:
    """
    Clip a signal by pushing values to zero and/or one.
    
    Methods:
        centre (normal): Pushes internal values to zero.
        outer (outer): Pushes outlier values to one (e.g., clipping a signal to amp=1)
        3-level (3level): Pushes values to one of 3-levels, -1, 0, or 1.
        trim (trim): Pushes outer values to threshold.

    Args:
        signal (np.ndarray): The input signal.
        threshold (float, optional): The clipping threshold. Defaults to 0.3.
        mode (str, optional): The clipping mode, one of [normal, outer, 3level, trim]. Defaults to "normal".
        pool (bool, optional): Pool the channels. Defaults to True.

    Returns:
        np.ndarray: The clipped signal.
    """
    
    if not isinstance(threshold, (int, float)):
        raise TypeError(f"[ERROR] Centre Clip: Threshold must be an integer or float. Got {type(threshold).__name__}.")
    if not isinstance(mode, str):
        raise TypeError(f"[ERROR] Centre Clip: Mode must be a string. Got {type(mode).__name__}.")
    if not isinstance(pool, bool):
        raise TypeError(f"[ERROR] Centre Clip: Pool must be a boolean. Got {type(pool).__name__}.")
    
    if threshold <= 0 or threshold > 1:
        raise ValueError(f"[ERROR] Centr Clip: Threshold must be in (0, 1]. Got {threshold}.")
    mode = mode.lower()
    if mode not in ["normal", "outer", "3level", "trim"]:
        raise ValueError(f"[ERROR] Centre Clip: Mode must be one of [normal, outer, 3level, trim]. Got {mode}.")
    
    signal = Array.validate(array=signal)
    
    if pool:
        # Set the threshold value
        threshold = threshold * np.max(np.abs(signal))
        
        # Normal centre clipping
        if mode == "normal":
            signal[np.abs(signal) <= threshold] = 0
            signal = signal + np.sign(-signal) * threshold
            
        # 3level centre clipping -> [-1, 0, 1]
        elif mode == "3level":
            signal[np.abs(signal) <= threshold] = 0
            signal = np.sign(signal) * 1
            
        # Pushes oouter values to +/- 1
        elif mode == "outer":
            signal[signal >= threshold] = 1
            signal[signal <= -threshold] = -1
            
        # Clips outer values down to threshold
        elif mode == "trim":
            signal[signal >= threshold] = threshold
            signal[signal <= -threshold] = -threshold
    else:
        # Set the threshold value
        threshold = threshold * np.max(np.abs(signal), axis=0)
        
        for c in range(signal.shape[1]):
            
            # Normal centre clipping
            if mode == "normal":
                signal[np.abs(signal[:, c]) <= threshold[c], c] = 0
                signal[:, c] = signal[:, c] + np.sign(-signal[:, c]) * threshold[c]
                
            # 3level centre clipping -> [-1, 0, 1]
            elif mode == "3level":
                signal[np.abs(signal[:, c]) <= threshold[c], c] = 0
                signal[:, c] = np.sign(signal[:, c]) * 1
                
            # Pushes oouter values to +/- 1
            elif mode == "outer":
                signal[signal[:, c] >= threshold[c], c] = 1
                signal[signal[:, c] <= -threshold[c], c] = -1
                
            # Clips outer values down to threshold
            elif mode == "trim":
                signal[signal[:, c] >= threshold[c], c] = threshold[c]
                signal[signal[:, c] <= -threshold[c], c] = -threshold[c]
    
    return signal


class Quantiser():
    """
    A class used to quantize and dequantize signals.

    Attributes
        __bits (int): The number of bits used for quantization.
        __levels (int): The number of quantization levels derived from bits.

    Methods
        get_bit_count: Returns the number of bits used for quantization.
        get_level_count: Returns the number of levels for quantization.
        set_bit_count: Sets the number of bits for quantization and updates levels.
        set_level_count: Updates the number of levels based on bits.
        quantise_signal: Quantizes the input signal array.
        dequantise_signal: Dequantizes the input signal array.
    """
    
    def __init__(self, bits: int = 32):
        """
        Initializes the Quantiser with a specified number of bits.

        Arguments:
            bits (int, optional): The number of bits for quantization (default is 32).
        """
        
        self.set_bit_count(bits=bits)
        self.set_level_count()
        
        
    def __quantise(self, signal: np.ndarray, quanta: Union[int, float]) -> np.ndarray:
        """
        Private method to quantise a signal.
        
        Formula:
            y(t) = ⌊0.5 + x(t) / q⌋

        Arguments:
            signal (np.ndarray): The signal to be quantized.
            quanta (int or float): The quantization step size.

        Returns:
            int or float: The quantized signal.
        """
        
        signal = Array.validate(array=signal, direction="up")
        signal = np.floor(signal / quanta + 1 / 2)
        
        return signal
    
    
    def __dequantise(self, signal: np.ndarray, quanta: Union[int, float]) -> np.ndarray:
        """
        Private method to dequantize a single signal.
        
        Formula:
            y(t) = q * x(t)

        Arguments:
            signal (np.ndarray): The quantized signal.
            quanta (int or float): The quantization step size.

        Returns:
            int or float: The dequantized signal.
        """
        
        signal = Array.validate(array=signal, direction="up")
        
        signal = amplify(signal=signal, alpha=quanta)
        
        return signal
    
    
    def get_bit_count(self) -> int:
        """
        Gets the number of bits used for quantization.

        Returns:
            int: The number of bits.
            
        Raise:
            ValueError: If bit count is not set.
        """
        
        if self.__bits is None:
            raise ValueError("[ERROR] Get bit count: Bit count is not set.")
        
        return self.__bits
    
    
    def get_level_count(self) -> int:
        """
        Gets the number of quantization levels.

        Returns:
            int: The number of levels.
            
        Raise:
            ValueError: If level count is not set.
        """
        
        if self.__levels is None:
            raise ValueError("[ERROR] Get level count: Level count is not set.")
        
        return self.__levels
    
    
    def set_bit_count(self, bits: int) -> None:
        """
        Sets the number of bits and updates the quantization levels.

        Arguments:
            bits (int): The number of bits for quantization.

        Raises:
            ValueError: If bits is not a positive integer.
        """
        
        if not isinstance(bits, int) or bits <= 0:
            raise ValueError(f"[ERROR] Set bit count: Bit count must be a positive integer. Got {type(bits).__name__} {bits}.")
        
        self.__bits = bits
        self.set_level_count()
        
        
    def set_level_count(self) -> None:
        """
        Updates the number of quantization levels based on the number of bits.
        
        Formula:
            levels = 2 ^ bits - 1
        """
        
        self.__levels = 2 ** self.get_bit_count() - 1
        
        
    def quantise_signal(self, signal: np.ndarray, pool: bool = True) -> Tuple[np.ndarray, float, float]:
        """
        Quantizes the input signal array.
        
        Formula:
            quanta = |max(x) - min(x)| / L
            y(t) = ⌊0.5 + x(t) - min(x) / quanta⌋

            Where:
                L is the number of quantization levels.

        Arguments:
            signal (np.ndarray): The input signal array to be quantized.
            pool (bool, optional): Pool channels or not.

        Returns:
            Tuple[np.ndarray, float, float]: A tuple containing the quantized signal, the quantization 
                                             step size (quanta), and the amplitude shift.

        Raises:
            TypeError: If the signal is not a numpy ndarray or pool is not bool.
            ValueError: If the signal array is empty.
        """
        
        if not isinstance(pool, bool):
            raise TypeError(f"[ERROR] Quantise Signal: Pool must be a boolean. Got {type(pool).__name__}.")
        
        # Validate signal array
        signal = Array.validate(array=signal)
        
        # Determine the dynamic range of the signal and amplitude shift in order to fit the signal into target number of levels
        axis = None if pool else 0
        amp_max = np.max(signal, axis=axis)
        amp_min = np.min(signal, axis=axis)
        amp_range = np.abs(amp_max - amp_min)
        amp_shift = amp_min
        
        # Calculate the quanta (unit step of quantisation)
        quanta = amp_range / self.get_level_count()
        
        # Perform quantisation of the signal
        new_signal = np.zeros(signal.shape)
        for t in range(signal.shape[0]):
            new_signal[t, :] = self.__quantise(signal[t, :] - amp_shift, quanta).flatten()
            
        return new_signal, quanta, amp_shift
    
    
    def dequantise_signal(
        self, 
        signal: np.ndarray, 
        quanta: Union[int, float], 
        amp_shift: Union[int, float],
        pool: bool = True
    ) -> np.ndarray:
        """
        Dequantizes the input signal array using given quanta and amplitude shift.
        
        Formula:
            y(t) = quanta * x(t) + shift

        Arguments:
            signal (np.ndarray): The quantized signal array to be dequantized.
            quanta (int or float): The quantization step size used during quantization.
            amp_shift (int or float): The amplitude shift applied during quantization.

        Returns:
            np.ndarray: The dequantized signal array.
            pool (bool): Pool channels or not.

        Raises:
            TypeError: If the signal is not a numpy ndarray or pool is not bool.
            ValueError: If the signal array is empty.
        """
        
        if not isinstance(pool, bool):
            raise TypeError(f"[ERROR] Quantise Signal: Pool must be a boolean. Got {type(pool).__name__}.")
        
        if not isinstance(quanta, (int, float, np.ndarray)):
            raise TypeError(f"[ERROR] Dequantise signal: Quanta must be an integer or float. Got {type(quanta).__name__}.")
        if not isinstance(amp_shift, (int, float, np.ndarray)):
            raise TypeError(f"[ERROR] Dequantise signal: Amp shift must be an integer or float. Got {type(amp_shift).__name__}.")
        
        # Validate signal array
        signal = Array.validate(array=signal)
        
        # Perform dequantisation using given quanta and amplitude shift
        new_signal = np.zeros(signal.shape)
        for t in range(signal.shape[0]):
            new_signal[t, :] = self.__dequantise(signal[t, :], quanta).flatten() + amp_shift
            
        return new_signal