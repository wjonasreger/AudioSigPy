from typing import Optional, Tuple, Union
import numpy as np
from . import array as Array
from . import windows as Windows
from . import frequency as Frequency


def dft(array: np.ndarray, inverse: bool = False) -> np.ndarray:
    """
    Compute the Discrete Fourier Transform (DFT) of a given array.
    
    Formula:
        # Forward DFT
        dft_array = array * exp(1j * (2 * π * freq_idx * freq_idx^T) / N)

        # Inverse DFT
        idft_array = array * exp(-1j * (2 * π * freq_idx * freq_idx^T) / N)


    Arguments:
        array (np.ndarray): The input array.
        inverse (bool, optional): Whether to compute the inverse DFT. Default is False.

    Returns:
        np.ndarray: The transformed array.
    
    Raises:
        TypeError: If array is not a numpy array.
    """
    
    if not isinstance(array, np.ndarray):
        raise TypeError(f"[ERROR] DFT: Array must be a numpy array. Got {type(array).__name__}.")

    # Coerce to complex if inverse 
    if inverse:
        array = Array.validate(array=array, coerce="complex64")
    else:
        array = Array.validate(array=array)
    
    # Create frequency index
    freq_idx = Array.linseries(start=0, end=array.shape[0], endpoint=False)
    
    # Compute phase angles 
    # phase_angles = 2 * np.pi * np.outer(freq_idx, freq_idx) / array.shape[0]
    phase_angles = 2 * np.pi * freq_idx * freq_idx.transpose() / array.shape[0]
    
    # Compute dft array
    complex_coef = 1.j if inverse else -1.j
    exp_matrix = np.exp(complex_coef * phase_angles)
    array = np.dot(exp_matrix, array)
    
    # Coerce array to real domain if inverse
    if inverse:
        array = np.real(array)
    
    return array


def fft(
    array: np.ndarray, 
    size: Optional[int] = None, 
    axis: int = 0, 
    iter_axis: int = 1, 
    inverse: bool = False, 
    mode: str = "stable", 
    top: bool = True
) -> np.ndarray:
    """
    Compute the one-dimensional n-point discrete Fourier Transform (DFT)
    using the Cooley-Tukey FFT algorithm along the specified axis.
    
    Formula:
        Let f(x) be a function, where x represents the input array of length N. The Fast Fourier Transform (FFT) is defined recursively as follows (Experimental version):

        1. Base Case:
            If N = 1, then:
            FFT(f(x)) = f(x)

        2. Recursive Case:
            Otherwise, decompose f(x) into even and odd indexed elements:
            f_even(x) = f(x_{2k}) for k = 0, 1, ..., floor(N/2) - 1
            f_odd(x) = f(x_{2k+1}) for k = 0, 1, ..., floor(N/2) - 1

            Compute the FFT of these subsequences recursively:
            F_even(k) = FFT(f_even(x))
            F_odd(k) = FFT(f_odd(x))

        3. Combine the results:
            F(k) = F_even(k) + W_N^k * F_odd(k)
            F(k + N/2) = F_even(k) + W_N^k * F_odd(k)
            where W_N^k = exp(-2*j*pi*k/N) is the twiddle factor (Forward FFT).

        4. Repeat steps 1 to 3 until the base case is reached.
    
    Arguments:
        array (np.ndarray): Input array to be transformed. Must be coercible to np.ndarray.
        size (int, optional): The length of the transformed axis of the output. Default is None.
        axis (int, optional): Axis over which to compute the FFT. Default is 0.
        iter_axis (int, optional): Axis used for determining array direction. Default is 1.
        inverse (bool, optional): If True, compute the inverse FFT. Default is False.
        mode (str, optional): FFT mode. Must be either 'experimental' or 'stable'. Default is 'stable'.
        top (bool, optional): Internal flag for recursive FFT calls. Default is True.
    
    Returns:
        np.ndarray: Transformed array with the same type as the input.
    
    Raises:
        ValueError: If mode is not 'experimental' or 'stable'.
        TypeError: If array is not of a supported type.
        
    Example:
        Special use case when using inverse FFT:
        The inverse method doesn't automatically scale the data back to original 
        scale - there are a few ways to achieve this. Below is an example using 
        the root mean square error as the scalar.
            ```
            np.random.seed(0)
            x = np.random.random(20)
            rms = calc_rms(array=x, pool=False)
            y = fft(array=x, axis=0, mode="experimental")
            y = fft(array=y, axis=0, inverse=True, mode="experimental")
            y = np.real(y)
            y, _ = rms_transform(array=y, target_rms=rms)
            ```
    """
    
    if not isinstance(array, (np.ndarray, list)):
        raise TypeError(f"[ERROR] FFT: Array must be a numpy ndarray or list. Got {type(array).__name__}.")
    
    if not isinstance(size, (int, type(None))):
        raise TypeError(f"[ERROR] FFT: Size must be an integer or None. Got {type(size).__name__}.")
    
    if not isinstance(axis, int):
        raise TypeError(f"[ERROR] FFT: Axis must be an integer. Got {type(axis).__name__}.")
    
    if not isinstance(iter_axis, int):
        raise TypeError(f"[ERROR] FFT: Iterative axis must be an integer. Got {type(iter_axis).__name__}.")
    
    if not isinstance(inverse, bool):
        raise TypeError(f"[ERROR] FFT: Inverse must be a boolean. Got {type(inverse).__name__}.")
    
    if not isinstance(mode, str):
        raise TypeError(f"[ERROR] FFT: Mode must be a string. Got {type(mode).__name__}.")
    mode = mode.lower()
    if mode not in ["experimental", "stable"]:
        raise ValueError(f"[ERROR] FFT: Mode must be one of experimental or stable. Got {mode}.")
    
    # Ensure validation as recursive arrays get smaller
    if array.ndim == 1:
        array = Array.unsqueeze(array=array)
    direction = "up" if array.shape[axis] < array.shape[iter_axis] else "down"
    array = Array.validate(array=array, coerce="complex128", direction=direction)
    
    # Perform experimental fft algorithm (recursive)
    if mode == "experimental":
        # Get a size if none given
        if size is None:
            size = array.shape[axis]
            
        # Adjust array size to given fft size
        if array.shape[axis] != size:
            shape = list(array.shape)
            index = [slice(None)] * array.ndim
            if shape[axis] > size:
                index[axis] = slice(0, size)
                array = array[tuple(index)]
            else:
                index[axis] = slice(0, shape[axis])
                shape[axis] = size
                temp = np.zeros(shape, array.dtype.char)
                temp[tuple(index)] = array
                array = temp
        
        # L is original length, N is length of next power of 2
        L = N = array.shape[axis]
        
        # Pad the input array to the next power of 2
        if N & (N - 1) != 0:
            target_length = 1 << (N - 1).bit_length()
            shape = list(array.shape)
            shape[axis] = target_length - N
            padding = np.zeros(shape, dtype=array.dtype)
            array = np.concatenate([array, padding], axis=axis)
            N = target_length
        
        # When array is length 1 at end of recursion depth
        if N <= 1:
            return array
        
        # FFT algorithm: recursively divide and conquer
        even = fft(
            array=np.take(array, indices=range(0, N, 2), axis=axis), 
            size=None,
            axis=axis, 
            iter_axis=iter_axis, 
            inverse=inverse,
            top=False
        )
        odd = fft(
            array=np.take(array, indices=range(1, N, 2), axis=axis), 
            size=None,
            axis=axis, 
            iter_axis=iter_axis, 
            inverse=inverse,
            top=False
        )
        
        # adjust size of factor array to be twice as large as even/odd
        M = 2 * even.shape[axis]
        if not inverse:
            factor = np.exp(-2j * np.pi * np.arange(M) / M).reshape((-1, 1))
        else:  # is inverse
            factor = np.exp(2j * np.pi * np.arange(M) / M).reshape((-1, 1))
        
        even_factor = np.take(factor, indices=range(0, int(M/2), 1), axis=axis)
        odd_factor = np.take(factor, indices=range(int(M/2), M, 1), axis=axis)
        
        # Update array with even and odd factors
        array = np.concatenate([
                even + even_factor * odd,
                even + odd_factor * odd
            ], axis=axis)
        
        # Clip and scale final array to match closely to original signal
        if top and inverse:
                array = np.take(array, indices=range(0, L, 1), axis=axis)
                array = array / L
                
    # Use numpy's fft functions
    elif mode == "stable":
        if inverse:
            array = np.fft.ifft(a=array, n=size, axis=axis)
        else:
            array = np.fft.fft(a=array, n=size, axis=axis)
    
    return array


def stft(
    signal: np.ndarray, 
    sample_rate: Optional[float] = None, 
    fft_size: int = 512, 
    window_size: float = 0.01, 
    overlap: float = 0.5, 
    window: str = "hamming",
    transform: bool = True,
    clip: bool = True
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """
    Calculate the STFT matrix of a signal (i.e., spectrogram).

    Theory:
        Applies a shift window over the signal in the time domain and computes the 
        spectra of the window to analyze the frequencies in the frequency domain. 

    Args:
        signal (np.ndarray): The input signal.
        sample_rate (float, optional): The sample rate. Defaults to None.
        fft_size (int, optional): The FFT size of frequency domain. Defaults to 512.
        window_size (float, optional): The short time proportional window size. Defaults to 0.01.
        overlap (float, optional): The short time proportional overlap. Defaults to 0.5.
        window (str, optional): The window function name. Defaults to "hamming".
        transform (bool, optional): To transform the matrix with FFT. Defaults to True.
        clip (bool, optional): The clip the matrix in half. Defaults to True.

    Returns:
        Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]: The output STFT matrix or spectrogram as either (spectra, magnitude, phase) or time-domain signal, both with shape (F, T, C).
        
    Raises:
        TypeError: If window size or overlap is not float, FFT size is not int, or transform is not bool.
        ValueError: If window size is out of (0, 1) or overlap is out of (0, 0.95).
    """
    
    if not isinstance(window_size, float):
        raise TypeError(f"[ERROR] STFT: Window size must be float. Got {type(window_size).__name__}.")
    if not isinstance(fft_size, int):
        raise TypeError(f"[ERROR] STFT: FFT size must be int. Got {type(fft_size).__name__}.")
    if not isinstance(overlap, float):
        raise TypeError(f"[ERROR] STFT: Overlap must be float. Got {type(overlap).__name__}.")
    if not isinstance(transform, bool):
        raise TypeError(f"[ERROR] STFT: Transform must be boolean. Got {type(transform).__name__}.")
    
    if window_size <= 0 or window_size >=1:
        raise ValueError(f"[ERROR] STFT: Window size must be in (0, 1). Got {window_size}.")
    
    if overlap > 0.95:
        overlap = 0.95
        print(f"[WARNING] STFT: Overlap should be 0.96 at most for stability purposes. Got {overlap}. Resetting overlap value to 0.95.")
        
    if overlap < 0 or overlap > 0.95:
        raise ValueError(f"[ERROR] STFT: Overlap must be in [0, 0.95]. Got {overlap}.")
    
    # Validate signal and sample rate
    signal = Array.validate(array=signal)
    sample_rate = Frequency.val_sample_rate(sample_rate=sample_rate)
    sample_count, channel_count = signal.shape
    
    # Calculate frame parameters
    frame_size = round(sample_rate * window_size)
    shift_size = round(frame_size * (1 - overlap))
    frame_count = int(np.ceil((sample_count - frame_size) / shift_size + 1))
    
    # Zero pad signal data
    zero_padding = np.zeros((frame_count * frame_size - sample_count, channel_count))
    signal = np.concatenate([signal, zero_padding], axis=0)
    
    # Allocate space for frames data and generate window function
    frames = np.zeros((frame_size * frame_count, channel_count))
    window = Windows.choose_window(name=window, length=frame_size, mode="run")
    
    # Calculate frames
    for f in range(frame_count):
        i, j = f * frame_size, f * shift_size
        d = frame_size
        frames[i:i + d] = signal[j:j + d] * window
        
    # Reshape frames data into time-frequency domain structure
    frames = frames.reshape((frame_size, frame_count, channel_count), order="F")
    
    # Calculate the spectra
    if transform:
        spectra, magnitude, phase = calc_spectra(signal=frames, size=fft_size, stft=True, clip=clip)
        return spectra, magnitude, phase
    # Return untransformed frames
    else:
        return frames
    

def istft(
    frames: np.ndarray,
    frame_size: int,
    shift_size: int,
    sample_count: int,
    transform: bool = True,
    window: str = "hamming"
) -> np.ndarray:
    """
    Calculate the inverse STFT matrix of a spectrogram (i.e., time-domian signal).

    Args:
        frames (np.ndarray): The spectrogram matrix.
        frame_size (int): The frame size (upper bounded by FFT size). 
        shift_size (int): The shift size (upper bounded by frame size).
        sample_count (int): The sample count of the signal.
        transform (bool, optional): To do inverse FFT on spectral data. Defaults to True.
        window (str, optional): The window function name. Defaults to "hamming".

    Returns:
        np.ndarray: The time-domain signal.
        
    Raises:
        TypeError: If frame size, shift size, or sample count is not int, or transform is not bool.
        ValueError: If shift size > frame size, or sample count is out of (0, frame count * frame size].
    """
    
    if not isinstance(frame_size, int):
        raise TypeError(f"[ERROR] ISTFT: Frame size must be an integer. Got {type(frame_size).__name__}.")
    if not isinstance(shift_size, int):
        raise TypeError(f"[ERROR] ISTFT: Shift size must be an integer. Got {type(shift_size).__name__}.")
    if not isinstance(sample_count, int):
        raise TypeError(f"[ERROR] ISTFT: Sample count must be an integer. Got {type(sample_count).__name__}.")
    if not isinstance(transform, bool):
        raise TypeError(f"[ERROR] ISTFT: Transform must be a boolean. Got {type(transform).__name__}.")
    
    # Set size parameters from frames shape
    fft_size = frames.shape[0]
    frame_count = frames.shape[1]
    channel_count = frames.shape[2]
    
    # Validate size parameters based on bounds
    if shift_size > frame_size:
        raise ValueError(f"[ERROR] ISTFT: Shift size must be less than or equal to Frame size. Got {shift_size} > {frame_size}.")
    if sample_count <= 0 or sample_count > frame_count * frame_size:
        raise ValueError(f"[ERROR] ISTFT: Sample count must be in (0, {frame_count * frame_size}]. Got {sample_count}.")
    
    # Transform frames to time domain if in frequency domain
    if transform:
        frames = Array.validate(array=frames, coerce="complex64", ref_axes=(0, 2))
        frames = fft(array=frames, size=fft_size, axis=0, inverse=True)
        frames = np.real(frames)
        
    # Trim off excess data from FFT sizing (remove zero padding)
    frames = Array.validate(array=frames, coerce="float", ref_axes=(0, 2))
    if frame_size < fft_size:
        frames = Array.subset(array=frames, method="index", limits=frame_size-1, axes=0, ref_axes=(0, 2))
    
    # Reshape frames data back to vector-like structure
    frames = frames.reshape((-1, channel_count), order="F")
    
    # Allocate space for signal and generate window function
    signal = np.zeros_like(frames)
    window = Windows.choose_window(name=window, length=frame_size, mode="run")
    
    # Apply inverse windowing to frames while populating signal data
    for f in range(frame_count):
        i, j = f * frame_size, f * shift_size
        d = frame_size
        signal[j:j + d] = frames[i:i + d] / window
        
    # Trim off excess data (zero padding) to get desired sample count
    signal = Array.subset(array=signal, limits=sample_count-1, axes=0, method="index")
    
    return signal


def calc_spectra(
    signal: np.ndarray, 
    size: Optional[int] = None, 
    filter: float = 1e-6, 
    stft: bool = False,
    clip: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the spectra of a signal in the frequency domain.

    Args:
        signal (np.ndarray): The input signal.
        size (int, optional): The FFT size. Defaults to None.
        filter (float, optional): The filter to set small values to 0. Defaults to 1e-6.
        stft (bool, optional): Indicator if signal is a short-time fourier transform with (M, N, C) shape. Defaults to False.
        clip (bool, optional): To clip the spectra signal in half (doesn't apply to spectra output). Default to False.

    Returns:
        (np.ndarray, np.ndarray, np.ndarray): The return spectra, magnitude, and phase of the signal.
        
    Raises:
        TypeError: if size is not integer, filter is not float, or stft is not boolean.
    """
    
    if not isinstance(size, (int, type(None))):
        raise TypeError(f"[ERROR] Calc Spectra: Size must be an integer. Got {type(size).__name__}.")
    if not isinstance(filter, float):
        raise TypeError(f"[ERROR] Calc Spectra: Filter must be a float. Got {type(filter).__name__}.")
    if not isinstance(stft, bool):
        raise TypeError(f"[ERROR] Calc Spectra: STFT must be a boolean. Got {type(stft).__name__}.")
    if not isinstance(clip, bool):
        raise TypeError(f"[ERROR] Calc Spectra: Clip must be a boolean. Got {type(clip).__name__}.")
    
    if filter < 0:
        filter = 0
        print(f"[WARNING] Calc Spectra: Filter should be non-negative. Got {filter}. Resetting filter to 0.")
    
    if stft:
        signal = Array.validate(array=signal, ref_axes=(0, 2))
    else:
        signal = Array.validate(array=signal)
        
    # Default size to size of signal
    if size is None:
        size = signal.shape[0]
        
    # Lower size values sometimes cause issues
    # if size < 128:
    #     size = 128
    #     print(f"[WARNING] Calc Spectra: The FFT size should be greater than 128. Got {size}. Resetting size to 128.")
        
    # Compute the spectra of the signal
    spectra = fft(array=signal, size=size, axis=0)
    
    # Apply filtering to reduce near-zero noise
    if filter:
        spectra[np.abs(spectra) < filter] = 0
        
    # Magnitude and Phase spectra computed directly
    magnitude = np.abs(spectra)
    phase = np.angle(spectra)
    
    if clip:
        if stft:
            magnitude = Array.subset(array=magnitude, limits=[0.5], axes=[0], how="right", ref_axes=(0, 2))
            phase = Array.subset(array=phase, limits=[0.5], axes=[0], how="right", ref_axes=(0, 2))
        else:
            magnitude = Array.subset(array=magnitude, limits=[0.5], axes=[0])
            phase = Array.subset(array=phase, limits=[0.5], axes=[0])
    
    return spectra, magnitude, phase


def calc_spectrogram(
    signal: np.ndarray,
    sample_rate: Optional[float] = None,
    window_size: float = 0.01, 
    fft_size: int = 512, 
    overlap: float = 0.5, 
    window: str = "hamming",
    clip: bool = True
) -> np.ndarray:
    """
    Calculate the spectrogram of the signal through STFT.

    Args:
        signal (np.ndarray): The input signal.
        sample_rate (float, optional): The sample rate of the signal. Defaults to 0.
        window_size (float, optional): The proportional size of the window. Defaults to 0.01.
        fft_size (int, optional): The size of the FFT transform. Defaults to 512.
        overlap (float, optional): The proportional window overlap size. Defaults to 0.5.
        window_function (str, optional): The name of the window function. Defaults to "hamming".

    Returns:
        np.ndarray: The spectrogram of the signal.
    """
    
    # Calculate the spectral data
    spectrogram = stft(
        signal=signal, 
        sample_rate=sample_rate, 
        fft_size=fft_size, 
        window_size=window_size, 
        overlap=overlap, 
        window=window,
        transform=True,
        clip=clip
    )
    
    # Get the magnitude output
    magnitude = spectrogram[1]
    
    # return spectrogram data
    return magnitude