from typing import Union, Optional, List, Tuple
import numpy as np
import scipy
from . import array as Array
from . import spectral as Spectral
from . import transforms as Transforms
from . import frequency as Frequency
from . import windows as Windows


def pole_mapping(
    signal: np.ndarray, 
    time_series: np.ndarray, 
    angular_freq: float, 
    damping_coef: float, 
    gain: float
) -> np.ndarray:
    """
    Apply a pole-mapping linear time-invariant (LTI) transformation to a signal.
    
    Math:
        polemapping: y(t) = x(t) + k*x(t-1) + 2k*y(t-1) - k*k*y(t-2)
    
        The input signal is modulated by an exponential term to shift its frequency:
            s(t) = exp(-1j * w_f * t) * s(t)
        
        The signal is then processed using a linear filter defined by the coefficients:
            b = [1, d]
            a = [1, -2 * d, d^2]
        
        After filtering, the signal is demodulated and gain is applied:
            s(t) = sqrt(g) * real(exp(1j * w_f * t) * s(t))
            
        Where:
            w_f is the angular frequency.
            d is the damping coefficient.
            g is the gain.

    Args:
        signal (np.ndarray): The input signal.
        time_series (np.ndarray): The time series of the signal.
        angular_freq (float): The angular frequency, determining the rate of oscillation in radians per second.
        damping_coef (float): The damping coefficient, indicating how quickly oscillations decay.
        gain (float): The gain, specifying the amplification or attenuation factor for the signal.

    Returns:
        np.ndarray: The transformed signal after applying the pole-mapping LTI process.
        
    Raises:
        TypeError: If angular frequency, damping coefficient, or gain is not float.
        ValueError: If damping coefficient or gain is negative.
    """
    
    if not isinstance(angular_freq, float):
        raise TypeError(f"[ERROR] Pole-mapping: Angular frequency must be a float. Got {type(angular_freq).__name__}.")
    if not isinstance(damping_coef, float):
        raise TypeError(f"[ERROR] Pole-mapping: Damping coefficient must be a float. Got {type(damping_coef).__name__}.")
    if not isinstance(gain, float):
        raise TypeError(f"[ERROR] Pole-mapping: Gain must be a float. Got {type(gain).__name__}.")
    
    if damping_coef < 0:
        raise ValueError(f"[ERROR] Pole-mapping: Damping coefficient must be non-negative. Got {damping_coef}.")
    if gain < 0:
        raise ValueError(f"[ERROR] Pole-mapping: Gain must be non-negative. Got {gain}.")
    
    signal = Array.validate(array=signal)
    time_series = Array.validate(array=time_series)

    for c in range(signal.shape[1]):
        
        channel = signal[:, c].reshape((-1, 1))
        
        # Apply exponential modulation to the signal based on the angular frequency
        exp_channel = np.exp(-1j * angular_freq * time_series) * channel

        # Compute the coefficients and linear filter for the signal
        num_coefs = [1, damping_coef]
        den_coefs = [1, -2 * damping_coef, damping_coef * damping_coef]
        exp_channel = linear_filter(num_coefs=num_coefs, den_coefs=den_coefs, signal=exp_channel, coerce="complex")
        
        # Apply inverse exponential modulation and gain adjustment to the signal
        inv_channel = np.sqrt(gain) * np.real(np.exp(1j * angular_freq * time_series) * exp_channel)
        
        signal[:, c] = inv_channel.reshape((-1, ))
    
    return signal


def gamma_tone_filter(
    signal: np.ndarray, 
    cutoff_freq: float, 
    sample_rate: Optional[float] = None
) -> np.ndarray:
    """
    Apply a series of pole-mapping LTI transformations to an input signal, modeling the auditory filter response 
    parameters derived from the Equivalent Rectangular Bandwidth (ERB) scale.
    
    Math:
        The corrected bandwidth B is calculated as:
            B = 1.019 * ERB(f)
        
        The time per sample Ts is given by:
            Ts = 2 * pi / fs
        
        The angular frequency omega is computed as:
            w_f = 2 * pi * f
        
        The exponential decay coefficient alpha is:
            a = exp(-B * Ts)
        
        The gain G for the pole-mapping transformation is:
            G = (B * Ts)^4 / 3
        
        The pole-mapping transformation is applied to the signal:
            x_tilde(t) = exp(-i * w_f * t) * x(t)
            y(t) = Filter(x_tilde(t), [1, a], [1, -2 * a, a^2])
            x_prime(t) = sqrt(G) * Re(exp(i * w_f * t) * y(t))
            
        Where:
            f is the cutoff frequency.
            fs is the sample rate.

    Args:
        signal (np.ndarray): The input signal.
        cutoff_freq (float): The cutoff frequency for the filter.
        sample_rate (float, optional): The sampling rate of the signal. Defaults to None.

    Returns:
        np.ndarray: The filtered signal after applying the gamma tone filter.
    """
    
    # Rotate the signal array up
    signal = Array.validate(array=signal, direction="up")
    
    # Validate the sample rate and construct the time series
    sample_rate = Frequency.val_sample_rate(sample_rate=sample_rate)
    time_series = np.arange(signal.shape[1]) / sample_rate
    
    # Apply a correction factor to the ERB to more accurately model human auditory filters
    bandwidth_correction = 1.019
    bandwidth = Frequency.calc_erb(cutoff_freq) * bandwidth_correction
    
    # Calculate the timpe per sample based on the sample rate
    time_per_sample = 2 * np.pi / sample_rate
    
    # Calculate the angular frequency from the cutoff frequency
    angular_freq = 2 * np.pi * cutoff_freq
    
    # Calculate the exponential decay coefficient based on bandwidth and time per sample
    exp_decay_coef = np.exp(-bandwidth * time_per_sample)
    
    # Compute the gain for the pole-mapping transformation
    gain = ((bandwidth * time_per_sample) ** 4) / 3
    
    # Apply the pole-mapping transformation twice, flipping the signal each time
    for i in range(2):
        signal = pole_mapping(signal=signal, time_series=time_series, angular_freq=angular_freq, damping_coef=exp_decay_coef, gain=gain)
        signal = np.flip(signal)
    
    return signal


def convolve(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Perform convolution of two 2D arrays along the first axis.
    
    Formula:
        Let `signal` be a 2D array with dimensions (N, C) where N is the number of samples and C is the number of channels.
        Let `kernel` be a 2D array with dimensions (M, C) where M is the number of samples and C is the number of channels.
        Let `size` be the next power of 2 greater than or equal to (N + M - 1).

        Define the output `output` as a 2D array with dimensions ((N + M - 1), C).

        The convolution operation for each channel c is performed as follows:

        1. Pad `signal[:, c]` and `kernel[:, c]` to length `size`:
            `signal_padded[:, c] = pad(signal[:, c], (0, size - N))`
            `kernel_padded[:, c] = pad(kernel[:, c], (0, size - M))`

        2. Compute the Fast Fourier Transform (FFT):
            `FFT_signal[:, c] = FFT(signal_padded[:, c])`
            `FFT_kernel[:, c] = FFT(kernel_padded[:, c])`

        3. Multiply the FFT results element-wise:
            `FFT_output[:, c] = FFT_signal[:, c] * FFT_kernel[:, c]`

        4. Compute the inverse FFT to get the convolved result:
            `output[:, c] = inverse FFT(FFT_output[:, c])[:(N + M - 1)]`

        The final convolved signal is `output`, a 2D array with dimensions ((N + M - 1), C).

    Arguments:
        signal (np.ndarray): The first input array, expected to be a 2D array.
        kernel (np.ndarray): The second input array, expected to be a 2D array.

    Returns:
        np.ndarray: The result of the convolution operation, a 2D array.

    Raises:
        TypeError: If either signal or kernel is not a numpy array.
        ValueError: If either signal or kernel is not 2D, or matching channels.
    """
    
    signal = Array.validate(signal)
    kernel = Array.validate(kernel)
    
    # Check if signal and kernel have same number of channels, or if kernel only has 1 channel then proceed
    if signal.shape[1] != kernel.shape[1] and kernel.shape[1] != 1:
        raise ValueError(f"[ERROR] Convolve: Signal and Kernel must have the same number of channels. Got {signal.shape[1]} != {kernel.shape[1]}.")
    
    # Convolve is commutable so order doesn't matter except signal should be the larger array as long as kernel has either
    # the same number of channels as the signal or only one channel.
    if signal.shape[1] == 1 and kernel.shape[1] > 1:
        signal, kernel = kernel, signal

    # Set sizing parameters
    signal_channels = signal.shape[1]
    signal_samples, kernel_samples = signal.shape[0], kernel.shape[0]
    output_samples = signal_samples + kernel_samples - 1
    
    # Pad both array and h to the same length, which is the next power of 2 for efficiency
    size = 2 ** np.ceil(np.log2(output_samples)).astype(int)
    output = np.zeros((output_samples, signal_channels))
    
    # Iterate over channels to perform convolution over each channel array
    for channel in range(signal_channels):
        # Add padding to each array
        signal_padded = np.pad(signal[:, channel], (0, size - signal_samples)).reshape((-1, 1))
        index = channel if kernel.shape[1] != 1 else 0
        kernel_padded = np.pad(kernel[:, index], (0, size - kernel_samples)).reshape((-1, 1))
        
        # Perform fft using numpy's functions
        fft_signal = Spectral.fft(signal_padded, mode="stable")
        fft_kernel = Spectral.fft(kernel_padded, mode="stable")
        
        # Convolution and return to time domain
        fft_output = fft_signal * fft_kernel
        output_channel = Spectral.fft(fft_output, inverse=True, mode="stable")
        
        # Insert transformed channel data
        output[:, channel] = np.real(output_channel[:output_samples]).reshape((-1, ))
        
    return output


def linear_filter(
    num_coefs: np.ndarray, 
    den_coefs: np.ndarray, 
    signal: np.ndarray,
    coerce: str = "float"
) -> np.ndarray:
    """
    Applies a linear filter to the input signal using the provided numerator and denominator coefficients.
    
    Formula:
        y[n] = (1 / a0) * (b0 * x[n] + b1 * x[n-1] + b2 * x[n-2] + ... + bN * x[n-N]
                - (a1 * y[n-1] + a2 * y[n-2] + ... + aM * y[n-M]))
                
        where:
            y[n] is the output signal at sample n.
            x[n] is the input signal at sample n.
            bi are the numerator coefficients.
            aj are the denominator coefficients.
            N is the number of numerator coefficients minus one.
            M is the number of denominator coefficients minus one.

    Arguments:
        num_coefs (np.ndarray): Numerator coefficients for the filter. Must be 1-dimensional.
        den_coefs (np.ndarray): Denominator coefficients for the filter. Must be 1-dimensional.
        signal (np.ndarray): Input array to be filtered. Must be at least 1-dimensional.
        coerce (str, optional): The data type to coerce to. Defaults to float.

    Returns:
        np.ndarray: The filtered signal.

    Raises:
        TypeError: If input types are not correct.
        ValueError: If input arrays do not meet dimension requirements or are empty.
    """
    
    # Validate and convert inputs to numpy arrays
    num_coefs = Array.validate(num_coefs)
    den_coefs = Array.validate(den_coefs)
    
    signal = Array.validate(signal, coerce=coerce)
    
    # Check dimensions and size of numerator coefficients
    if num_coefs.ndim > 2 or num_coefs.shape[1] > 1:
        raise ValueError(f"[ERROR] Linear filter: Numerator coefficients array must be 1-dimensional. Got {num_coefs.ndim} dimensions and {num_coefs.shape[1]} channels.")
    if num_coefs.size == 0:
        raise ValueError(f"[ERROR] Linear filter: Numerator coefficients array must have at least one value given. Got {num_coefs.size}")
    
    # Check dimensions and size of denominator coefficients
    if den_coefs.ndim > 2 or den_coefs.shape[1] > 1:
        raise ValueError(f"[ERROR] Linear filter: Denominator coefficients array must be 1-dimensional. Got {den_coefs.ndim} dimensions and {den_coefs.shape[1]} channels.")
    if den_coefs.size == 0:
        raise ValueError(f"[ERROR] Linear filter: Denominator coefficients array must have at least one value given. Got {den_coefs.size}.")
    
    # Initialize the output signal and the delay buffer for filtering
    filtered_signal = np.zeros(signal.shape, dtype=coerce)
    
    # Iterate over each channel and apply the filter
    for c in range(signal.shape[1]):
        for n in range(signal.shape[0]):
            # Apply feedforward coefficients
            for i in range(0, num_coefs.shape[0]):
                if n - i >= 0:
                    filtered_signal[n, c] += num_coefs[i] * signal[n - i, c]
                   
            # Apply feedback coefficients 
            for j in range(1, den_coefs.shape[0]):
                if n - j >= 0:
                    filtered_signal[n, c] -= den_coefs[j] * filtered_signal[n - j, c]
                    
            filtered_signal[n, c] /= den_coefs[0]
    
    return filtered_signal


def fir_coefs(
    length: Union[int, float], 
    cutoff: Union[int, float, np.ndarray], 
    pass_zero: bool = True, 
    sample_rate: Optional[float] = None, 
    scale: bool = True
) -> np.ndarray:
    """
    Calculate the FIR filter coefficients.
    
    Formula:
        1. Create response signal
            response(n) = sum_{bands} ( 
                right_i * sinc( 
                    right_i * ( n - (length - 1) / 2 ) 
                ) - 
                left_i * sinc( 
                    left_i * ( n - (length - 1) / 2 ) 
                ) 
            )

        2. Apply hamming window
            response(n) *= 0.54 - 0.46 * cos( 2 * pi * n / (length - 1) )

        3. Apply scaling
            response(n) *= 1 / sum ( response(n) * cos( pi * n * scale_frequency ) )

    Arguments:
        length (int): Length of the filter, i.e., order + 1
        cutoff (float or np.ndarray): Frequencies must be between 0 and Nyquist frequency (exclusive)
        pass_zero (bool, optional): Whether to pass the zero frequency
        sample_rate (float, optional): Sample rate, must be greater than zero
        scale (bool, optional): Whether to scale the coefficients

    Returns:
        np.ndarray: The calculated filter coefficients

    Raises:
        TypeError: If any argument is of incorrect type
        ValueError: If any argument is of invalid value
    """

    if not isinstance(pass_zero, bool):
        raise TypeError(f"[ERROR] FIR Coefficients: Pass zero must be a boolean. Got {type(pass_zero).__name__}.")
    if not isinstance(length, (int, float)) or length <= 0:
        raise ValueError(f"[ERROR] FIR Coefficients: Length must be a positive number. Got {type(length).__name__} {length}.")
    
    length = int(length)
    
    if length <= 0:
        raise ValueError(f"[ERROR] FIR Coefficients: Length must be positive. Got {length}.")

    # Set the sample rate and nyquist limit
    sample_rate = Frequency.val_sample_rate(sample_rate=sample_rate)
    nyquist_limit = sample_rate / 2

    # Check and validate the cutoff frequencies
    if not isinstance(cutoff, (int, float, np.ndarray, list)):
        raise TypeError(f"[ERROR] FIR Coefficients: Cutoff must be a float, ndarray, or list. Got {type(cutoff).__name__}.")
    
    cutoff = Array.validate(array=cutoff).reshape((-1, )) / nyquist_limit
    if cutoff.ndim > 2:
        raise ValueError(f"[ERROR] FIR Coefficients: Cutoff must be 1-dimensional. Got {cutoff.ndim}.")
    if cutoff.size == 0:
        raise ValueError(f"[ERROR] FIR Coefficients: Cutoff must have at least one frequency. Got {cutoff.size}.")
    if cutoff.size > 2:
        raise ValueError(f"[ERROR] FIR Coefficients: Cutoff must have at most two frequencies. Got {cutoff.size}.")
    
    # Check if cutoff is monotonically increasing between 0 and nyquist limit
    if not np.all(np.diff(cutoff) > 0):
        raise ValueError(f"[ERROR] FIR Coefficients: Cutoff frequencies must be monotonically increasing. Got {cutoff}.")
    if not np.all((0 < cutoff) & (cutoff < 1)):
        raise ValueError(f"[ERROR] FIR Coefficients: Cutoff frequencies must be between 0 and the Nyquist frequency (exclusive). Got {cutoff}.")
    
    # Set pass nyquist limit
    pass_nyquist = (cutoff.size % 2 != 0) != pass_zero
    if pass_nyquist and length % 2 == 0:
        raise ValueError("[ERROR] FIR Coefficients: Filter with even number of coefficients must have no response at Nyquist limit.")
    
    # Insert 0 or 1 at ends of cutoff to ensure length is even. Each pair are edges of a passband
    cutoff = np.hstack([[0.0] * pass_zero, cutoff, [1.0] * pass_nyquist])
    bands = cutoff.reshape((-1, 2))
    
    # Generate the centered index series
    midpoint = (length - 1) / 2
    centered_indices = Array.linseries(start=0, end=length, endpoint=False, coerce="float") - midpoint
    response = 0
    
    # Update the response with normalized sinc transforms
    for left, right in bands:
        response += right * Transforms.sinc(signal=right * centered_indices, point=True)
        response -= left * Transforms.sinc(signal=left * centered_indices, point=True)
    
    # Apply hamming window to response signal
    window = Windows.choose_window(length=length, name="hamming", mode="run")
    response *= window
    
    # Scale the response signal
    if scale:
        left, right = bands[0, :]
        if left == 0:
            scale_frequency = 0.0
        elif right == 1:
            scale_frequency = 1.0
        else:
            scale_frequency = (left + right) / 2
        
        # Apply positional encoding array to response signal to compute scalar for transform
        positional_encoding = np.cos(np.pi * centered_indices * scale_frequency)
        scalar = np.sum(response * positional_encoding)
        response = Transforms.scalar_transform(signal=response, scalar=1/scalar, max_scale=False)
    
    return response


def fir_filter(
    signal: np.ndarray, 
    cutoffs: Union[int, float, List, Tuple, np.ndarray],
    sample_rate: Optional[Union[int, float]] = None, 
    order: int = 200,
    band_type: str = "lowpass",
    method: str = "filtfilt",
    version: str = "stable"
) -> np.ndarray:
    """
    Generate and apply the FIR coefficients of Nth-order on a signal using one of the supported filtering methods.

    Args:
        signal (np.ndarray): The input signal.
        cutoffs (numeric): The cutoff frequencies in Hz.
        sample_rate (numeric, optional): Sampling rate of the signal. Defaults to None.
        order (int, optional): The order of the filter. Defaults to 200.
        band_type (str, optional): The filter band type. Defaults to "lowpass".
        method (str, optional): The filter application method. Defaults to "filtfilt".
        version (str, optional): The algorithm version. Defaults to "stable".

    Raises:
        TypeError: If inputs have unexpected types.
        ValueError: If inputs have unexpected values.

    Returns:
        np.ndarray: The filter response on the signal.
    """
    
    # Validate signal and sample rate
    signal = Array.validate(array=signal)
    sample_rate = Frequency.val_sample_rate(sample_rate=sample_rate)
    
    # Check typing
    if not isinstance(band_type, str):
        raise TypeError(f"[ERROR] FIR Filter: Band type must be a string. Got {type(band_type).__name__}.")
    if not isinstance(version, str):
        raise TypeError(f"[ERROR] FIR Filter: Version must be a string. Got {type(version).__name__}.")
    if not isinstance(method, str):
        raise TypeError(f"[ERROR] FIR Filter: Method must be a string. Got {type(method).__name__}.")
    
    # Validate parameters
    band_type = band_type.lower()
    if band_type not in ["lowpass", "highpass", "bandpass", "bandstop"]:
        raise ValueError(f"[ERROR] FIR Filter: Band type must be one of [lowpass, highpass, bandpass, bandstop]. Got {band_type}.")
    version = version.lower()
    if version not in ["stable", "experimental"]:
        raise ValueError(f"[ERROR] FIR Filter: Version must be one of [stable, experimental]. Got {version}.")
    method = method.lower()
    if method not in ["lfilter", "convolve", "filtfilt"]:
        raise ValueError(f"[ERROR] FIR Filter: Method must be one of [lfilter, convolve, filtfilt]. Got {method}.")
    
    # Set pass zero based on band type
    if band_type in ["lowpass", "bandstop"]:
        pass_zero = True
    else:
        pass_zero = False
    
    # Use experimental functions (custom)
    if version == "experimental":
        # Generate numerator coefficients for filter
        b = fir_coefs(
            length=order + 1,
            cutoff=cutoffs,
            pass_zero=pass_zero,
            sample_rate=sample_rate
        )
        
        # Use one of the methods to apply the filter coefficients on the signal
        if method == "lfilter":
            filter_signal = linear_filter(
                num_coefs=b, 
                den_coefs=[1.0], 
                signal=signal
            )
        if method =="convolve":
            filter_signal = convolve(
                signal=signal, 
                kernel=b
            )
            pad_length = np.ceil((filter_signal.shape[0] - signal.shape[0]) / 2)
            filter_signal = Array.subset(
                array=filter_signal, 
                limits=(pad_length - 1, filter_signal.shape[0] - pad_length - 2),
                axes=0,
                how="inner",
                method="index",
            )
    
    # Use stable functions (NumPy and SciPy)
    if version == "stable":
        # Generate numerator coefficients for filter
        b = scipy.signal.firwin(
            numtaps=order+1,
            cutoff=cutoffs,
            pass_zero=pass_zero,
            fs=sample_rate
        )
        
        # Use one of the methods to apply the filter coefficients on the signal
        if method == "lfilter":
            filter_signal = scipy.signal.lfilter(
                b=b.flatten(),
                a=[1.0],
                x=signal,
                axis=0
            )
        if method == "convolve":
            filter_channels = []
            for c in range(signal.shape[1]):
                fchannel = np.convolve(
                    b,
                    signal[:, c],
                    mode="same"
                )
                filter_channels.append(fchannel)
            filter_signal = np.vstack(filter_channels).transpose()
    
    # Apply the filter coefficients on the signal with SciPy's filtfilt function
    if method == "filtfilt":
        filter_signal = scipy.signal.filtfilt(
            b=b.flatten(), 
            a=[1.0], 
            x=signal, 
            axis=0
        )
    
    # Final validation of filtered signal
    filter_signal = Array.validate(array=filter_signal)
    
    return filter_signal


def butter_filter(
    signal: np.ndarray, 
    cutoffs: Union[int, float, List, Tuple, np.ndarray],
    sample_rate: Optional[Union[int, float]] = None, 
    order: int = 4,
    band_type: str = "lowpass",
    method: str = "filtfilt"
) -> np.ndarray:
    """
    Generate and apply the Butterworth coefficients of Nth-order on a signal using one of the supported filtering methods.

    Args:
        signal (np.ndarray): The input signal.
        cutoffs (numeric): The cutoff frequencies in Hz.
        sample_rate (numeric, optional): Sampling rate of the signal. Defaults to None.
        order (int, optional): The order of the filter. Defaults to 4.
        band_type (str, optional): The filter band type. Defaults to "lowpass".
        method (str, optional): The filter application method. Defaults to "filtfilt".

    Raises:
        TypeError: If inputs have unexpected types.
        ValueError: If inputs have unexpected values.

    Returns:
        np.ndarray: The filter response on the signal.
    """
    
    # Validate signal and sample rate
    signal = Array.validate(array=signal)
    sample_rate = Frequency.val_sample_rate(sample_rate=sample_rate)
    
    # Check typing
    if not isinstance(cutoffs, (int, float, list, tuple, np.ndarray)):
        raise TypeError(f"[ERROR] Butter Filter: Cutoffs must be numeric or an array. Got {type(cutoffs).__name__}.")
    if not isinstance(band_type, str):
        raise TypeError(f"[ERROR] Butter Filter: Band type must be a string. Got {type(band_type).__name__}.")
    if not isinstance(method, str):
        raise TypeError(f"[ERROR] Butter Filter: Method must be a string. Got {type(method).__name__}.")
    
    # Validate parameters
    band_type = band_type.lower()
    if band_type not in ["lowpass", "highpass", "bandpass", "bandstop"]:
        raise ValueError(f"[ERROR] Butter Filter: Band type must be one of [lowpass, highpass, bandpass, bandstop]. Got {band_type}.")
    method = method.lower()
    if method not in ["lfilter", "filtfilt"]:
        raise ValueError(f"[ERROR] Butter Filter: Method must be one of [lfilter, filtfilt]. Got {method}.")
    
    # Validate and flatten cutoffs array
    cutoffs = Array.validate(array=cutoffs).flatten()
    
    # Generate numerator coefficients for filter
    b, a = scipy.signal.butter(
        N=order,
        Wn=cutoffs,
        btype=band_type,
        fs=sample_rate
    )
    
    # Use one of the methods to apply the filter coefficients on the signal
    if method == "lfilter":
        filter_signal = scipy.signal.lfilter(
            b=b,
            a=a,
            x=signal,
            axis=0
        )
    if method == "filtfilt":
        filter_signal = scipy.signal.filtfilt(
            b=b, 
            a=a, 
            x=signal, 
            axis=0
        )
    
    # Final validation of filtered signal
    filter_signal = Array.validate(array=filter_signal)
    
    return filter_signal