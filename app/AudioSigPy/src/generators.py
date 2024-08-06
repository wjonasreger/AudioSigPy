from typing import Optional, Union
import numpy as np
from . import utils as Utils
from . import array as Array
from . import frequency as Frequency


# Generator gateway function

def choose_generator(
    duration: float, 
    name: str = "sine", 
    args: dict = {},
    mode: str = "run"
) -> Union[np.ndarray, bool]:
    """
    Gateway function for pre-selected default generator functions. Can be used to check availability and to generate waveforms.

    Args:
        duration (float): The duration of the waveform signal.
        name (str, optional): The name of the generator function. Defaults to "sine".
        args (dict, optional): Input parameters for the generator function. Defaults to {}.
        mode (str, optional): Gateway mode to check/run generators. Defaults to "run".

    Returns:
        Union[np.ndarray, bool]: The gateway output for the generator signal.
    """
    
    args["duration"] = duration
    generators = {
        "sine": sine,
        "square": square,
        "sawtooth": sawtooth,
        "triangle": triangle,
        "pulse": pulse,
        "white_noise": white_noise,
        "chirp": chirp,
        "glottal": glottal
    }
    
    gen_output = Utils.gateway(name=name, args=args, functions=generators, mode=mode)
    
    return gen_output


def sine(
    frequency: Union[float, int, np.ndarray], 
    duration: float, 
    sample_rate: Optional[float] = None, 
    phase: float = 0, 
    amplitude: float = 1
) -> np.ndarray:
    """
    Generate a sinusoidal wave signal.
    
    Math:
        s(n) = a * sin(2 * pi * f * t + p)
        where:
            a is the amplitude
            f is the frequency of the signal in Hz
            t is the time parameter of the signal in seconds
            p is the phase offset of the signal in radians

    Args:
        frequency (float): The frequency of the signal in Hz.
        duration (float): The duration of the signal in seconds.
        sample_rate (float or int, optional): The sampling frequency in Hz. Defaults to None.
        phase (float or int, optional): The phase offset of the signal in degrees. Defaults to 0.
        amplitude (float or int, optional): The amplitude of the signal. Defaults to 1.

    Returns:
        np.ndarray: The sinusoidal waveform as a numeric array.
        
    Raises:
        TypeError: If inputs are of invalid types.
        ValueError: If inputs have invalid values.
    """
    
    sample_rate = Frequency.val_sample_rate(sample_rate=sample_rate)
    
    if isinstance(frequency, np.ndarray):
        frequency = Array.validate(array=frequency)
        duration = frequency.shape[0] / sample_rate
    
    if not isinstance(frequency, (int, float, np.ndarray)):
        raise TypeError(f"[ERROR] Sine: Frequency must be an integer, float, or np.ndarray. Got {type(frequency).__name__}.")
    if not isinstance(duration, (int, float)):
        raise TypeError(f"[ERROR] Sine: Duration must be an integer or float. Got {type(duration).__name__}.")
    if not isinstance(phase, (int, float)):
        raise TypeError(f"[ERROR] Sine: Phase must be an integer or float. Got {type(phase).__name__}.")
    if not isinstance(amplitude, (int, float)):
        raise TypeError(f"[ERROR] Sine: Amplitude must be an integer or float. Got {type(amplitude).__name__}.")
    
    if np.any(frequency <= 0):
        raise ValueError(f"[ERROR] Sine: Frequency must be positive.")
    if duration <= 0:
        raise ValueError(f"[ERROR] Sine: Duration must be positive. Got {duration}.")
    if amplitude <= 0:
        raise ValueError(f"[ERROR] Sine: Amplitude must be positive. Got {amplitude}.")
    
    # Compute parameters for the signal
    sample_count = int(duration * sample_rate)
    t = Array.linseries(start=0, end=sample_count, endpoint=False) / sample_rate
    p = 2 * np.pi * (phase / 360)
    w = 2 * np.pi * frequency
    
    # Generate the signal array
    signal = amplitude * np.sin(w * t + p)
    
    signal = Array.validate(array=signal)
    
    return signal


def square(
    frequency: Union[float, int, np.ndarray], 
    duration: float, 
    sample_rate: Optional[float] = None, 
    phase: float = 0, 
    amplitude: float = 1, 
    harmonics: int = 12
) -> np.ndarray:
    """
    Generate a square wave signal.
    
    Math:
        s(n) = 4 * a / pi * sum_{k=1}^K (sin(2 * pi * (2 * k - 1) * f * t + p) / (2 * k - 1))
        where:
            a is the amplitude
            f is the frequency of the signal in Hz
            t is the time parameter of the signal in seconds
            p is the phase offset of the signal in radians
            K is the number of harmonic terms

    Args:
        frequency (float): The frequency of the signal in Hz.
        duration (float): The duration of the signal in seconds.
        sample_rate (float or int, optional): The sampling frequency in Hz. Defaults to None.
        phase (float or int, optional): The phase offset of the signal in degrees. Defaults to 0.
        amplitude (float or int, optional): The amplitude of the signal. Defaults to 1.
        harmonics (int, optional): The number of harmonic terms in the signal. Defaults to 12.

    Returns:
        np.ndarray: The square waveform as a numeric array.
        
    Raises:
        TypeError: If inputs are of invalid types.
        ValueError: If inputs have invalid values.
    """
    
    sample_rate = Frequency.val_sample_rate(sample_rate=sample_rate)
    
    if isinstance(frequency, np.ndarray):
        frequency = Array.validate(array=frequency)
        duration = frequency.shape[0] / sample_rate
    
    if not isinstance(frequency, (int, float, np.ndarray)):
        raise TypeError(f"[ERROR] Square: Frequency must be an integer, float, or np.ndarray. Got {type(frequency).__name__}.")
    if not isinstance(duration, (int, float)):
        raise TypeError(f"[ERROR] Square: Duration must be an integer or float. Got {type(duration).__name__}.")
    if not isinstance(phase, (int, float)):
        raise TypeError(f"[ERROR] Square: Phase must be an integer or float. Got {type(phase).__name__}.")
    if not isinstance(amplitude, (int, float)):
        raise TypeError(f"[ERROR] Square: Amplitude must be an integer or float. Got {type(amplitude).__name__}.")
    if not isinstance(harmonics, int):
        raise TypeError(f"[ERROR] Square: Harmonics must be an integer. Got {type(harmonics).__name__}.")
    
    if np.any(frequency <= 0):
        raise ValueError(f"[ERROR] Square: Frequency must be positive.")
    if duration <= 0:
        raise ValueError(f"[ERROR] Square: Duration must be positive. Got {duration}.")
    if amplitude <= 0:
        raise ValueError(f"[ERROR] Square: Amplitude must be positive. Got {amplitude}.")
    if harmonics <= 0:
        raise ValueError(f"[ERROR] Square: Harmonics must be positive. Got {harmonics}.")
    
    # Compute parameters for the signal
    sample_count = int(duration * sample_rate)
    t = Array.linseries(start=0, end=sample_count, endpoint=False) / sample_rate
    p = 2 * np.pi * (phase / 360)
    w = 2 * np.pi * frequency
    c = 4 * amplitude / np.pi
    
    # Generate the signal array
    signal = 0
    for k in range(1, harmonics + 1):
        n = 2 * k - 1
        signal += np.sin(n * w * t + p) / n
    signal = c * signal
    
    signal = Array.validate(array=signal)
        
    return signal


def triangle(
    frequency: Union[float, int, np.ndarray], 
    duration: float, 
    sample_rate: Optional[float] = None, 
    phase: float = 0, 
    amplitude: float = 1, 
    harmonics: int = 12
) -> np.ndarray:
    """
    Generate a sawtooth wave signal.
    
    Math:
        s(n) = 8 / pi^2 * sum_{k=1}^K (-1^k * sin(2 * pi * n * f * t + p) / n^2)
        where:
            a is the amplitude
            f is the frequency of the signal in Hz
            t is the time parameter of the signal in seconds
            p is the phase offset of the signal in radians
            K is the number of harmonic terms
            n = 2 * k + 1

    Args:
        frequency (float): The frequency of the signal in Hz.
        duration (float): The duration of the signal in seconds.
        sample_rate (float or int, optional): The sampling frequency in Hz. Defaults to None.
        phase (float or int, optional): The phase offset of the signal in degrees. Defaults to 0.
        amplitude (float or int, optional): The amplitude of the signal. Defaults to 1.
        harmonics (int, optional): The number of harmonic terms in the signal. Defaults to 12.

    Returns:
        np.ndarray: The triangle waveform as a numeric array.
        
    Raises:
        TypeError: If inputs are of invalid types.
        ValueError: If inputs have invalid values.
    """
    
    sample_rate = Frequency.val_sample_rate(sample_rate=sample_rate)
    
    if isinstance(frequency, np.ndarray):
        frequency = Array.validate(array=frequency)
        duration = frequency.shape[0] / sample_rate
    
    if not isinstance(frequency, (int, float, np.ndarray)):
        raise TypeError(f"[ERROR] Triangle: Frequency must be an integer, float, or np.ndarray. Got {type(frequency).__name__}.")
    if not isinstance(duration, (int, float)):
        raise TypeError(f"[ERROR] Triangle: Duration must be an integer or float. Got {type(duration).__name__}.")
    if not isinstance(phase, (int, float)):
        raise TypeError(f"[ERROR] Triangle: Phase must be an integer or float. Got {type(phase).__name__}.")
    if not isinstance(amplitude, (int, float)):
        raise TypeError(f"[ERROR] Triangle: Amplitude must be an integer or float. Got {type(amplitude).__name__}.")
    if not isinstance(harmonics, int):
        raise TypeError(f"[ERROR] Triangle: Harmonics must be an integer. Got {type(harmonics).__name__}.")
    
    if np.any(frequency <= 0):
        raise ValueError(f"[ERROR] Triangle: Frequency must be positive.")
    if duration <= 0:
        raise ValueError(f"[ERROR] Triangle: Duration must be positive. Got {duration}.")
    if amplitude <= 0:
        raise ValueError(f"[ERROR] Triangle: Amplitude must be positive. Got {amplitude}.")
    if harmonics <= 0:
        raise ValueError(f"[ERROR] Triangle: Harmonics must be positive. Got {harmonics}.")
    
    # Compute parameters for the signal
    sample_count = int(duration * sample_rate)
    t = Array.linseries(start=0, end=sample_count, endpoint=False) / sample_rate
    p = 2 * np.pi * (phase / 360)
    w = 2 * np.pi * frequency
    c = 8 / np.pi ** 2
    
    # Generate the signal array
    signal = 0
    for k in range(harmonics):
        n = 2 * k + 1
        signal += (-1)**k * np.sin(n * w * t + p) / n ** 2
    signal = c * signal
        
    signal = Array.validate(array=signal)
        
    return signal


def sawtooth(
    frequency: Union[float, int, np.ndarray], 
    duration: float, 
    sample_rate: Optional[float] = None, 
    phase: float = 0, 
    amplitude: float = 1, 
    harmonics: int = 12
) -> np.ndarray:
    """
    Generate a sawtooth wave signal.
    
    Math:
        s(n) = 2 * a / pi * sum_{k=1}^K (-1^k * sin(2 * pi * k * f * t + p) / k)
        where:
            a is the amplitude
            f is the frequency of the signal in Hz
            t is the time parameter of the signal in seconds
            p is the phase offset of the signal in radians
            K is the number of harmonic terms

    Args:
        frequency (float): The frequency of the signal in Hz.
        duration (float): The duration of the signal in seconds.
        sample_rate (float or int, optional): The sampling frequency in Hz. Defaults to None.
        phase (float or int, optional): The phase offset of the signal in degrees. Defaults to 0.
        amplitude (float or int, optional): The amplitude of the signal. Defaults to 1.
        harmonics (int, optional): The number of harmonic terms in the signal. Defaults to 12.

    Returns:
        np.ndarray: The sawtooth waveform as a numeric array.
        
    Raises:
        TypeError: If inputs are of invalid types.
        ValueError: If inputs have invalid values.
    """
    
    sample_rate = Frequency.val_sample_rate(sample_rate=sample_rate)
    
    if isinstance(frequency, np.ndarray):
        frequency = Array.validate(array=frequency)
        duration = frequency.shape[0] / sample_rate
    
    if not isinstance(frequency, (int, float, np.ndarray)):
        raise TypeError(f"[ERROR] Sawtooth: Frequency must be an integer, float, or np.ndarray. Got {type(frequency).__name__}.")
    if not isinstance(duration, (int, float)):
        raise TypeError(f"[ERROR] Sawtooth: Duration must be an integer or float. Got {type(duration).__name__}.")
    if not isinstance(phase, (int, float)):
        raise TypeError(f"[ERROR] Sawtooth: Phase must be an integer or float. Got {type(phase).__name__}.")
    if not isinstance(amplitude, (int, float)):
        raise TypeError(f"[ERROR] Sawtooth: Amplitude must be an integer or float. Got {type(amplitude).__name__}.")
    if not isinstance(harmonics, int):
        raise TypeError(f"[ERROR] Sawtooth: Harmonics must be an integer. Got {type(harmonics).__name__}.")
    
    if np.any(frequency <= 0):
        raise ValueError(f"[ERROR] Sawtooth: Frequency must be positive.")
    if duration <= 0:
        raise ValueError(f"[ERROR] Sawtooth: Duration must be positive. Got {duration}.")
    if amplitude <= 0:
        raise ValueError(f"[ERROR] Sawtooth: Amplitude must be positive. Got {amplitude}.")
    if harmonics <= 0:
        raise ValueError(f"[ERROR] Sawtooth: Harmonics must be positive. Got {harmonics}.")
    
    # Compute parameters for the signal
    sample_count = int(duration * sample_rate)
    t = Array.linseries(start=0, end=sample_count, endpoint=False) / sample_rate
    p = 2 * np.pi * (phase / 360)
    w = 2 * np.pi * frequency
    c = 2 * amplitude / np.pi
    
    # Generate the signal array
    signal = 0
    for k in range(1, harmonics + 1):
        signal += (-1)**k * np.sin(k * w * t + p) / k
    signal = c * signal
        
    signal = Array.validate(array=signal)
        
    return signal


def pulse(
    frequency: Union[float, int, np.ndarray], 
    duration: float, 
    sample_rate: Optional[float] = None, 
    phase: float = 0, 
    amplitude: float = 1, 
    duty_cycle: float = 0.5,
    harmonics: int = 12,
    center: bool = False
) -> np.ndarray:
    """
    Generate a pulse wave signal with a customizable duty cycle.
    
    Math:
        s(n) = a * d + 2 * a / pi * sum_{k=1}^K sin(k * 2 * pi * f * t + p) / k) *
                                                sin(pi * k * d)
        where:
            a is the amplitude
            f is the frequency of the signal in Hz
            t is the time parameter of the signal in seconds
            p is the phase offset of the signal in radians
            d is the duty cycle of the pulse wave (0 < d <= 1)
            K is the number of harmonic terms

    Args:
        frequency (float): The frequency of the signal in Hz.
        duration (float): The duration of the signal in seconds.
        sample_rate (float or int, optional): The sampling frequency in Hz. Defaults to None.
        phase (float or int, optional): The phase offset of the signal in degrees. Defaults to 0.
        amplitude (float or int, optional): The amplitude of the signal. Defaults to 1.
        duty_cycle (float, optional): The duty cycle of the pulse wave (0 < duty_cycle <= 1). Defaults to 0.5.
        harmonics (int, optional): The number of harmonic terms in the signal. Defaults to 12.
        center (bool): Centers the signal at 0. Default to False.

    Returns:
        np.ndarray: The pulse waveform as a numeric array.
        
    Raises:
        TypeError: If inputs are of invalid types.
        ValueError: If inputs have invalid values.
    """
    
    sample_rate = Frequency.val_sample_rate(sample_rate=sample_rate)
    
    if isinstance(frequency, np.ndarray):
        frequency = Array.validate(array=frequency)
        duration = frequency.shape[0] / sample_rate
    
    if not isinstance(frequency, (int, float, np.ndarray)):
        raise TypeError(f"[ERROR] Pulse: Frequency must be an integer, float, np.ndarray. Got {type(frequency).__name__}.")
    if not isinstance(duration, (int, float)):
        raise TypeError(f"[ERROR] Pulse: Duration must be an integer or float. Got {type(duration).__name__}.")
    if not isinstance(phase, (int, float)):
        raise TypeError(f"[ERROR] Pulse: Phase must be an integer or float. Got {type(phase).__name__}.")
    if not isinstance(amplitude, (int, float)):
        raise TypeError(f"[ERROR] Pulse: Amplitude must be an integer or float. Got {type(amplitude).__name__}.")
    if not isinstance(duty_cycle, (int, float)):
        raise TypeError(f"[ERROR] Pulse: Duty cycle must be an integer or float. Got {type(duty_cycle).__name__}.")
    if not isinstance(harmonics, int):
        raise TypeError(f"[ERROR] Pulse: Harmonics must be an integer. Got {type(harmonics).__name__}.")
    if not isinstance(center, bool):
        raise TypeError(f"[ERROR] Pulse: Center must be a boolean. Got {type(center).__name__}.")
    
    if np.any(frequency <= 0):
        raise ValueError(f"[ERROR] Pulse: Frequency must be positive. Got {frequency}.")
    if duration <= 0:
        raise ValueError(f"[ERROR] Pulse: Duration must be positive. Got {duration}.")
    if amplitude <= 0:
        raise ValueError(f"[ERROR] Pulse: Amplitude must be positive. Got {amplitude}.")
    if not (0 < duty_cycle <= 1):
        raise ValueError(f"[ERROR] Pulse: Duty cycle must be between 0 and 1. Got {duty_cycle}.")
    if harmonics <= 0:
        raise ValueError(f"[ERROR] Pulse: Harmonics must be positive. Got {harmonics}.")
    
    # Compute parameters for the signal
    sample_count = int(duration * sample_rate)
    t = Array.linseries(start=0, end=sample_count, endpoint=False) / sample_rate
    p = 2 * np.pi * (phase / 360)
    w = 2 * np.pi * frequency
    d = duty_cycle
    c = 2 * amplitude / np.pi
    
    # Generate the signal array
    signal = 0
    for k in range(1, harmonics + 1):
        signal += np.sin(np.pi * k * d) * np.cos(k * w * t + p) / k
    signal = amplitude * d + c * signal
    
    if center:
        signal -= amplitude / 2
    
    signal = Array.validate(array=signal)
        
    return signal


def white_noise(
    duration: float, 
    sample_rate: Optional[float] = None, 
    amplitude: float = 1, 
    dist: str = "uniform",
    clip: bool = True
) -> np.ndarray:
    """
    Generate a white noise wave signal.
    
    Math:
    Uniform:
        s(n) = Uniform(-a, a)
        where:
            a is the amplitude
            
    Gaussian:
        s(n) = N(0, a / 3.5)
        where:
            a is the amplitude

    Args:
        duration (float): The duration of the signal in seconds.
        sample_rate (float or int, optional): The sampling frequency in Hz. Defaults to None.
        amplitude (float or int, optional): The amplitude of the signal. Defaults to 1.
        dist (str, optional): The probability distribution to sample. Defaults to uniform.
        clip (bool, optional): The option to clip the signal when gaussian is used. Defaults to True.

    Returns:
        np.ndarray: The white noise waveform as a numeric array.
        
    Raises:
        TypeError: If inputs are of invalid types.
        ValueError: If inputs have invalid values.
    """
    
    if not isinstance(duration, (int, float)):
        raise TypeError(f"[ERROR] White Noise: Duration must be an integer or float. Got {type(duration).__name__}.")
    if not isinstance(amplitude, (int, float)):
        raise TypeError(f"[ERROR] White Noise: Amplitude must be an integer or float. Got {type(amplitude).__name__}.")
    if not isinstance(dist, str):
        raise TypeError(f"[ERROR] White Noise: Distribution must be a string. Got {type(dist).__name__}.")
    if not isinstance(clip, bool):
        raise TypeError(f"[ERROR] White Noise: Clip must be a boolean. Got {type(clip).__name__}.")
    
    if duration <= 0:
        raise ValueError(f"[ERROR] White Noise: Duration must be positive. Got {duration}.")
    if amplitude <= 0:
        raise ValueError(f"[ERROR] White Noise: Amplitude must be positive. Got {amplitude}.")
    if dist not in ["gaussian", "uniform"]:
        raise ValueError(f"[ERROR] White Noise: Distribution must be one of [uniform, gaussian]. Got {dist}.")
    
    sample_rate = Frequency.val_sample_rate(sample_rate=sample_rate)
    
    # Compute parameters for the signal
    sample_count = int(duration * sample_rate)
    
    # Generate the signal array
    if dist == "gaussian":
        signal = np.random.normal(0, amplitude/3.5, sample_count)
        if clip:
            signal[signal > amplitude] = 1.0
            signal[signal < -amplitude] = -1.0
    elif dist == "uniform":
        signal = np.random.uniform(-amplitude, amplitude, sample_count)
        
    signal = Array.validate(array=signal)
    
    return signal


def chirp(
    freq_start: float, 
    freq_end: float, 
    duration: float, 
    sample_rate: Optional[float] = None, 
    phase: float = 0, 
    amplitude: float = 1
) -> np.ndarray:
    """
    Generate a linear chirp wave signal (sweep signal).
    
    Math:
        s(n) = a * sin(2 * pi * (f_s + (t * f_d / 2)) * t + p)
        where:
            a is the amplitude
            f_s is the starting frequency of the signal in Hz
            f_d is the frequency delta of the signal in Hz/s
                f_d = (f_e - f_s) / d
                f_e is the ending frequency of the signal in Hz
                d is the duration of the signal in seconds
            t is the time parameter of the signal in seconds
            p is the phase offset of the signal in radians

    Args:
        freq_start (float): The starting frequency of the signal in Hz.
        freq_end (float): The ending frequency of the signal in Hz.
        duration (float): The duration of the signal in seconds.
        sample_rate (float or int, optional): The sampling frequency in Hz. Defaults to None.
        phase (float or int, optional): The phase offset of the signal in degrees. Defaults to 0.
        amplitude (float or int, optional): The amplitude of the signal. Defaults to 1.

    Returns:
        np.ndarray: The chirp waveform as a numeric array.
        
    Raises:
        TypeError: If inputs are of invalid types.
        ValueError: If inputs have invalid values.
    """
    
    if not isinstance(freq_start, (int, float)):
        raise TypeError(f"[ERROR] Chirp: Starting frequency must be an integer or float. Got {type(freq_start).__name__}.")
    if not isinstance(freq_end, (int, float)):
        raise TypeError(f"[ERROR] Chirp: Ending frequency must be an integer or float. Got {type(freq_end).__name__}.")
    if not isinstance(duration, (int, float)):
        raise TypeError(f"[ERROR] Chirp: Duration must be an integer or float. Got {type(duration).__name__}.")
    if not isinstance(phase, (int, float)):
        raise TypeError(f"[ERROR] Chirp: Phase must be an integer or float. Got {type(phase).__name__}.")
    if not isinstance(amplitude, (int, float)):
        raise TypeError(f"[ERROR] Chirp: Amplitude must be an integer or float. Got {type(amplitude).__name__}.")
    
    if freq_start <= 0:
        raise ValueError(f"[ERROR] Chirp: Starting frequency must be positive. Got {freq_start}.")
    if freq_end <= 0:
        raise ValueError(f"[ERROR] Chirp: Ending frequency must be positive. Got {freq_end}.")
    if duration <= 0:
        raise ValueError(f"[ERROR] Chirp: Duration must be positive. Got {duration}.")
    if amplitude <= 0:
        raise ValueError(f"[ERROR] Chirp: Amplitude must be positive. Got {amplitude}.")
    
    sample_rate = Frequency.val_sample_rate(sample_rate=sample_rate)
    
    # Compute parameters for the signal
    sample_count = int(duration * sample_rate)
    t = Array.linseries(start=0, end=sample_count, endpoint=False) / sample_rate
    p = 2 * np.pi * (phase / 360)
    d = (freq_end - freq_start) / duration
    w = 2 * np.pi * ((t * d / 2) + freq_start)
    
    # Generate the signal array
    signal = amplitude * np.sin(w * t + p)
    
    signal = Array.validate(array=signal)
    
    return signal


def glottal(
    base_freq: float, 
    head_freq: float, 
    duration: float, 
    sample_rate: Optional[float] = None, 
    phase: float = 0, 
    amplitude: float = 1, 
    alpha: int = 100
) -> np.ndarray:
    """
    Generate a glottal wave signal.
    
    Math:
        s(n) = {
            a * 0.1 * (1 + cos(2 * pi * f_b * t + p)),          n % A != 0
            0.8 + a * 0.1 * (1 + cos(2 * pi * f_h * t + p)),    n % A == 0
        }
        where:
            a is the amplitude
            f_b is the frequency of the base signal in Hz
            f_h is the frequency of the header signal in Hz
            t is the time parameter of the signal in seconds
            p is the phase offset of the signal in radians
            A (alpha) adjusts the balance of resolution between the two signals

    Args:
        base_freq (float): The frequency of the base signal in Hz.
        head_freq (float): The frequency of the header signal in Hz.
        duration (float): The duration of the signal in seconds.
        sample_rate (float or int, optional): The sampling frequency in Hz. Defaults to None.
        phase (float or int, optional): The phase offset of the signal in degrees. Defaults to 0.
        amplitude (float or int, optional): The amplitude of the signal. Defaults to 1.
        alpha (int, optional): The resolution or "sample rate" step of the header signal. Defaults to 100.

    Returns:
        np.ndarray: The glottal waveform as a numeric array.
        
    Raises:
        TypeError: If inputs are of invalid types.
        ValueError: If inputs have invalid values.
    """
    
    if not isinstance(base_freq, (int, float)):
        raise TypeError(f"[ERROR] Glottal: Base frequency must be an integer or float. Got {type(base_freq).__name__}.")
    if not isinstance(head_freq, (int, float)):
        raise TypeError(f"[ERROR] Glottal: Head frequency must be an integer or float. Got {type(head_freq).__name__}.")
    if not isinstance(duration, (int, float)):
        raise TypeError(f"[ERROR] Glottal: Duration must be an integer or float. Got {type(duration).__name__}.")
    if not isinstance(phase, (int, float)):
        raise TypeError(f"[ERROR] Glottal: Phase must be an integer or float. Got {type(phase).__name__}.")
    if not isinstance(amplitude, (int, float)):
        raise TypeError(f"[ERROR] Glottal: Amplitude must be an integer or float. Got {type(amplitude).__name__}.")
    if not isinstance(alpha, int):
        raise TypeError(f"[ERROR] Glottal: Alpha must be an integer. Got {type(alpha).__name__}.")
    
    if base_freq <= 0:
        raise ValueError(f"[ERROR] Glottal: Base frequency must be positive. Got {base_freq}.")
    if head_freq <= 0:
        raise ValueError(f"[ERROR] Glottal: Head frequency must be positive. Got {head_freq}.")
    if duration <= 0:
        raise ValueError(f"[ERROR] Glottal: Duration must be positive. Got {duration}.")
    if amplitude <= 0:
        raise ValueError(f"[ERROR] Glottal: Amplitude must be positive. Got {amplitude}.")
    if alpha <= 0:
        raise ValueError(f"[ERROR] Glottal: Alpha must be positive. Got {alpha}.")
    
    sample_rate = Frequency.val_sample_rate(sample_rate=sample_rate)
    
    # Compute parameters for the signal
    sample_count = int(duration * sample_rate)
    t = Array.linseries(start=0, end=sample_count, endpoint=False) / sample_rate
    p = 2 * np.pi * (phase / 360)
    bw = 2 * np.pi * base_freq
    hw = 2 * np.pi * head_freq
    
    # Generate the two signal arrays
    base_signal = amplitude * 0.1 * (1 + np.cos(bw * t + p))
    head_signal = amplitude * 0.1 * (1 + np.cos(hw * t + p))
    
    # Shift step points to "head" of signal
    size = sample_count // alpha
    head_sample_idx = np.arange(sample_count) % size == 0
    head_signal[head_sample_idx] += 0.8
    base_signal[head_sample_idx] = head_signal[head_sample_idx]
    
    base_signal = Array.validate(array=base_signal)
    
    return base_signal