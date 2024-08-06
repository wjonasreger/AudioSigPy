from typing import Optional, Union, List, Tuple
import numpy as np
from . import array as Array
from . import transforms as Transforms


def val_sample_rate(sample_rate: Optional[Union[int, float]] = None) -> Union[int, float]:
    """
    Validate the sample rate. Return a default if not provided.

    Arguments:
        sample_rate (int or float, optional): The sample rate to validate.

    Returns:
        Union[int, float]: The validated sample rate.
    
    Raises:
        ValueError: If sample_rate is not a positive number.
    """

    # Standard sample rate of 44.1 kHz is default
    if sample_rate is None:
        sample_rate = 44100
    
    if sample_rate is not None:
        if not isinstance(sample_rate, (int, float)) or sample_rate <= 0:
            raise ValueError(f"[ERROR] Validate sample rate: Sampling rate must be a number greater than zero. Got {type(sample_rate).__name__} {sample_rate}.")
        
    return sample_rate


def calc_lowcut(frequency: float, factor: float = np.sqrt(2)) -> float:
    """
    Calculate the lower bound frequency given a factor.

    Args:
        frequency (float): The frequency in Hz.
        factor (float, optional): The multiplicative factor. Defaults to np.sqrt(2).
        
    Raises:
        TypeError: If frequency or factor is not a number.
        ValueError: If frequency or factor is not positive.
        
    Returns:
        float: The lowcut frequency.
    """
    
    if not isinstance(factor, (float, int)):
        raise TypeError(f"[ERROR] Calculate Lowcut: Factor must be a number. Got {type(factor).__name__}.")
    if factor <= 0:
        raise ValueError(f"[ERROR] Calculate Lowcut: Factor must be a positive number. Got {factor}.")
    
    if isinstance(frequency, (list, np.ndarray)):
        frequency = Array.validate(array=frequency)
    
    if not isinstance(frequency, (float, int, np.ndarray)):
        raise TypeError(f"[ERROR] Calculate Lowcut: Frequency must be a number. Got {type(frequency).__name__}.")
    if np.any(frequency <= 0):
        raise ValueError(f"[ERROR] Calculate Lowcut: Frequency must be a positive number. Got {frequency}.")
    
    lowcut = frequency / factor
    return lowcut


def calc_highcut(frequency: float, factor: float = np.sqrt(2)) -> float:
    """
    Calculate the lower bound frequency given a factor.

    Args:
        frequency (float): The frequency in Hz.
        factor (float, optional): The multiplicative factor. Defaults to np.sqrt(2).
        
    Raises:
        TypeError: If frequency or factor is not a number.
        ValueError: If frequency or factor is not positive.
        
    Returns:
        float: The highcut frequency.
    """
    
    if not isinstance(factor, (float, int)):
        raise TypeError(f"[ERROR] Calculate Highcut: Factor must be a number. Got {type(factor).__name__}.")
    if factor <= 0:
        raise ValueError(f"[ERROR] Calculate Highcut: Factor must be a positive number. Got {factor}.")
    
    if isinstance(frequency, (list, np.ndarray)):
        frequency = Array.validate(array=frequency)
    
    if not isinstance(frequency, (float, int, np.ndarray)):
        raise TypeError(f"[ERROR] Calculate Highcut: Frequency must be a number. Got {type(frequency).__name__}.")
    if np.any(frequency <= 0):
        raise ValueError(f"[ERROR] Calculate Highcut: Frequency must be a positive number. Got {frequency}.")
    
    highcut = frequency * factor
    return highcut


def calc_step_bands(
    lowcut: int = 20, 
    highcut: int = 20000, 
    steps: int = 1, 
    base_freq: float = 125.0,
    centre_freqs: Optional[Union[int, float, List, np.ndarray]] = None,
    scale: str = None,
    dedup: bool = True
) -> np.ndarray:
    """
    This function generates frequency bands using a method called "step scaling," which 
    generalizes the concept of octaves and notes from music theory to any desired interval 
    structure.
    
    Theory:
        In music theory, notes and octaves are equidistant on a logarithmic freqeuncy scale. 
        The interval between two adjacent notes in an octave is defined by a constant 
        multiplicative factor, which is 2^(1/N), where N represents the number of notes within 
        an octave. For instance, in a standard Western music scale, N is 12, corresponding to 
        the twelve semitones in an octave.
        
        To extend this concept, "step scaling" can divide the frequency range into any number 
        of steps, not necessarily limited to musical notes. The frequency bands are computed 
        based on "half steps," each having a multiplicative factor of 2^(1/(2N)). This allows 
        for a finer resolution of frequency bands, suitable for various applications in signal 
        processing and audio analysis. 
        
        By using this generalized approach, the function can create custom frequency bands that
        maintain a consistent ratio between adjacent bands, similar to the way musical notes 
        are spaced, but adaptable to different contexts and requirements beyond traditional 
        music theory. This helps preserve the perceptual relevance of the intervals, while 
        being able to apply it broadly in musical contexts and signal processing applications.

    Args:
        lowcut (int, optional): Lower frequency bound. Defaults to 20.
        highcut (int, optional): Upper frequency bound. Defaults to 20000.
        steps (int, optional): Number of steps splitting an octave. Defaults to 1.
        base_freq (float, optional): Reference frequency for centre frequency generation. Defaults to 125.0.
        centre_freqs (np.ndarray, optional): Target centre frequencies to be fitted to step scale. Defaults to None.
        scale (str, optional): Octave or note scaling (i.e., steps of 1 or 12). Defaults to None.
        dedup (bool, optional): Option to deduplicate bands. Defaults to True.

    Raises:
        TypeError: If any inputs have unexpected types.
        ValueError: If any inputs have unexpected values.

    Returns:
        np.ndarray: Numpy array with 3 columns for lowcuts, centres, and highcuts of frequency bands.
    """
    
    # Type checking
    if not isinstance(lowcut, (float, int)):
        raise TypeError(f"[ERROR] Calculate Step Bands: Lowcut frequency must be a number. Got {type(lowcut).__name__}.")
    if not isinstance(highcut, (float, int)):
        raise TypeError(f"[ERROR] Calculate Step Bands: Highcut must be a number. Got {type(highcut).__name__}.")
    if not isinstance(steps, (int, float)):
        raise TypeError(f"[ERROR] Calculate Step Bands: Steps must be a number. Got {type(steps).__name__}.")
    if not isinstance(base_freq, (float, int)):
        raise TypeError(f"[ERROR] Calculate Step Bands: Base frequency must be a number. Got {type(base_freq).__name__}.")
        
    if scale is not None:
        if not isinstance(scale, str):
            raise TypeError(f"[ERROR] Calculate Step Bands: Scale must be a string. Got {type(scale).__name__}.")
    else:
        scale = "none"
        
    if not isinstance(dedup, bool):
        raise TypeError(f"[ERROR] Calculate Step Bands: Dedup must be a boolean. Got {type(dedup).__name__}.")
        
    # Value checking
    if lowcut <= 0:
        raise ValueError(f"[ERROR] Calculate Step Bands: Lowcut frequency must be in (0, Nyq]. Got {lowcut}.")
    if highcut <= 0:
        raise ValueError(f"[ERROR] Calculate Step Bands: Highcut frequency must be in (0, Nyq]. Got {highcut}.")
    if steps <= 0 or steps > 1000:
        raise ValueError(f"[ERROR] Calculate Step Bands: Steps must be in (0, 1000]. Got {steps}.")
    if base_freq <= 0:
        raise ValueError(f"[ERROR] Calculate Step Bands: Base frequency must be in (0, Nyq]. Got {base_freq}.")
    
    scale = scale.lower()
    if scale not in ["octave", "note", "none"]:
        raise ValueError(f"[ERROR] Calculate Step Bands: Scale must be one of [octave, note, none]. Got {scale}.")
    
    # Set parameters for custom scales
    if scale == "octave":
        steps, base_freq = 1, 125.0
    if scale == "note":
        steps, base_freq = 12, 440.0
    
    # Seek a valid base frequency
    while base_freq * 2 ** (1 / steps) < lowcut:
        base_freq *= 2 ** (1 / steps)
    
    while base_freq / 2 ** (1 / steps) > lowcut:
        base_freq /= 2 ** (1 / steps)
        
    if base_freq > lowcut and base_freq >= highcut:
        raise ValueError(f"[ERROR] Calculate Step Bands: Base frequency must be in [{lowcut}, {highcut}]. Got [{lowcut}, {base_freq}, {highcut}].")
        
    # Generate list of centre frequencies 
    step_freqs = [base_freq]
    while step_freqs[-1] * 2 ** (1 / steps) < highcut:
        freq = step_freqs[-1] * 2 ** (1 / steps)
        step_freqs.append(freq)
        
    step_freqs = Array.validate(array=step_freqs)
        
    # Fit centre frequencies to step centre frequencies
    if centre_freqs is not None:
        # Validate centre frequencies array
        centre_freqs = Array.validate(array=centre_freqs)
        if centre_freqs.shape != (centre_freqs.shape[0], 1):
            raise ValueError(f"[ERROR] Calculate Step Bands: Centre frequencies must be a 1-dim array. Got {centre_freqs.shape}.")
        if np.any(centre_freqs <= 0) or np.any(centre_freqs >= 96000):
            raise ValueError(f"[ERROR] Calculate Step Bands: Centre frequencies must be in (0, Nyq].")
        
        # Fit frequencies
        fitted_freqs = []
        
        for cf in centre_freqs:
            # Get nearest step frequency and fit input frequency to step value
            distances = np.abs(step_freqs - cf)
            near_idx = np.max(np.where(distances == np.min(distances))[0])
            near_cf = step_freqs[near_idx]
            fitted_freqs.append(near_cf)
        
        # Reset to selected values if centre frequencies are given
        step_freqs = fitted_freqs
            
    # Compute lowcuts and highcuts to get bands
    factor = 2 ** (1 / (2 * steps))
    lowcuts = calc_lowcut(frequency=step_freqs, factor=factor)
    highcuts = calc_highcut(frequency=step_freqs, factor=factor)
    step_bands = np.hstack([lowcuts, step_freqs, highcuts])
    
    # Remove duplicates
    if dedup:
        step_freqs = np.unique(step_freqs, axis=0)
        step_bands = np.unique(step_bands, axis=0)
        
    return step_bands


def calc_erb(frequency: float) -> float:
    """
    Calculate the Equivalent Rectangular Bandwidth (ERB) of a given frequency.
    
    Formula:
        f(frequency) = 24.7 * (0.00437 * frequency + 1)

    Arguments:
        frequency (float): The frequency in Hz for which the ERB is calculated. Must be a positive float.

    Returns:
        float: The ERB value for the given frequency.

    Raises:
        ValueError: If the input frequency is not a positive float.
    """
    
    if isinstance(frequency, (list, np.ndarray)):
        frequency = Array.validate(array=frequency)
    
    if not isinstance(frequency, (int, float, np.ndarray)) or np.any(frequency <= 0):
        raise ValueError(f"[ERROR] ERB: Input frequency must be a positive float. Got {type(frequency).__name__} {frequency}.")
    
    erb_value = 24.7 * (0.00437 * frequency + 1)
    
    return erb_value


def hz2erb_rate(frequency: float) -> float:
    """
    Convert a given frequency to its corresponding ERB rate.
    
    Formula:
        f(frequency) = 21.4 * log_10(0.00437 * frequency + 1)

    Arguments:
        frequency (float): The frequency in Hz to be converted. Must be a positive float.

    Returns:
        float: The ERB rate corresponding to the given frequency.

    Raises:
        ValueError: If the input frequency is not a positive float.
    """
    
    if isinstance(frequency, (list, np.ndarray)):
        frequency = Array.validate(array=frequency)
    
    if not isinstance(frequency, (int, float, np.ndarray)) or np.any(frequency <= 0):
        raise ValueError(f"[ERROR] Hz to ERB Rate: Input frequency must be a positive float. Got {type(frequency).__name__} {frequency}.")
    
    erb_rate = Transforms.logn_transform(signal=0.00437 * frequency + 1, coefficient=21.4, base=10)
    if isinstance(frequency, (int, float)):
        erb_rate = erb_rate[0][0]
    
    return erb_rate


def erb_rate2hz(erb: float) -> float:
    """
    Convert an ERB rate to its corresponding frequency in Hz.
    
    Formula:
        f(ERB) = (10 ^ (ERB / 21.4) - 1) / 0.00437

    Arguments:
        erb (float): The ERB rate to be converted. Must be a positive float.

    Returns:
        float: The frequency in Hz corresponding to the given ERB rate.

    Raises:
        ValueError: If the input ERB rate is not a positive float.
    """
    
    if isinstance(erb, (list, np.ndarray)):
        erb = Array.validate(array=erb)
    
    if not isinstance(erb, (int, float, np.ndarray)) or np.any(erb <= 0):
        raise ValueError(f"[ERROR] ERB Rate to Hz: Input ERB must be a positive float. Got {type(erb).__name__} {erb}.")
    
    frequency = (10 ** (erb / 21.4) - 1) / 0.00437
    
    return frequency


def erb_centre_freqs(
    lowcut: float, 
    highcut: float, 
    band_count: int = 55
) -> np.ndarray:
    """
    Calculate the center frequencies of ERB bands within a given frequency range.

    Arguments:
        lowcut (float): The lower bound of the frequency range in Hz. Must be a positive float.
        highcut (float): The upper bound of the frequency range in Hz. Must be a positive float greater than lowcut.
        band_count (int, optional): The number of ERB bands. Must be a positive integer. Default is 55.

    Returns:
        np.ndarray: A numpy array of center frequencies for the ERB bands within the specified range.

    Raises:
        ValueError: If the input frequencies are not positive floats or if band_count is not a positive integer.
    """
    
    if lowcut >= highcut:
        raise ValueError(f"[ERROR] ERB centre frequencies: Upper frequency must be greater than lower frequency. Got {lowcut} > {highcut}.")
    if not isinstance(band_count, int) or band_count <= 0:
        raise ValueError(f"[ERROR] ERB centre frequencies: Band count must be a positive integer. Got {type(band_count).__name__} {band_count}.")
    
    erb_rates = Array.linseries(
                    start=hz2erb_rate(lowcut), 
                    end=hz2erb_rate(highcut), 
                    size=band_count,
                    endpoint=True
                )

    centre_frequencies = Array.validate(array=erb_rate2hz(erb=erb_rates))
    
    return centre_frequencies


def note2frequency(
    note: np.ndarray, 
    octave: np.ndarray, 
    base_frequency: float = 261.63, 
    base_octave: int = 4, 
    alpha: int = 12
) -> np.ndarray:
    """
    Convert musical note and octave to frequency.
    
    Math:
        f(n, o) = f_0 * 2^(n / a) * 2^(o - o_0),    0 <= n < a
        
        where:
            f_0 = 261.63 is the base frequency
            a = 12 is the number of steps in an octave
            o_0 = 4 is the base frequency
            n is the nth step in an octave
            o is the octave
    
    Usage:
        # Get frequencies for [C4, E4, B5]
        notes = [0, 4, 11]
        octaves = [4, 4, 5]
        frequencies = note2frequency(note=notes, octave=octaves)

    Args:
        note (np.ndarray): Array of note values.
        octave (np.ndarray): Array of octave values.
        base_frequency (float, optional): Base frequency for reference note. Must be positive. Defaults to 261.63.
        base_octave (int, optional): Base octave for reference note. Defaults to 4.
        alpha (int, optional): Number of notes per octave. Must be positive. Defaults to 12.

    Returns:
        np.ndarray: Array of corresponding frequencies.

    Raises:
        ValueError: If note or octave array dimensions exceed 2.
        ValueError: If note and octave shapes do not match required conditions.
        ValueError: If any inputs are not as expected
    """
    
    if not isinstance(base_frequency, float) or base_frequency <= 0:
        raise ValueError(f"[ERROR] Note to frequency: Base frequency must be a positive float. Got {type(base_frequency).__name__} {base_frequency}.")
    if not isinstance(base_octave, int):
        raise ValueError(f"[ERROR] Note to frequency: Base octave must be an integer. Got {type(base_octave).__name__} {base_octave}.")
    if not isinstance(alpha, int) or alpha <= 0:
        raise ValueError(f"[ERROR] Note to frequency: Alpha must be a positive integer. Got {alpha}.")

    note = Array.validate(array=note)
    octave = Array.validate(array=octave)

    if note.ndim > 2 or octave.ndim > 2:
        raise ValueError(f"[ERROR] Note to frequency: Note and octave must have up to 2 dimensions. Got {note.ndim} Note dims, {octave.ndim} Octave dims.")

    if note.shape != octave.shape:
        if octave.shape != (1, 1) and octave.shape != (note.shape[0], 1):
            raise ValueError(f"[ERROR] Note to frequency: Octave must have one value, match Notes sample size, or match Notes shape. Got Notes shape {note.shape} and Octaves shape {octave.shape}.")

    # Compute frequency of notes
    note_term = 2 ** (note / alpha)
    octave_term = 2 ** (octave - base_octave)
    frequency = base_frequency * note_term * octave_term

    return frequency


def frequency2note(
    frequency: np.ndarray, 
    lower_octave: int = 0, 
    upper_octave: int = 9, 
    alpha: int = 12, 
    labels: List[str] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert frequency to musical note and octave.
    
    Usage:
        # Get closest notes to input frequencies
        frequencies = [
            [100, 1000, 261], 
            [200, 2000, 329], 
            [400, 4000, 987]
        ]
        notes, names = frequency2note(frequency=frequencies)
        
        # Get notes from bandwidths
        frequencies = filter.get_bandwidths()
        notes, names = frequency2note(frequency=frequencies)

    Args:
        frequency (np.ndarray): Array of frequencies.
        lower_octave (int, optional): Lower bound of octave range. Defaults to 0.
        upper_octave (int, optional): Upper bound of octave range. Defaults to 9.
        alpha (int, optional): Number of notes per octave. Must be positive. Defaults to 12.
        labels (List[str], optional): List of note labels. Defaults to standard 12-note chromatic scale.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of corresponding notes and names.

    Raises:
        ValueError: If the number of labels does not match alpha.
        ValueError: If any inputs are not as expected.
    """
    
    if not isinstance(lower_octave, int) or not isinstance(upper_octave, int):
        raise ValueError(f"[ERROR] Frequency to note: Lower octave and upper octave must be integers. Got {type(lower_octave).__name__} and {type(upper_octave).__name__}.")
    if not isinstance(alpha, int) or alpha <= 0:
        raise ValueError(f"[ERROR] Frequency to note: Alpha must be a positive integer. Got {type(alpha).__name__} {alpha}.")
    if len(labels) != alpha:
        raise ValueError(f"[ERROR] Frequency to note: Alpha and number of labels must be equal. Got {len(labels)} labels and alpha={alpha}.")

    # Generate lists of note frequencies and names
    notes = []
    names = []
    for octave in range(lower_octave, upper_octave + 1):
        for note in range(alpha):
            note_frequency = note2frequency(note=np.array([[note]]), octave=np.array([[octave]]), alpha=alpha)
            note_name = labels[note] + str(octave)
            notes.append(note_frequency)
            names.append(note_name)

    # Validate data
    notes = Array.validate(array=notes)
    names = Array.validate(array=names, coerce="str", numeric=False)
    frequency = Array.validate(array=frequency)

    # Transform frequencies to log space
    log_notes = Transforms.logn_transform(signal=notes)
    log_frequency = Transforms.logn_transform(signal=frequency)

    # Get index of closest known notes to input frequencies
    deltas = np.abs(log_notes - log_frequency)
    index = np.argmin(deltas, axis=0)

    # Get the note frequencies and names
    notes = np.squeeze(notes[index])
    names = np.squeeze(names[index])

    # Validate data again
    if notes.shape == ():
        notes = notes.reshape((1, 1))
        names = names.reshape((1, 1))
        
    notes = Array.validate(array=notes)
    names = Array.validate(array=names, coerce="str", numeric=False)

    return notes, names


def fft_frequencies(size: int, sample_rate: int = None) -> np.ndarray:
    """
    Generate the FFT frequency bins for a given size.
    
    Formula:
        N = size
        m_1 = floor((N - 1) / 2) + 1
        m_2 = floor(N / 2)
        pos_series = [0, 1, 2, ..., m_1 - 1]
        neg_series = [-m_2, -m_2 + 1, ..., -1]
        indices = [pos_series, neg_series]
        frequencies = indices / N

    Arguments:
        size (int): The size of the FFT window.
        sample_rate (int, optional): The sample rate of the FFT window. Default to None.

    Returns:
        np.ndarray: A numpy array containing the frequency bins.
    
    Raises:
        ValueError: If size is not a positive integer.
    """
    
    sample_rate = val_sample_rate(sample_rate=sample_rate)
    
    if not isinstance(size, int):
        raise TypeError(f"[ERROR] FFT frequencies: Size must be an int. Got {type(size).__name__}.")
    if size <= 0:
        raise ValueError(f"[ERROR] FFT frequencies: Size must be a positive integer. Got {size}.")

    # Set a frequencies array and compute size margins
    frequencies = np.zeros((size, 1))
    margins = [(size - 1) // 2 + 1, size // 2]
    
    # Create positive and negative index arrays
    pos_series = Array.linseries(start=0, end=margins[0], endpoint=False)
    neg_series = Array.linseries(start=-margins[1], end=0, endpoint=False)
    
    # Insert index arrays and scale to frequency
    frequencies[:margins[0], :] = pos_series[:]
    frequencies[margins[0]:, :] = neg_series[:]
    frequencies = frequencies / size * sample_rate
    
    return frequencies