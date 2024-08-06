from typing import Union, List, Optional
import numpy as np
from . import utils as Utils
from . import array as Array


# Window gateway function

def choose_window(
    length: float, 
    name: str = "hamming", 
    args: dict = {},
    mode: str = "run"
) -> Union[np.ndarray, bool]:
    """
    Gateway function for pre-selected default window functions. Can be used to check availability and to generate windows.

    Args:
        length (int): The length of the window.
        name (str, optional): The name of the generator function. Defaults to "sine".
        args (dict, optional): Input parameters for the generator function. Defaults to {}.
        mode (str, optional): Gateway mode to check/run generators. Defaults to "run".

    Returns:
        Union[np.ndarray, bool]: The gateway output for the generator signal.
    """
    
    args["length"] = length
    windows = {
        "rectangular": rectangular,
        "triangle": triangle,
        "sine": sine,
        "hann": hann,
        "hamming": hamming,
        "blackman": blackman,
        "flat_top": flat_top,
        "generalized_gaussian": generalized_gaussian,
        "gaussian": gaussian,
        "confined_gaussian": confined_gaussian,
        "exponential": exponential,
        "tukey": tukey,
        "kaiser": kaiser
    }
    
    gen_output = Utils.gateway(name=name, args=args, functions=windows, mode=mode)
    
    return gen_output


# Linear window

def linear(length: int) -> np.ndarray:
    """
    Generate a Linear window.
    
    Arguments:
        length (int): The length of the window.

    Returns:
        np.ndarray: The Linear window.

    Raises:
        TypeError: If length is not an integer.
        ValueError: If length is not positive.
    """
    
    if not isinstance(length, int):
        raise TypeError(f"[ERROR] Linear: Length must be an int. Got {type(length).__name__}.")
    if length <= 0:
        raise ValueError(f"[ERROR] Linear: Length must be positive. Got {length}.")
    
    # Generate linear series index
    window = Array.linseries(start=0, end=length, endpoint=True)
    
    return window


# Rectangular window

def rectangular(length: int, alpha: float = 0.0) -> np.ndarray:
    """
    Generate a Rectangular window. Supports special case for generating a window with zero boundaries.
    
    Theory:
        The rectangular window, also known as the boxcar, uniform, or Dirichlet window, is a fundamental window function 
        used in signal processing. It simply retains a contiguous segment of the signal while setting all other values to 
        zero, causing the signal to abruptly start and stop. This approach is useful for minimizing the mean square error 
        in the estimation of the Discrete-time Fourier Transform (DTFT), although it introduces issues like scalloping loss
        and limited dynamic range. Historically, the rectangular window has been essential in spectral analysis for its 
        simplicity and effectiveness in basic applications, despite the trade-offs in spectral leakage and resolution.
    
    Arguments:
        length (int): The length of the window.
        alpha (float, optional): The spacing parameter of the Rectangular window. Default is 0.0.

    Returns:
        np.ndarray: The Rectangular window.

    Raises:
        TypeError: If length is not an integer or alpha is not a float.
        ValueError: If length is not positive or alpha is not between 0 and 1.
    """
    
    if not isinstance(length, int):
        raise TypeError(f"[ERROR] Rectangular: Length must be an int. Got {type(length).__name__}.")
    if not isinstance(alpha, (int, float)):
        raise TypeError(f"[ERROR] Rectangular: Alpha must be a float or int. Got {type(alpha).__name__}.")
    if length <= 0:
        raise ValueError(f"[ERROR] Rectangular: Length must be positive. Got {length}.")
    if not (0 <= alpha <= 1):
        raise ValueError(f"[ERROR] Rectangular: Alpha must be between 0 and 1. Got {alpha}.")
    
    # Generate a series of ones
    window = Array.validate(array=[1.0] * length)
    
    # Compute space margin size
    delta = int(alpha * length // 2)

    # Compute the rectangular window
    if delta > 0:
        window[:delta, :] = 0.0
        window[-delta:, :] = 0.0
    
    return window


# B-spline window

def triangle(length: int, delta: float = 0) -> np.ndarray:
    """
    Generate a Triangle window, or 2nd-order B-spline window.
    
    Theory:
        The Triangular window, also known as the Bartlett or Fejér window, is a 2nd-order B-spline window function that is 
        particularly useful in signal processing for smoothing and reducing spectral leakage. It achieves this by 
        convolving two rectangular windows of half-width, resulting in a piecewise linear function that tapers the signal's 
        edges to zero. This window is beneficial for its simplicity and effectiveness in various applications, especially 
        when a balance between main lobe width and side lobe levels is desired. The triangular window was developed to 
        improve upon the basic rectangular window by providing a smoother transition, making it an essential tool in 
        spectral analysis. Historical references indicate its development was influenced by earlier work in window 
        functions to enhance signal clarity and analysis accuracy.

    Math:
        The Triangle window is defined as:
            w(n) = 1 - |(n - N/2) / (L/2)|,     0 <= n <= N,
            
        where 
            N is the length of the window, 
            delta is a parameter that controls the shape of the window, and 
            L can be N + delta. 
            
            The delta = 0 case is known as the Bartlett window.

    Arguments:
        length (int): The length of the window.
        delta (float, optional): The shape parameter of the Triangle window. Default is 0.

    Returns:
        np.ndarray: The Triangle window.

    Raises:
        TypeError: If delta is not a float or integer.
        ValueError: If length + delta = 0.
    """
    
    if not isinstance(delta, (float, int)):
        raise TypeError(f"[ERROR] Triangle: Delta must be a float or integer. Got {type(delta).__name__}.")
    if length + delta == 0:
        raise ValueError(f"[ERROR] Triangle: Length + Delta must be non-zero. Got {length} + {delta} = 0.")
    
    # Generate linear series index
    index = linear(length=length)
    
    # Compute the triangle window
    window = 1 - np.abs((index - length / 2) / ((length + delta) / 2))
    
    return window


# Sine window

def sine(length: int) -> np.ndarray:
    """
    Generate a Sine window.
    
    Theory:
        The sine window, also known as the cosine window, half-sine window, or half-cosine window, is a function used in 
        signal processing to taper the edges of a signal smoothly to zero. This window function minimizes spectral leakage 
        by ensuring that the signal transitions gradually, reducing abrupt discontinuities. The sine window's 
        autocorrelation results in the Bohman window, adding to its utility. Although specific historical details regarding 
        its development are scarce, the sine window is a foundational tool in digital signal processing, widely adopted for 
        its effectiveness in improving signal analysis and processing outcomes.

    Math:
        The Sine window is defined as:
            w(n) = sin(pi * n / N),     0 <= n <= N
            
        where:
            N is the length of the window

    Arguments:
        length (int): The length of the window.

    Returns:
        np.ndarray: The Sine window.
    """
    
    # Generate linear series index
    index = linear(length=length)
    
    # Compute the Sine window
    window = np.sin(np.pi * index / length)
    
    return window


# Cosine windows

def generalized_cosine(length: int, terms: int, alphas: Union[np.ndarray, List, int, float]) -> np.ndarray:
    """
    Generate a generalized cosine window.
    
    Theory:
        The cosine-sum windows, also known as generalized cosine windows, are mathematical functions used in signal 
        processing to mitigate the discontinuities at the boundaries of a signal segment. This window function is expressed 
        as a sum of cosine terms, each scaled by a coefficient, where the coefficients are usually non-negative. The 
        primary utility of the cosine-sum window lies in its ability to reduce spectral leakage, thereby improving the 
        frequency resolution of the signal's discrete Fourier transform (DFT). Special cases of these windows include the 
        well-known Hamming and Hann windows, which are specific instances characterized by particular sets of coefficients. 
        The development of these windows dates back to early work in digital signal processing, attributed to the 
        pioneering efforts of researchers like Richard Hamming and Julius von Hann, who sought to enhance the analysis of 
        finite-length signals by addressing edge effects.

    Math:
        The generalized cosine window is defined as:
            w(n) = sum_{k=0}^K (-1)^k * a_k * cos(2 * pi * k * n / N),      0 <= n <= N
            
        where:
            N is the length of the window
            K is the number of cosine terms
            a_k is the kth coefficient, which controls the shape of the window

    Arguments:
        length (int): The length of the window.
        terms (int): The number of cosine terms in the generalized cosine window.
        alphas (ndarray, List, int, float): The coefficients to each term.

    Returns:
        np.ndarray: The generalized cosine window.

    Raises:
        TypeError: If terms is not an integer or alphas is not an array, list, float or integer.
        ValueError: If terms is not non-negative, or number of alphas doesn't match number of terms.
    """
    
    if not isinstance(terms, int):
        raise TypeError(f"[ERROR] Generalized cosine: Terms must be an int. Got {type(terms).__name__}.")
    if not isinstance(alphas, (np.ndarray, tuple, list, int, float)):
        raise TypeError(f"[ERROR] Generalized cosine: Alphas must be an array or number. Got {type(alphas).__name__}.")
    
    if terms <= 0:
        raise ValueError(f"[ERROR] Generalized cosine: Terms must be positive. Got {terms}.")
    
    alphas = Array.validate(array=alphas)
    
    if np.any(alphas < 0):
        print(f"[WARNING] Generalized cosine: Alpha coefficients are generally positive. Got at least one negative alpha, which may potentially affect related operations.")
    
    if terms != alphas.shape[0]:
        raise ValueError(f"[ERROR] Generalized cosine: Number of terms and alphas must match. Got {terms} != {alphas.shape[0]}.")
    
    # Generate linear series index
    index = linear(length=length)
    
    # Compute the generalized cosine window
    window = np.zeros(index.shape)
    for k in range(terms):
        window += (-1) ** k * alphas[k] * np.cos(2 * np.pi * k * index / length)
    
    return window


def generalized_hann(length: int, delta: float = 0.5) -> np.ndarray:
    """
    Generate a Generalized Hann window.
    
    Theory:
        The Hann window, also known as the raised cosine window, is a type of window function named after Julius von Hann. 
        It is sometimes erroneously referred to as the Hanning window due to its similarity to the Hamming window. This 
        window function is used to taper the edges of a signal to zero, reducing spectral leakage when performing a Fourier 
        transform. By smoothing the signal edges, it minimizes discontinuities, making it useful in signal processing and 
        analysis. The Hann window is a special case where a_0 = 0.5. Developed in the early 20th century, this window 
        function aids in reducing the artifacts in the frequency domain representation of signals.

    Math:
        The Generalized Hann window is defined as:
            w(n) = sum_{k=0}^K (-1)^k * a_k * cos(2 * pi * k * n / N),      0 <= n <= N
            
        Where:
            K = 1
            a_0 = delta
            a_1 = 1 - delta
            N is the length of the window
            delta is a parameter to adjust the Hann window shape

    Arguments:
        length (int): The length of the window.
        delta (float, optional): The shape parameter.

    Returns:
        np.ndarray: The Generalized Hann window.

    Raises:
        TypeError: If delta is not float or integer.
    """
    
    if not isinstance(delta, (int, float)):
        raise TypeError(f"[ERROR] Generalized Hann: Delta must be an int or float. Got {type(delta).__name__}.")
    
    # Compute the Generalized Hann window
    alphas = [delta, 1 - delta]
    window = generalized_cosine(length=length, terms=len(alphas), alphas=alphas)
    
    return window


def hann(length: int) -> np.ndarray:
    """
    Generate a Hann window.
    
    Theory:
        See Generalized Hann window.

    Math:
        The Hann window is defined as:
            w(n) = sum_{k=0}^K (-1)^k * a_k * cos(2 * pi * k * n / N),      0 <= n <= N
            
        Where:
            K = 1
            a_0 = 0.5
            a_1 = 0.5
            N is the length of the window

    Arguments:
        length (int): The length of the window.

    Returns:
        np.ndarray: The Hann window.
    """
    
    # Compute the Hann window
    window = generalized_hann(length=length)
    
    return window


def hamming(length: int) -> np.ndarray:
    """
    Generate a Hamming window.
    
    Theory:
        The Hamming window, also known as the Hamming blip when used for pulse shaping, is a window function developed by 
        Richard W. Hamming. It is designed to minimize the first sidelobe, placing a zero-crossing at frequency 5π/(N - 1) 
        and reducing the sidelobe height to about one-fifth that of the Hann window. This function tapers the signal to 
        reduce spectral leakage, making it particularly useful in signal processing for applications like spectral analysis 
        and filter design. Developed in the mid-20th century, the Hamming window has become a standard tool due to its 
        balance between mainlobe width and sidelobe suppression, improving the accuracy and resolution of the frequency 
        analysis.

    Math:
        The Hamming window is defined as:
            w(n) = sum_{k=0}^K (-1)^k * a_k * cos(2 * pi * k * n / N),      0 <= n <= N
            
        Where:
            K = 1
            a_0 = 25/46 
            a_1 = 1 - 25/46
            N is the length of the window

    Arguments:
        length (int): The length of the window.

    Returns:
        np.ndarray: The Hamming window.
    """
    
    # Compute the Hamming window
    window = generalized_hann(length=length, delta=25/46)
    
    return window


def blackman(length: int, delta: float = 0.16) -> np.ndarray:
    """
    Generate a Blackman window.
    
    Theory:
        The Blackman window, sometimes referred to as the Blackman-Harris window, is a tapering function used in signal 
        processing to reduce spectral leakage. Developed by Richard Blackman in 1958, this window smooths the edges of a 
        signal by applying a specific weighted cosine series, effectively minimizing discontinuities and improving 
        frequency resolution. With its common parameters (a = 0.16), the Blackman window balances main lobe width and side 
        lobe suppression, making it particularly useful for Fourier Transform applications. This window function is favored 
        in scenarios where both high dynamic range and minimal spectral leakage are required, such as in audio signal 
        processing and other fields where precise frequency analysis is crucial.

    Math:
        The Blackman window is defined as:
            w(n) = sum_{k=0}^K (-1)^k * a_k * cos(2 * pi * k * n / N),      0 <= n <= N
            
        Where:
            K = 2
            a_0 = (1 - delta) / 2
            a_1 = 1 / 2
            a_2 = delta / 2
            N is the length of the window and 
            delta is a parameter that controls the shape of the window. 
            
        The the Blackman's "not very serious proposal" uses delta = 0.16.

    Arguments:
        length (int): The length of the window.
        delta (float, optional): The shape parameter.

    Returns:
        np.ndarray: The Blackman window.

    Raises:
        TypeError: If delta is not float or integer.
    """
    
    if not isinstance(delta, (int, float)):
        raise TypeError(f"[ERROR] Blackman: Delta must be an int or float. Got {type(delta).__name__}.")
    
    # Compute the Blackman window
    alphas = [(1 - delta) / 2, 1 / 2, delta / 2]
    window = generalized_cosine(length=length, terms=len(alphas), alphas=alphas)
    
    return window


def flat_top(length: int) -> np.ndarray:
    """
    Generate a Flat Top window.
    
    Theory:
        A flat top window, also known as a flat top weighting function, is a partially negative-valued window function 
        designed to minimize scalloping loss in the frequency domain, making it particularly useful for accurate amplitude 
        measurement of sinusoidal frequency components. By applying this window to a signal, the frequency resolution is 
        broadened, which can result in increased noise bandwidth and a wider frequency selection—benefits that can vary by 
        application. Flat top windows can be constructed using low-pass filter design methods or through the cosine-sum 
        approach. Developed in the mid-20th century, primarily by researchers in the fields of signal processing and 
        acoustics, this window function addresses the need for precise amplitude measurements in various scientific and 
        engineering contexts. Its unique properties make it a vital tool for applications where accurate amplitude 
        detection is crucial.
        
    Math:
        The Flat Top window is defined as:
            w(n) = sum_{k=0}^K (-1)^k * a_k * cos(2 * pi * k * n / N),      0 <= n <= N
            
        Where:
            K = 4
            a_0 = 0.21557895
            a_1 = 0.41663158
            a_2 = 0.277263158
            a_3 = 0.083578947
            a_4 = 0.006947368
            N is the length of the window.

    Arguments:
        length (int): The length of the window.

    Returns:
        np.ndarray: The Flat Top window.
    """
    
    # Compute the Flat Top window
    alphas = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
    window = generalized_cosine(length=length, terms=len(alphas), alphas=alphas)
    
    return window


# Gaussian windows

def generalized_gaussian(
    length: int, 
    mu: Optional[float] = None, 
    sigma: float = None, 
    alpha: Optional[float] = np.sqrt(2), 
    power: float = 2
) -> np.ndarray:
    """
    Generate a Generalized Gaussian window.
    
    Theory:
        The generalized normal window, also known as the generalized Gaussian window, is a versatile window function used 
        in signal processing. It modifies a signal by applying an exponential function that varies based on a parameter 
        (p), allowing it to transition smoothly from a Gaussian window (p = 2) to a rectangular window as (p) approaches 
        infinity. This adaptability makes it valuable for controlling spectral leakage and frequency resolution, offering a 
        balanced compromise between the Gaussian and rectangular windows. Developed to enhance time-frequency 
        representation, this window function provides adjustable bandwidth and amplitude attenuation, proving useful in 
        various applications where signal smoothness and control are crucial.

    Math:
        The Generalized Gaussian window is defined as:
            w(n) = exp(-((x + m) / (s * a)) ^ p),     0 <= n <= N
            
        where:
            N is the length of the window
            m is the shift parameter on x
            s is the standard deviation of the gaussian kernel
            a is the parameter to adjust the shape of the window
            p is the power of the kernel
            
    Arguments:
        length (int): The length of the window.
        mu (float, optional): The mean shift parameter. Defaults to 0.
        sigma (float, optional): The standard deviation of the kernel. Defaults to 1.
        alpha (float, optional): The shape parameter. Defaults to sqrt(2).
        power (float, optional): The power of the kernel. Defaults to 2.

    Returns:
        np.ndarray: The Generalized Gaussian window.

    Raises:
        TypeError: If sigma, mu, alpha, or power is not a float or integer.
        ValueError: If alpha is zero.
    """
    
    if mu is None:
        mu = length / 2
        
    if sigma is None:
        sigma = length / 7
    
    if not isinstance(sigma, (float, int)):
        raise TypeError(f"[ERROR] Generalized Gaussian: Sigma must be a float or integer. Got {type(sigma).__name__}.")
    if not isinstance(mu, (float, int)):
        raise TypeError(f"[ERROR] Generalized Gaussian: Mu must be a float or integer. Got {type(mu).__name__}.")
    if not isinstance(alpha, (float, int)):
        raise TypeError(f"[ERROR] Generalized Gaussian: Alpha must be a float or integer. Got {type(alpha).__name__}.")
    if not isinstance(power, (float, int)):
        raise TypeError(f"[ERROR] Generalized Gaussian: Power must be a float or integer. Got {type(power).__name__}.")
    
    if alpha == 0:
        raise ValueError(f"[ERROR] Generalized Gaussian: Alpha must be a non-zero number. Got {alpha}.")
    
    if sigma < 0:
        print(f"[WARNING] Generalized Gaussian: Sigma is generally positive. Got {sigma}, which may potentially affect related operations.")

    # Generate linear series index
    index = linear(length=length)
    
    # Compute the Generalized Gaussian window
    window = np.exp(-1 * ((index - mu) / (sigma * alpha)) ** power)
    
    return window


def gaussian(length: int, sigma: float = 0.4) -> np.ndarray:
    """
    Generate a Gaussian window.
    
    Theory:
        The Gaussian window function, also known as the Gaussian taper or Gaussian bell, is a windowing function used in 
        signal processing to minimize spectral leakage. It applies a Gaussian curve to the signal, effectively smoothing 
        the edges by reducing the amplitude at the ends of the window. This window is particularly useful because its 
        Fourier transform is also a Gaussian, making it ideal for applications requiring precise frequency estimation and 
        spectral analysis. A special case of the Gaussian window is when the standard deviation parameter, sigma, is set to 
        0.5, which provides an optimal balance between time and frequency resolution. The Gaussian window function was 
        developed by mathematicians and engineers in the mid-20th century to address the need for more accurate spectral 
        analysis tools. Its unique properties have made it a staple in both theoretical and practical signal processing.

    Math:
        The Gaussian window is defined as:
            w(n) = exp(-((x + N / 2) / (s * sqrt(2) * N / 2)) ^ 2),     0 <= n <= N
            
        where:
            N is the length of the window
            s is the standard deviation of the gaussian kernel
            
    Arguments:
        length (int): The length of the window.
        sigma (float, optional): The standard deviation of the kernel. Defaults to 0.4.

    Returns:
        np.ndarray: The Gaussian window.
    """
    
    if sigma > 0.5:
        print(f"[WARNING] Gaussian: Sigma is typically less than or equal to 0.5. Got {sigma}, which may potentially affect related operations.")
    
    # Compute the Gaussian window
    window = generalized_gaussian(length=length, mu=length / 2, sigma=sigma, alpha=np.sqrt(2) * length / 2, power=2)
    
    return window


def confined_gaussian(length: int, sigma: float = 0.1) -> np.ndarray:
    """
    Generate a Confined Gaussian window.
    
    Theory:
        The confined Gaussian window, also known as the minimum eigenvector window, optimizes the root mean square (RMS) 
        time-frequency bandwidth product by minimizing the RMS frequency width for a given temporal width. It shapes the 
        signal in a way that provides the smallest possible frequency spread, enhancing time-frequency analysis. This 
        window includes the sine window and the Gaussian window as special cases, corresponding to large and small temporal 
        widths, respectively. Developed through advancements in signal processing, it is valued for its precision in 
        balancing temporal and frequency resolution.

    Math:
        The Confined Gaussian window is defined as:
            w(n) = g(n) - g(-1 / 2) * (g(n + L) + g(n - L)) / (g(-1 / 2 + L) + g(-1 / 2 - L)),     0 <= n <= N
            
        where:
            g(n) = exp(-((x + N / 2) / (s * 2 * L)) ^ 2)
            L = N + 1
            N is the length of the window
            s is the standard deviation of the gaussian kernel
            
    Arguments:
        length (int): The length of the window.
        sigma (float, optional): The standard deviation of the kernel. Defaults to 0.1.

    Returns:
        np.ndarray: The Confined Gaussian window.
    """
    
    if sigma >= 0.14:
        print(f"[WARNING] Confined Gaussian: Sigma is typically less than 0.14. Got {sigma}, which may potentially affect related operations.")
        
    deltas = [-1 / 2, length, -length, -1 / 2 + length, -1 / 2 - length]
    alpha = 2 * (length + 1)
    g0 = generalized_gaussian(length=length, mu=length / 2 - deltas[0], sigma=sigma, alpha=alpha, power=2)[0][0]
    g1 = generalized_gaussian(length=length, mu=length / 2 - deltas[1], sigma=sigma, alpha=alpha, power=2)
    g2 = generalized_gaussian(length=length, mu=length / 2 - deltas[2], sigma=sigma, alpha=alpha, power=2)
    g3 = generalized_gaussian(length=length, mu=length / 2 - deltas[3], sigma=sigma, alpha=alpha, power=2)[0][0]
    g4 = generalized_gaussian(length=length, mu=length / 2 - deltas[4], sigma=sigma, alpha=alpha, power=2)[0][0]
    
    # Compute the Confined Gaussian window
    window = generalized_gaussian(length=length, mu=length / 2, sigma=sigma, alpha=alpha, power=2)
    window -= (g0 * (g1 + g2)) / (g3 + g4)
    
    return window


# Other windows

def exponential(length: int, alpha: Optional[float] = None, decay: float = 8.69) -> np.ndarray:
    """
    Generate a Exponential window.
    
    Theory:
        The Poisson window, also known as the exponential window, is a window function that increases exponentially towards 
        its center and decreases exponentially towards its edges. This window function shapes the signal by applying an 
        exponential weighting, which helps reduce spectral leakage, making it useful in signal processing tasks such as 
        spectral analysis and filtering. A notable characteristic of the Poisson window is its non-zero values at the 
        boundaries, unlike other window functions that typically approach zero. It is believed to have been developed to 
        address the limitations of traditional window functions, providing a balance between frequency resolution and 
        side-lobe attenuation. The exponential window's unique properties make it especially effective in applications 
        requiring precise control over signal decay and window shape.

    Math:
        The Exponential window is defined as:
            w(n) = exp(-|n - N/2| * 1 / alpha),     0 <= n <= N
            
        where:
            N is the length of the window
            Alpha is the time constant of the function.
            
        Special case:
            Alpha = Alpha' * 8.69 / D
        where:
            D is the targeted decay over half the window in dB
            Alpha is the final time constant determined by alpha' and decay D

    Arguments:
        length (int): The length of the window.
        alpha (float, optional): The shape parameter of the Exponential window. Default is N/2.
        decay (float, optional): The targeted decay of hald the window. Default is 8.69.

    Returns:
        np.ndarray: The Exponential window.

    Raises:
        TypeError: If alpha is not a float or integer.
        ValueError: If alpha is zero.
    """
    
    if alpha is None:
        alpha = length / 2
    
    if not isinstance(alpha, (float, int)):
        raise TypeError(f"[ERROR] Exponential: Alpha must be a float or integer. Got {type(alpha).__name__}.")
    if not isinstance(decay, (float, int)):
        raise TypeError(f"[ERROR] Exponential: Decay must be a float or integer. Got {type(decay).__name__}.")
    
    if alpha == 0:
        raise ValueError(f"[ERROR] Exponential: Alpha must be a non-zero number. Got {alpha}.")
    if decay == 0:
        raise ValueError(f"[ERROR] Exponential: Decay must be a non-zero number. Got {decay}.")
    
    if alpha < 0:
        print(f"[WARNING] Exponential: Alpha is generally positive. Got {alpha}, which may potentially affect related operations.")
    if decay < 0:
        print(f"[WARNING] Exponential: Decay is generally positive. Got {decay}, which may potentially affect related operations.")

    # Generate linear series index
    index = linear(length=length)
    
    # Compute the Exponential window
    window = np.exp(-np.abs(index - length / 2) / (alpha * 8.69 / decay))
    
    return window


def tukey(length: int, alpha: float = 0.5) -> np.ndarray:
    """
    Generate a Tukey window.
    
    Theory:
        The Tukey window, also known as the cosine-tapered window, is a versatile window function that shapes the signal by 
        tapering the beginning and end with a cosine lobe while maintaining a rectangular shape in the middle. This window 
        is useful for reducing spectral leakage in signal processing, as it smooths the discontinuities at the edges of the 
        signal. The Tukey window transitions smoothly between a rectangular window (when a = 0) and a Hann window (when a = 
        1), making it adaptable for various applications. Developed by John Tukey in the mid-20th century, this window 
        function was created to provide a flexible tool for analyzing signals with varying characteristics. Its ability to 
        control the trade-off between main-lobe width and side-lobe level makes it particularly valuable in applications 
        requiring precise spectral analysis.

    Math:
        The Tukey window is defined as:
            beta = alpha * N / 2
            w(n) = 1 / 2 * (1 - cos(2 * pi * n / (alpha * N))),             0 <= n < beta
            w(n) = 1                                                        beta <= n <= N - beta
            w(n) = 1 / 2 * (1 - cos(2 * pi * (n + beta) / (alpha * N))),    N - beta < n <= N
            
        where:
            N is the length of the window
            Alpha is a parameter that controls the shape of the window
            Beta is the threshold between rectangular and cosine sections of the window
        
        alpha = 0 is the simple rectangular window.
        alpha = 1 is the Hann window.

    Arguments:
        length (int): The length of the window.
        alpha (float, optional): The shape parameter of the Tukey window. Default is 0.5.

    Returns:
        np.ndarray: The Tukey window.

    Raises:
        TypeError: If alpha is not a float or integer.
        ValueError: If alpha is not in [0, 1]
    """
    
    if not isinstance(alpha, (float, int)):
        raise TypeError(f"[ERROR] Tukey: Alpha must be a float or integer. Got {type(alpha).__name__}.")
    
    if alpha < 0 or alpha > 1:
        raise ValueError(f"[ERROR] Tukey: Alpha must be a number in [0, 1]. Got {alpha}.")

    # Generate linear series index
    index = linear(length=length)
    
    # Compute the threshold boundary
    boundary = alpha * length / 2
    
    # Compute the Tukey window
    # Set to rectangular window for n <= |alpha * N / 2|
    window = np.ones(index.shape)
    
    # Hann window when alpha -> 1, skip if alpha = 0:
    if alpha != 0 and alpha != 1:
        cosine_window = 1 / 2 * (1 - np.cos(2 * np.pi * index / (alpha * length)))
        
        cover = sum(index < boundary)[0]
        
        # Set to left side generalized window for 0 <= n < alpha * N / 2
        window[:cover] = cosine_window[:cover]
        
        # Set to left side generalized window for alpha * N / 2 < n <= N
        window[-cover:] = np.flip(cosine_window[:cover])
        
    elif alpha == 1:
        window = hann(length=length)
    
    return window


def kaiser(length: int, delta: float = 3) -> np.ndarray:
    """
    Generate a Kaiser window.
    
    Theory:
        The Kaiser window, also known as the Kaiser-Bessel window, is a type of window function introduced by James Kaiser. 
        This window function applies a taper to the signal, reducing spectral leakage by controlling the trade-off between 
        the main lobe width and side lobe levels using the parameter a (alpha). It approximates the Discrete Prolate 
        Spheroidal Sequence (DPSS) window by utilizing Bessel functions. Developed in the mid-20th century, the Kaiser 
        window is particularly useful in signal processing for its ability to provide adjustable spectral characteristics, 
        making it adaptable to various applications. A notable feature is its balance between the main lobe width and the 
        suppression of side lobes, with a typical alpha value of 3.

    Math:
        The Kaiser window is defined as:
            beta = pi * delta
            w(n) = I0(beta * sqrt(1 - (2 * n / N - 1) ^ 2)) / I0(pi * delta), 0 <= n <= N
            
        where 
            N is the length of the window and 
            delta is a parameter that controls the shape of the window. 
            The I0 function is the 0th-order modified Bessel function of the first kind.

    Arguments:
        length (int): The length of the window.
        delta (float, optional): The shape parameter of the Kaiser window. Default is 3.

    Returns:
        np.ndarray: The Kaiser window.

    Raises:
        TypeError: If delta is not a float or integer.
    """
    
    if not isinstance(delta, (float, int)):
        raise TypeError(f"[ERROR] Kaiser: Delta must be a float or integer. Got {type(delta).__name__}.")
    
    # Function to compute the zeroth order modified Bessel function of the first kind
    def bessel_I0(x):
        if not isinstance(x, np.ndarray):
            x = Array.validate(array=x)
        result = np.ones(x.shape)
        term = np.ones(x.shape)
        k = 1
        while np.any(term) > 1e-10:
            term *= (x / (2 * k)) ** 2
            result += term
            k += 1
            
        return result
    
    # Generate linear series index
    index = linear(length=length)
    
    # Compute the kaiser window
    window = np.pi * delta * np.sqrt(1 - (2 * index / length - 1) ** 2)
    window = bessel_I0(window) / bessel_I0(np.pi * delta)
    
    return window