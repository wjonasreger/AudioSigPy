from typing import List, Optional, Union
import numpy as np
from . import array as Array

    
def find_local_maxima(signal: np.ndarray) -> List[np.ndarray]:
    """
    Find local maxima in a signal. A local maximum is a point which is higher than its 
    immediate neighbors and higher than or equal to all other points in a flat region.

    Arguments:
        signal (np.ndarray): The input signal.

    Returns:
        (List[np.ndarray, ...]): A list containing (N, 3) shaped arrays of indices for 
        left edges, midpoints, right edges of each flat region containing local maxima. 
        There are C arrays in the output list, one for each channel of the signal. 
        There is no option to pool the channel data for this function.
    """
    
    signal = Array.validate(array=signal)
    
    # Initialize output list and set the end limit
    end = signal.shape[0] - 1
    output = []
    
    # Iterate over channels
    for c in range(signal.shape[1]):
        left_edges = []
        right_edges = []
        midpoints = []
        i = 1
        
        # Search for peaks in the signal
        while i < end:
            # Check if current spot is higher than previous neighbor
            if signal[i - 1, c] < signal[i, c]:
                # Look at next neighbor and check for plateau
                next = i + 1
                while next < end and signal[next, c] == signal[i, c]:
                    next += 1
                    
                # If next neighbor is lower than current spot
                if signal[next, c] < signal[i, c]:
                    # Pick up the left, mid, and right points of the peak
                    left = i
                    right = next - 1
                    midpoint = (left + right) // 2
                    
                    left_edges.append(left)
                    right_edges.append(right)
                    midpoints.append(midpoint)
                    
                    # Push current spot to furthest neighbor searched
                    i = next
            
            # Push current spot forward one
            i += 1
            
        left_edges = Array.validate(array=left_edges, coerce="int")
        right_edges = Array.validate(array=right_edges, coerce="int")
        midpoints = Array.validate(array=midpoints, coerce="int")
        
        # Add peak data to output list
        peak_data = np.hstack([left_edges, midpoints, right_edges])
        output.append(peak_data)
        
    return output


def find_peaks(
    signal: np.ndarray, 
    height: Optional[Union[int, float]] = None, 
    width: Optional[Union[int, float]] = None, 
    distance: Optional[Union[int, float]] = None,
    include_edge_peaks: bool = True
) -> List[np.ndarray]:
    """
    Find all peaks in a signal according to parameters to identify peaks with certain characteristics.

    Args:
        signal (np.ndarray): The input signal.
        height (int or float, optional): The minimum height threshold of a peak. Defaults to None.
        width (int or float, optional): The minimum width threshold of peak width. Defaults to None.
        distance (int or float, optional): The minimum distance threshold between peaks. Defaults to None.
        include_edge_peaks (bool, optional): Include outer peaks when using distance thresholding. Defaults to True.

    Returns:
        List[np.ndarray]: The identified peaks in the signal given the thresholding parameters.
        
    Raises:
        TypeError: If height, width, or distance is not an integer or float. If include_edge_peaks is not bool.
    """
    
    if height is not None:
        if not isinstance(height, (int, float)):
            raise TypeError(f"[ERROR] Find Peaks: Height must be an integer or float. Got {type(height).__name__}.")
        if height < 0:
            print(f"[WARNING] Find Peaks: Height should be non-negative. Got {height}. This may impact related operations.")
            
    if width is not None:
        if not isinstance(width, (int, float)):
            raise TypeError(f"[ERROR] Find Peaks: Width must be an integer or float. Got {type(width).__name__}.")
        if width < 0:
            print(f"[WARNING] Find Peaks: Width should be non-negative. Got {width}. This may impact related operations.")
            
    if distance is not None:
        if not isinstance(distance, (int, float)):
            raise TypeError(f"[ERROR] Find Peaks: Distance must be an integer or float. Got {type(distance).__name__}.")
        if distance < 0:
            print(f"[WARNING] Find Peaks: Distance should be non-negative. Got {distance}. This may impact related operations.")
        if not isinstance(include_edge_peaks, bool):
            raise TypeError(f"[ERROR] Find Peaks: Include edge peaks must be a boolean. Got {type(include_edge_peaks).__name__}.")
    
    signal = Array.validate(array=signal)
    
    # Extracted local maxima of the signal
    local_maxima = find_local_maxima(signal)
    output = []
    
    # Iterate over channels to select desired peaks
    for c in range(signal.shape[1]):
        
        # Ensure peaks are found
        if local_maxima[c].shape != (1, 0):
            left_edges, peaks, right_edges = local_maxima[c][:, 0], local_maxima[c][:, 1], local_maxima[c][:, 2]
        
            peaks = Array.validate(array=peaks, coerce="int")
            left_edges = Array.validate(array=left_edges, coerce="int")
            right_edges = Array.validate(array=right_edges, coerce="int")
        
            keep = np.ones(peaks.shape)
        
            # Choose peaks by height
            if height is not None:
                peak_values = signal[peaks.squeeze()]
                select = peak_values >= height
                keep = keep * select
                
            # Choose peaks by width
            if width is not None:
                widths = right_edges - left_edges
                select = widths >= width
                keep = keep * select
            
            # Select peaks so far from height and width criteria
            keep = Array.validate(array=keep, numeric=False, coerce="bool")
            peaks = Array.validate(array=peaks[keep], coerce="int")
            keep = np.ones(peaks.shape)
                
            # Choose peaks by neighbor distances
            if distance is not None:
                distances = np.diff(peaks, axis=0)
                shape = list(distances.shape)
                shape[0] = 1
                temp = np.ones(shape)
                
                distances = np.hstack([
                    np.concatenate([temp * peaks[0], distances]),
                    np.concatenate([distances, temp * (signal.shape[0] - peaks[-1])])
                ])
                
                distances = Array.validate(array=distances, coerce="int")
                select = Array.validate(array=np.sum(distances > distance, axis=1) > 1, coerce="bool", numeric=False)
                if include_edge_peaks:
                    select[0] = True
                    select[-1] = True
                keep = keep * select
            
            keep = Array.validate(array=keep, numeric=False, coerce="bool")
            peaks = peaks[keep]
            
            output.append(peaks)
            
        # If no peaks are found, yield empty array
        else:
            output.append(local_maxima[c])
    
    return output


def find_first_peak(
    signal: np.ndarray, 
    normalize: bool = False, 
    threshold: float = 0.7,
    pool: bool = True,
    mode: str = "stable"
) -> np.ndarray:
    """
    Find the first peak in a time series signal.

    Args:
        signal (np.ndarray): A time series signal.
        normalize (bool, optional): Whether to normalize the signal. Default is False.
        threshold (float, optional): Threshold for clipping. Default is 0.7.
        pool (bool, optional): Whether to pool the channels. Default is True.
        mode (str, optional): Peak detection mode. Default is stable.

    Returns:
        np.ndarray: Index of the first peak in the signal. N indices given for N channels.
        
    Raise:
        TypeError: If normalize, pool are not boolean. If threshold is not an integer or float. If mode is not a string.
        ValueError: If threshold is not in [0, 1]. If mode is not experimental or stable.
    """
    
    if not isinstance(normalize, bool):
        raise TypeError(f"[ERROR] Find First Peak: Normalize must be a boolean. Got {type(normalize).__name__}.")
    if not isinstance(pool, bool):
        raise TypeError(f"[ERROR] Find First Peak: Pool must be a boolean. Got {type(pool).__name__}.")
    if not isinstance(threshold, (int, float)):
        raise TypeError(f"[ERROR] Find First Peak: Threshold must be an integer or float. Got {type(threshold).__name__}.")
    if not isinstance(mode, str):
        raise TypeError(f"[ERROR] Find First Peak: Mode must be a string. Got {type(mode).__name__}.")
    
    if threshold < 0 or threshold > 1:
        raise ValueError(f"[ERROR] Find First Peak: Threshold must be in [0, 1]. Got {threshold}.")
    
    mode = mode.lower()
    if mode not in ["experimental", "stable"]:
        raise ValueError(f"[ERROR] Find First Peak: Mode must be one of [experimental, stable]. Got {mode}.")
    
    signal = Array.validate(array=signal)
    
    axis = None if pool else 0
    
    # Apply normalisation with shifting and scaling
    if normalize:
        min_val = np.min(signal, axis=axis)
        max_val = np.max(signal, axis=axis)
        if np.any(max_val - min_val == 0):
            raise ValueError(f"[ERROR] Find First Peak: Max-Min cannot be zero for normalization. Got {max_val} - {min_val}.")
        signal = (signal - min_val) / (max_val - min_val)
        
    # Apply thresholding to flatten lower levels
    if threshold > 0:
        signal = np.array(signal)
        signal[signal < np.max(signal, axis=axis) * threshold] = 0
        
    # Search for peaks in the signal over each channel
    first_peaks = []
    for c in range(signal.shape[1]):
        # Detected peaks with experimental peak detection
        if mode == "experimental":
            peaks = find_peaks(signal)[c]
    
            # If no peaks found, choose first index as only peak
            if peaks.size > 0:
                first_peak = peaks[0]
            else:
                first_peak = 0
                
        # Stable peak detection approach
        elif mode == "stable":
            first_peak = 0
            # Search for first sample to reach 0
            while first_peak < signal.shape[0]:
                if signal[first_peak, c] == 0.0:
                    break
                first_peak += 1
                
            # Search for first sample to reach 1
            while first_peak < signal.shape[0]:
                if signal[first_peak, c] == 1.0:
                    break
                first_peak += 1
            
            # If no peaks found, choose first index as only peak
            if first_peak == signal.shape[0]:
                first_peak = 0
            
        first_peaks.append(first_peak)
        
    first_peaks = Array.validate(array=first_peaks)
        
    return first_peaks