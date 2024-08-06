import numpy as np
from typing import List, Optional, Tuple, Union


# array manipulations

def unsqueeze(array: np.ndarray) -> np.ndarray:
    """
    Unsqueezes a 1-dimensional numpy array to a 2-dimensional array. 

    Arguments:
        array (np.ndarray): Input array to be unsqueezed.
        
    Returns:
        np.ndarray: Reshaped array with an additional dimension.
        
    Raises:
        TypeError: If the input is not a numpy array.
    """
    
    # Check if the input is a numpy array
    if not isinstance(array, np.ndarray):
        # Raise a TypeError if the input is not a numpy array
        raise TypeError(f"[ERROR] Unsqueeze: Array must be numpy.ndarray. Got {type(array).__name__}.")
        
    # Check if the input array is 1-dimensional
    if len(array.shape) == 1:
        # Reshape the array to add an additional dimension
        array = array.reshape((-1, 1))
    
    # Return the reshaped array
    return array


def align(
    array: np.ndarray, 
    direction: str = "down", 
    axis0: int = 0, 
    axis1: int = 1
) -> np.ndarray:
    """
    Align the given array based on the specified direction.

    Arguments:
        array (np.ndarray): The array to be aligned.
        direction (str, optional): The direction to align the array ("down", "up", "none").
        axis0 (int, optional): The first reference axis (y). Default to 0.
        axis1 (int, optional): The second reference axis (x). Default to 1.
        
    Returns:
        The aligned array.
        
    Raises:
        ValueError: If the direction is not "down", "up", or "none".
                    If a reference axis is not a valid dimension.
        TypeError: If direction is not a string.
                   If reference axis is not an integer.
    """
    
    if not isinstance(direction, str):
        raise TypeError(f"[ERROR] Align: Direction must be a string. Got {type(direction).__name__}.")
    
    # Convert direction to lowercase for consistency
    direction = direction.lower()
    
    # Validate the direction input
    if direction not in ["down", "up", "none"]:
        raise ValueError(f"[ERROR] Align: Alignment direction must be 'up', 'down', or 'none'. Got {direction}.")
    
    for ax in [axis0, axis1]:
        if not isinstance(ax, int):
            raise ValueError(f"[ERROR] Align: Reference axis must be an integer. Got {type(ax).__name__}.")
        if ax < 0 or ax >= array.ndim:
            raise ValueError(f"[ERROR] Align: Reference axis must be in [0, {array.ndim - 1}]. Got {ax}.")
    
    # Determine if the array needs to be transposed based on its shape and the direction
    pull = array.shape[axis0] < array.shape[axis1] and direction == "down"
    push = array.shape[axis0] > array.shape[axis1] and direction == "up"
    stay = direction == "none"

    # Transpose the array if necessary
    if (pull or push) and not stay:
        array = array.transpose()

    # Return the (potentially transposed) array
    return array


def validate(
    array: np.ndarray, 
    coerce: str = "float", 
    direction: str = "down", 
    ref_axes: Tuple[int, int] = (0, 1),
    numeric: bool = True
) -> np.ndarray:
    """
    Validate and transform the input array.

    Arguments:
        array (np.ndarray): The input array to validate and transform.
        coerce (str, optional): The data type to coerce the array elements to.
        direction (str, optional): The direction for aligning the array.
        ref_axes (tuple(int, int), optional): Reference axes for alignment operation.
        numeric (bool, optional): Indicates if array must be numeric.

    Returns:
        np.ndarray: The validated and transformed array.

    Raises:
        TypeError: If the input is not a numpy array or cannot be coerced to the specified type.
                   If the reference axes are not contained in a tuple or list.
        ValueError: If the input array contains non-numeric elements.
                    If the reference axes are not passed as 2 values.
        Other: If type coercion fails on the array.
    """
    
    # Validate numpy array type
    if not isinstance(array, np.ndarray):
        # Check if input is numeric value and convert to numpy array
        if isinstance(array, (int, float, complex)):
            array = np.array(array).reshape((1, 1))
        elif isinstance(array, (list, tuple)):
            array = np.array(array)
        else:
            raise TypeError(f"[ERROR] Validate: Array must be a numpy array. Got {type(array).__name__}.")

    # Validate numeric array
    if numeric:
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError("[ERROR] Validate: Array must have all numeric elements.")
    
    if not isinstance(ref_axes, (tuple, list)):
        raise TypeError(f"[ERROR] Validate: Reference axes must be a tuple or list. Got {type(ref_axes).__name__}.")
    if len(ref_axes) != 2:
        raise ValueError(f"[ERROR] Validate: Reference axes must have two values. Got {len(ref_axes)} values.")

    # Unsqueeze 1-dimensional array
    array = unsqueeze(array=array)

    # Align array
    array = align(array=array, direction=direction, axis0=ref_axes[0], axis1=ref_axes[1])

    # Set to floating types
    if coerce:
        try:
            array = array.astype(coerce)
        except Exception as e:
            raise Exception(f"[ERROR] Validate: Type coercion failed. Error: {str(e)}")

    return array


def downsample(array: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Downsample the given array along the specified axis.

    Arguments:
        array (np.ndarray): The input array.
        axis (int, optional): The axis along which to downsample. Default is 0.

    Returns:
        np.ndarray: The downsampled array.
        
    Raise:
        TypeError: If axis is not an integer.
        ValueError: If axis is an invalid dimension of array.
    """
    
    if not isinstance(axis, int):
        raise TypeError(f"[ERROR] Downsample: Axis must be an integer. Got {type(axis).__name__}.")
    
    if axis < 0 or axis >= array.ndim:
        raise ValueError(f"[ERROR] Downsample: Axis must be in [0, {array.ndim - 1}]. Got {axis}.")
    
    axis_size = array.shape[axis]
    
    # Generate indices for downsampling
    indices = linseries(start=0, end=axis_size, size=axis_size // 2, endpoint=False, coerce="int").flatten()
    # indices = align(array=indices, direction="up")
    
    # Generate slices for each axis
    slices = [slice(None)] * array.ndim
    slices[axis] = indices
    
    # downsample array
    array = array[tuple(slices)]
    
    return array


def linseries(
    start: float, 
    end: float, 
    size: Optional[int] = None, 
    endpoint: bool = False,
    coerce: str = "float"
) -> np.ndarray:
    """
    Create an array of evenly spaced values between start and end. Analogous to numpy.linspace method.

    Arguments:
        start (float): The starting value of the sequence.
        end (float): The end value of the sequence.
        size (int, optional): Number of values to generate. If None, it is computed as the absolute difference between start and end.
        endpoint (bool, optional): If True, the stop value is included in the sequence.
        coerce (str, optional): The type to coerce the elements of the array to. Must be either "float" or "int".

    Returns:
        np.ndarray: Array of evenly spaced values.

    Raises:
        TypeError: If the types of start, end, size, or endpoint are incorrect.
        ValueError: If start and end are both zero and size is None.
        ValueError: If coerce is not "float" or "int".
        ValueError: If size is not a positive integer.
    """
    
    # Type checking
    if not isinstance(start, (float, int)):
        raise TypeError(f"[ERROR] Linseries: Start must be a float or int. Got {type(start).__name__}.")
    if not isinstance(end, (float, int)):
        raise TypeError(f"[ERROR] Linseries: End must be a float or int. Got {type(end).__name__}.")
    if size is not None and not isinstance(size, int):
        raise TypeError(f"[ERROR] Linseries: Size must be an int. Got {type(size).__name__}.")
    if not isinstance(endpoint, bool):
        raise TypeError(f"[ERROR] Linseries: Endpoint must be a bool. Got {type(endpoint).__name__}.")
    if not isinstance(coerce, str):
        raise TypeError(f"[ERROR] Linseries: Coerce must be a str. Got {type(coerce).__name__}.")
    
    # Value checking
    if coerce not in ["float", "int"]:
        raise ValueError(f"[ERROR] Linseries: Coerce must be 'float' or 'int'. Got {coerce}.")
    if size is not None and size <= 0:
        raise ValueError(f"[ERROR] Linseries: Size must be a positive integer. Got {size}.")
    if size is None:
        if start == 0 and end == 0:
            raise ValueError(f"[ERROR] Linseries: Start and end cannot both be zero when size is None. Got {start} and {end}.")

    # Compute size if not provided
    if size is None:
        size = int(np.abs(start - end))

    # Generate the array of evenly spaced values
    array = np.linspace(start=start, stop=end, num=size, endpoint=endpoint)

    # Validate the generated array
    array = validate(array=array, coerce=coerce)

    return array


# helper functions for subset function

def val_subset_params(
    array: np.ndarray, 
    limits: Union[int, float, List, Tuple], 
    axes: Union[int, List[int]], 
    x: List[Union[List, np.ndarray]], 
    how: str, 
    method: str
):
    """
    Validate input parameters for the subset function prior to subset operation.

    Args:
        array (np.ndarray): The input array.
        limits (numeric or list of numerics): The subset limit values.
        axes (integer or list of integers): The subset axes.
        x (list or np.ndarray): The axis series values for value comparison method.
        how (str): The subset method of left, right, inner, or outer.
        method (str): The limit method of proportion, value, or index.

    Raises:
        TypeError: If any inputs have incorrect typing.
        ValueError: If any inputs have incorrect values.

    Returns:
        (np.ndarray, np.ndarray, np.ndarray, List[np.ndarray, str, str]): The validated subset parameters.
    """
    
    # Validate how and method
    if not isinstance(method, str):
        raise TypeError(f"[ERROR] Validate subset parameters: Method must be string. Got {type(method).__name__}.")
    if not isinstance(how, str):
        raise TypeError(f"[ERROR] Validate subset parameters: How must be string. Got {type(how).__name__}.")
    
    method = method.lower()
    how = how.lower()
    if method not in ["proportion", "value", "index"]:
        raise ValueError(f"[ERROR] Validate subset parameters: Method must be one of [proportion, value, index]. Got {method}.")
    if how not in ["left", "right", "inner", "outer"]:
        raise ValueError(f"[ERROR] Validate subset parameters: How must be one of [left, right, inner, outer]. Got {how}.")
    
    # Validate limits
    if not isinstance(limits, (list, np.ndarray)):
        if not isinstance(limits, (int, float, tuple)):
            raise TypeError(f"[ERROR] Validate subset parameters: Limit values must be numeric. Got {type(limits).__name__}.")
        if isinstance(limits, tuple) and not np.all([isinstance(i, (int, float)) for i in limits]):
            raise TypeError(f"[ERROR] Validate subset parameters: Limit values must be numeric. Got {type(limits).__name__} for {limits}.")
        limits = [limits]

    # Ensure limits orientation is consistent
    direction = "down"
    
    if len(limits) == 1:
        direction = "up"
    limits = validate(array=limits, direction=direction)
    
    # Ensure only 1 or 2 limit values per axis is given
    if limits.shape[1] not in [1, 2]:
        raise ValueError(f"[ERROR] Validate subset parameters: Limits must have 1 or 2 values per axis. Got {limits.shape[1]}.")
    
    # Validate axes
    if not isinstance(axes, (list, np.ndarray)):
        if not isinstance(axes, int):
            raise TypeError(f"[ERROR] Validate subset parameters: Axes values must be integer. Got {type(axes).__name__}.")
        axes = [axes]
        
    axes = validate(array=axes, coerce="int")
    
    # Ensure axes are given as 1-dim
    if axes.shape[1] != 1:
        raise ValueError(f"[ERROR] Validate subset parameters: Axis must be 1-dim. Got {axes.shape[1]}.")
    
    # Ensure axis values are within bounds of array dims and unique
    if np.any(axes < 0) or np.any(axes >= array.ndim):
        raise ValueError(f"[ERROR] Validate subset parameters: Axes are out of bounds. Got {axes} for array with {array.ndim}-dim.")
    if np.any(np.unique(axes, return_counts=True)[1][0] != 1):
        raise ValueError(f"[ERROR] Validate subset parameters: Axes values must be unique. Got {axes}.")
        
    # Ensure limits and axes have same size
    if limits.shape[0] != axes.shape[0]:
        raise ValueError(f"[ERROR] Validate subset parameters: Limits and axes must have the same number of elements. Got {limits.shape[0]} != {axes.shape[0]}.")
        
    # Ensure limits have appropriate number of values based on method
    if how in ["left", "right"] and limits.shape[1] != 1:
        raise ValueError(f"[ERROR] Validate subset parameters: Need one limit value per axis for {how} methods. Got {limits.shape[1]}.")
    if how in ["outer", "inner"] and limits.shape[1] != 2:
        raise ValueError(f"[ERROR] Validate subset parameters: Need two limit values per axis for {how} methods. Got {limits.shape[1]}.")
    
    # Validate x series if value method is chosen
    if method == "value":
        if not isinstance(x, list):
            if not isinstance(x, np.ndarray):
                raise TypeError(f"[ERROR] Validate subset parameters: X series values must be a numeric array or list. Got {type(x).__name__}.")
            x = [x]
        elif not isinstance(x[0], (list, np.ndarray)):
            x = [x]
        
        # Enforce x size == limit size
        if limits.shape[0] != len(x):
            raise ValueError(f"[ERROR] Validate subset parameters: Limits size and x size must be same. Got {limits.shape[0]} != {len(x)}.")
            
        # Validate X
        for i in range(len(x)):
            x[i] = validate(array=x[i])
            if x[i].shape[0] != array.shape[axes[i][0]]:
                raise ValueError(f"[ERROR] Validate subset parameters: X series array length must match input array dimension length. Got {x[i].shape[0]} != {array.shape[axes[i][0]]}.")
            if x[i].shape[1] != 1:
                raise ValueError(f"[ERROR] Validate subset parameters: X series array is too deep. Got {x[i].shape[1]} != 1 for series {i}.")
            
    return array, limits, axes, x, how, method


def val_limits(
    array: np.ndarray, 
    limits: np.ndarray, 
    axes: np.ndarray, 
    x: List[np.ndarray], 
    method: str
) -> np.ndarray:
    """
    Validate the limits based on limit method (proportion, value, index) and convert to index-like domain.

    Args:
        array (np.ndarray): The input array.
        limits (np.ndarray): The subset limit values.
        axes (np.ndarray): The subset axes.
        x (List[np.ndarray]): The axis series for value comparison.
        method (str): The subset method for limit values.

    Raises:
        ValueError: If input parameters have incorrect values.

    Returns:
        np.ndarray: The validated limit values converted to index-like domain (not integers yet).
    """
    
    # Get shape of array for selection
    shape = validate(array=array.shape, coerce="int")
    shape = shape[axes].squeeze().reshape((-1, 1)) - 1

    # Validate limits if proportions
    if method == "proportion":
        # Check if any values are out of [0, 1]
        if np.any(limits < 0) or np.any(limits > 1):
            raise ValueError(f"[ERROR] Validate subset limits: Limits for proportional subset must be in [0, 1]. Got {limits}.")
        limits = limits * shape
        
    # Validate limits if index
    if method == "index":
        limits = validate(array=limits, direction="none")
        
        # Get upper bounds of index support (N)
        shape = validate(array=array.shape, coerce="int")
        shape = shape[axes].squeeze().reshape((-1, 1))

        # Check if any values are out of [0, N]
        if np.any(limits < 0) or np.any(limits > shape-1):
            raise ValueError(f"[ERROR] Validate subset limits: Limits for index subset must be in [0, N] for each respective axis. Got {limits} for shape {shape}.")
        
    # Validate limits if value
    if method == "value":
        for i in range(len(x)):
            for j in range(limits.shape[1]):
                cover = np.where(x[i] < limits[i, j])[0]
                
                # Get index values from coverage
                if len(cover) == 0 and x[i][0] - x[i][-1] < 0:
                    cover = 0
                elif len(cover) == 0 and x[i][0] - x[i][-1] > 0:
                    cover = -1
                elif x[i][0] - x[i][-1] < 0:
                    cover = cover[-1]
                else:
                    cover = cover[0]
                
                limits[i, j] = cover
                
            # If an x series is reversed, flip index values
            if limits[i, 0] == -1 and limits.shape[0] > 1:
                limits[i, 0], limits[i, 1] = limits[i, 1], limits[i, 0]
                
    # Sort limits in increasing value
    if limits.shape[1] > 1:
        for i in range(limits.shape[0]):
            for j in range(limits.shape[1]):
                if limits[i, 1] != -1 and limits[i, 1] < limits[i, 0]:
                    limits[i, 0], limits[i, 1] = limits[i, 1], limits[i, 0]
    
    return limits


def val_index(
    limits: np.ndarray, 
    how: str, 
    method: str, 
    x: List[np.ndarray]
) -> np.ndarray:
    """
    Validate the limits based on the subset method (left, right, inner, outer) and convert completely to index domain.

    Args:
        limits (np.ndarray): The subset limit values.
        how (str): The subset method.
        method (str): The limit method.
        x (List[np.ndarray]): The axis series for value comparison method.

    Returns:
        np.ndarray: The validated and cleaned index value for subsetting operation.
    """
    
    # Apply adjustments if value method is used
    if method == "value":
        for i in range(len(x)):
            # Corrections if used with decreasing x series
            if x[i][0] - x[i][-1] > 0:
                if how == "left":
                    limits[i, :] -= 1
                if how == "right":
                    pass
                if how == "inner":
                    limits[i, 1] -= 1
                if how == "outer":
                    limits[i, 0] -= 1
            # Corrections if used with increasing x series
            else:
                if how == "left":
                    pass
                if how == "right":
                    if limits[i, 0] != -1:
                        limits[i, 0] += 1
                    if limits[i, 1] != -1:
                        limits[i, 1] += 1  
                if how == "inner":
                    if limits[i, 0] != -1:
                        limits[i, 0] += 1
                if how == "outer":
                    if limits[i, 1] != -1:
                        limits[i, 1] += 1                
            
            # Reset lower limit back to -1 if shifted to -2
            if limits[i, 0] == -2:
                limits[i, 0] = -1
            if limits.shape[1] > 1:
                if limits[i, 1] == -2:
                    limits[i, 1] = -1
                    
    # Apply floor/ceil to round limit values to nearest integer to cover desired regions
    if how == "left":
        limits = np.floor(limits)
    if how == "right":
        limits = np.ceil(limits)
    if how == "inner":
        limits[:, 0] = np.ceil(limits[:, 0])
        limits[:, 1] = np.floor(limits[:, 1])
    if how == "outer":
        limits[:, 0] = np.floor(limits[:, 0])
        limits[:, 1] = np.ceil(limits[:, 1])
        
    # Convert to integer for indexing
    limits = limits.astype(int)
        
    return limits


def subset(
    array: np.ndarray, 
    limits: Union[int, float, List, Tuple] = 0.5, 
    axes: Union[int, List] = 0, 
    how: str = "left", 
    method: str = "proportion", 
    x: Optional[Union[np.ndarray, List[np.ndarray]]] = None, 
    ref_axes: Tuple[int, int] = (0, 1), 
    coerce: str = "float"
) -> np.ndarray:
    """
    Apply subset operation on an array with given limits, axes, methods, and limit types.

    Args:
        array (np.ndarray): The input array.
        limits (int or float, optional): The limit values. Defaults to 0.5.
        axes (int, optional): The axes to be subsetted. Defaults to 0.
        how (str, optional): The direction of the subset. Defaults to "left".
        method (str, optional): The limit type. Defaults to "proportion".
        x (np.ndarray, optional): The value series for value comparison. Defaults to None.
        ref_axes (tuple, optional): The reference axes for array validation. Defaults to (0, 1).
        coerce (str, optional): The data type to coerce the array to. Defaults to "float".

    Returns:
        np.ndarray: The subsetted array.
    """
    
    # Validate input array
    array = validate(array=array, coerce=coerce, ref_axes=ref_axes)
    
    # Validate input parameters
    array, limits, axes, x, how, method = val_subset_params(array, limits, axes, x, how, method)
    
    # Validate limits and convert into indices for subsetting
    limits = val_limits(array, limits, axes, x, method)
    limits = val_index(limits, how, method, x)
    
    # Get final subset index arrays
    subset_indices = []
    for i in range(array.ndim):
        # Get subset indices for axes with subset selections
        if i in axes:
            idx = np.where(axes == i)[0][0]
            lim = limits[idx, :]
            # Upper bound to set -1 values to
            lim[lim == -1] = array.shape[idx] - 1
            
            # Geenrate subset indices from slices based on how methods
            if how == "left":
                indices = np.r_[slice(0, lim[0] + 1, 1)]
            if how == "right":
                indices = np.r_[slice(lim[0], array.shape[idx], 1)]
            if how == "inner":
                indices = np.r_[slice(lim[0], lim[1] + 1, 1)]
            if how == "outer":
                indices = []
                if lim[0] != 0:
                    indices.append(np.r_[slice(0, lim[0] + 1, 1)])
                if lim[1] != -1:
                    indices.append(np.r_[slice(lim[1], array.shape[idx], 1)])
                if len(indices) == 0:
                    indices.append(np.r_[slice(0, array.shape[idx], 1)])
                    
                indices = tuple(j for j in indices)
        # Get full subset if axes have no selection defined by user
        else:
            indices = slice(0, array.shape[i], 1)
            
        indices = np.r_[indices]
        subset_indices.append(indices)
        
    # Apply subset operation on input array
    array = array[np.ix_(*subset_indices)]
    
    return array