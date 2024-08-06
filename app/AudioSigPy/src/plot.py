from typing import Tuple, Optional, List, Union
import numpy as np
import matplotlib.pyplot as plt
from . import array as Array


def validate_params(
    x: np.ndarray, 
    y: np.ndarray, 
    xlim: Optional[Tuple[float, float]] = None, 
    ylim: Optional[Tuple[float, float]] = None, 
    figsize: Optional[Tuple[float, float]] = None, 
    factor: float = 0.05
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    Validates and sets default parameters for plot limits and figure size.

    Args:
        x (np.ndarray): Array of x values.
        y (np.ndarray): Array of y values.
        xlim ((float, float), optional): Tuple specifying x-axis limits. Default is None.
        ylim ((float, float), optional): Tuple specifying y-axis limits. Default is None.
        figsize ((float, float), optional): Tuple specifying figure size. Default is None.
        factor (float, optional): Factor to adjust y-axis limits. Default is 0.05.

    Returns:
        Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]: Validated xlim, ylim, and figsize.
        
    Raises:
        TypeError: If inputs are not of expected types.
        ValueError: If limits are not valid.
    """
    
    # Validate input types
    if not isinstance(x, np.ndarray):
        raise TypeError(f"[ERROR] Validate parameters: x must be a numpy array. Got {type(x).__name__}.")
    if not isinstance(y, np.ndarray):
        raise TypeError(f"[ERROR] Validate parameters: y must be a numpy array. Got {type(y).__name__}.")
    if not isinstance(factor, (int, float)):
        raise TypeError(f"[ERROR] Validate parameters: factor must be an integer or float. Got {type(factor).__name__}.")
    if factor < 0 or factor >=1:
        raise ValueError(f"[ERROR] Validate parameters: Factor must be in [0, 1]. Got {factor}.")

    # Set default xlim and ylim if not provided
    if xlim is None:
        xlim = (np.min(x), np.max(x))
    if ylim is None:
        ylim = (np.min(y) - factor * np.abs(np.min(y)), np.max(y) + factor * np.abs(np.max(y)))

    # Set default figsize if not provided
    if figsize is None:
        figsize = (12, 6)

    # Helper function to validate limits
    def validate_limit(limit, name):
        if not isinstance(limit, (tuple, list)):
            raise ValueError(f"[ERROR] Validate parameters: {name} must be a tuple or list. Got {type(limit).__name__}.")
        if len(limit) != 2:
            raise ValueError(f"[ERROR] Validate parameters: {name} must have exactly two elements. Got {len(limit)}.")
        for value in limit:
            if not isinstance(value, (int, float)):
                raise ValueError(f"[ERROR] Validate parameters: Each value in {name} must be an integer or float. Got {type(value).__name__} for {value}.")
        return tuple(float(value) for value in limit)

    # Validate and convert limits and figsize
    xlim = validate_limit(xlim, 'xlim')
    ylim = validate_limit(ylim, 'ylim')
    figsize = validate_limit(figsize, 'figsize')

    # Validate xlim and ylim ranges
    if xlim[0] >= np.max(x):
        raise ValueError(f"[ERROR] Validate parameters: Minimum value must be less than the maximum value. Got {xlim[0]} >= {np.max(x)} for xlim.")
    if xlim[1] <= np.min(x):
        raise ValueError(f"[ERROR] Validate parameters: Maximum value must be greater than the minimum value. Got {xlim[1]} <= {np.min(x)} for xlim.")
    if ylim[0] >= np.max(y):
        raise ValueError(f"[ERROR] Validate parameters: Minimum value must be less than the maximum value. Got {ylim[0]} >= {np.max(y)} for ylim.")
    if ylim[1] <= np.min(y):
        raise ValueError(f"[ERROR] Validate parameters: Maximum value must be greater than the minimum value. Got {ylim[1]} <= {np.min(y)} for ylim.")

    # Return validated parameters
    return xlim, ylim, figsize


def plot_series(
    x: np.ndarray,
    y: np.ndarray,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    xlab: Optional[str] = None,
    ylab: Optional[Union[str, List[str]]] = None,
    title: Optional[str] = None,
    overlay: bool = False,
    vlines: Optional[List[Union[int, float]]] = None,
    hlines: Optional[List[Union[int, float]]] = None
) -> None:
    """
    Plots a time series with the given parameters.

    Arguments:
        x (np.ndarray): The x-axis data.
        y (np.ndarray): The y-axis data. Expected shape (SAMPLES, CHANNELS, GROUPS).
        xlim ((float, float), optional): Limits for the x-axis.
        ylim ((float, float), optional): Limits for the y-axis.
        figsize ((float, float), optional): Size of the figure.
        xlab (str, optional): Label for the x-axis.
        ylab (str or [str], optional): Label(s) for the y-axis.
        title (str, optional): Title of the plot.
        overlay (bool, optional): Whether to overlay channels on the same plot.

    Returns:
        None: Plot

    Raises:
        TypeError: If any input is of incorrect type.
        ValueError: If any input has an invalid value.
    """
    
    # Type checks
    if xlim and not (isinstance(xlim, tuple) and len(xlim) == 2):
        raise TypeError(f"[ERROR] Plot series: Xlim must be a tuple of two floats. Got {type(xlim).__name__}.")
    if ylim and not (isinstance(ylim, tuple) and len(ylim) == 2):
        raise TypeError(f"[ERROR] Plot series: Ylim must be a tuple of two floats. Got {type(ylim).__name__}.")
    if figsize and not (isinstance(figsize, tuple) and len(figsize) == 2):
        raise TypeError(f"[ERROR] Plot series: Figsize must be a tuple of two floats. Got {type(figsize).__name__}.")
    if xlab and not isinstance(xlab, str):
        raise TypeError(f"[ERROR] Plot series: Xlab must be a string. Got {type(xlab).__name__}.")
    if ylab and not (isinstance(ylab, (str, list))):
        raise TypeError(f"[ERROR] Plot series: Ylab must be a string or a list of strings. Got {type(ylab).__name__}.")
    if title and not isinstance(title, str):
        raise TypeError(f"[ERROR] Plot series: Title must be a string. Got {type(title).__name__}.")
    if not isinstance(overlay, bool):
        raise TypeError(f"[ERROR] Plot series: Overlay must be a boolean. Got {type(overlay).__name__}.")

    # validate arrays
    x = Array.validate(array=x)
    y = Array.validate(array=y)
    
    if vlines is not None:
        if not isinstance(vlines, (list, np.ndarray)):
            raise TypeError(f"[ERROR] Plot series: Vlines must be a list of numbers. Got {type(vlines).__name__}.")
        vlines = Array.validate(array=vlines, direction="none")
    
    if hlines is not None:
        if not isinstance(hlines, (list, np.ndarray)):
            raise TypeError(f"[ERROR] Plot series: Hlines must be a list of numbers. Got {type(hlines).__name__}.")
        hlines = Array.validate(array=hlines, direction="none")

    # assumes input y has shape (SAMPLES, CHANNELS, GROUPS)
    # y may have 2 or 3 dims
    if y.ndim not in [2, 3]:
        raise ValueError(f"[ERROR] Plot series: Y must have 2 or 3 dimensions. Got {y.ndim}.")
    
    if y.ndim == 3 and y.shape[2] == 1:
        shape = y.shape
        y = y.reshape(shape[:-1])
        
    # validate figsize first
    _, _, figsize = validate_params(x=x, y=y, xlim=None, ylim=None, figsize=figsize)
    
    # get channel count and group count
    channel_count = y.shape[1]
    if y.ndim == 2:
        group_count = 1
    else: 
        group_count = y.shape[2]
    
    # validate y labels for N-D cases (for N groups)
    if not isinstance(ylab, (list, tuple)):
        if isinstance(ylab, str) and group_count == 1:
            ylab = [ylab]
    if len(ylab) != group_count:
        raise ValueError(f"[ERROR] Plot series: Y axis label must have as many labels as groups. Got {len(ylab)} vs {group_count}.")
    
    # adjust figsize 
    if channel_count == 1 or overlay:
        figsize = (figsize[0], figsize[1] / 2)
    if not overlay:
        if channel_count > 1:
            figsize = (figsize[0], figsize[1] * channel_count / 2)
            
    if group_count > 1:
        figsize = (figsize[0], figsize[1])
    
    # create subplot object
    if overlay:
        figure, axes = plt.subplots(1, group_count, figsize=figsize)
    else:
        figure, axes = plt.subplots(channel_count, group_count, figsize=figsize)
    
    if title:
        figure.suptitle(title)
        figure.subplots_adjust(hspace=0.16)
    else:
        figure.subplots_adjust(hspace=0.16)
    
    # iterate across groups
    for group in range(group_count):
        if not isinstance(ylab[group], str):
            raise ValueError(f"[ERROR] Plot series: Y axis label must be a valid string. Got {type(ylab[group]).__name__}.")
        
        if group_count > 1:
            xlim = None
            ylim = None
        
        # overlay true, overlay channels on same plot
        if overlay:
            if y.ndim == 3:
                if group_count > 1:
                    ax = axes[group]
                else:
                    ax = axes[group]
                y_temp = y[:, :, group]
            else:
                ax = axes
                y_temp = y[:, :]
                
            xlim, ylim, figsize = validate_params(x=x, y=y_temp, xlim=xlim, ylim=ylim, figsize=figsize)
            
            ax.plot(x, y_temp)
            
            if vlines is not None:
                vlines = np.unique(vlines.flatten())
                for v in vlines:
                    ax.axvline(v, color="r", linestyle=":")
            
            if hlines is not None:
                hlines = np.unique(hlines.flatten())
                for h in hlines:
                    ax.axhline(h, color="r", linestyle=":")
            
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            
            if xlab:
                ax.set_xlabel(xlab)
            if ylab:
                ax.set_ylabel(ylab[group])
        else: # plot each channel separately
            for channel in range(channel_count):
                if y.ndim == 3:
                    if channel_count > 1 and group_count > 1:
                        ax = axes[channel, group]
                    elif channel_count == 1 and group_count > 1:
                        ax = axes[group]
                    elif channel_count > 1 and group_count == 1:
                        ax = axes[channel]
                    else:
                        ax = axes
                    y_temp = y[:, channel, group]
                else:
                    if channel_count > 1:
                        ax = axes[channel]
                    else:
                        ax = axes
                        
                    y_temp = y[:, channel]
                
                if channel != y.shape[1] - 1:
                    ax.get_xaxis().set_visible(False)
                    
                xlim, ylim, figsize = validate_params(x=x, y=y_temp, xlim=xlim, ylim=ylim, figsize=figsize)
                    
                ax.plot(x, y_temp)
            
                if vlines is not None:
                    for v in vlines[channel, :]:
                        ax.axvline(v, color="r", linestyle=":")
                
                if hlines is not None:
                    for h in hlines[channel, :]:
                        ax.axhline(h, color="r", linestyle=":")
                    
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                
                if channel_count > 1:
                    ax.set_title(f"Channel {channel}")
                
                if xlab:
                    ax.set_xlabel(xlab)
                if ylab:
                    ax.set_ylabel(ylab[group])
            
    plt.show()
        
        
def plot_image(
    array: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    xlab: Optional[str] = None,
    ylab: Optional[str] = None,
    title: Optional[str] = None,
    iter_axis: int = 2,
    color_label: str = "Sound Pressure (dB)",
    color_map: str = "inferno",
    y_bin_count: int = 10
) -> None:
    """
    Plots a 3D array with optional customization for axes limits, labels, and titles.

    Arguments:
        array (np.ndarray): A 3D numpy array with shape (SAMPLES, CHANNELS, GROUPS).
        x (np.ndarray): An array representing the x-axis values.
        y (np.ndarray): An array representing the y-axis values.
        xlim ((float, float), optional): Tuple specifying the x-axis limits.
        ylim ((float, float), optional): Tuple specifying the y-axis limits.
        figsize ((float, float), optional): Tuple specifying the figure size.
        xlab (str, optional): Label for the x-axis.
        ylab (str, optional): Label for the y-axis.
        title (str, optional): Title for the entire figure.
        iter_axis (int, optional): Axis along which to iterate and create subplots.
        color_label (str): Label for colorbar. Default to "Sound Pressure (dB)".
        color_map (str): Name of color map. Default to "inferno".
        y_bin_count (int): The number of bins for y-axis tick labels. Default to 10.
    
    Raises:
        ValueError: If the input array does not have 3 dimensions.
        TypeError: If any input argument is not of the expected type.
    """
    
    # Validate types of inputs
    if xlab and not isinstance(xlab, str):
        raise TypeError(f"[ERROR] Plot image: Xlab must be a string. Got {type(xlab).__name__}.")
    if ylab and not (isinstance(ylab, (str, list))):
        raise TypeError(f"[ERROR] Plot image: Ylab must be a string or a list of strings. Got {type(ylab).__name__}.")
    if title and not isinstance(title, str):
        raise TypeError(f"[ERROR] Plot image: Title must be a string. Got {type(title).__name__}.")
    if not isinstance(y_bin_count, int):
        raise TypeError(f"[ERROR] Plot image: Y-axis bin count must be an integer. Got {type(y_bin_count).__name__}.")
    if not isinstance(iter_axis, int):
        raise TypeError(f"[ERROR] Plot image: Iterative axis must be an integer. Got {type(iter_axis).__name__}.")
    
    if y_bin_count < 4 or y_bin_count > 20:
        raise ValueError(f"[ERROR] Plot image: Y-axis bin count must be in [4, 20]. Got {y_bin_count}.")
    
    # Assumes input array has shape (FREQUENCIES, TIMES, CHANNELS)
    # Validate arrays
    x = Array.validate(array=x)
    y = Array.validate(array=y)
    array = Array.validate(array=array, ref_axes=(0, 2))
    
    # Validate the shape of array
    if array.ndim != 3:
        raise ValueError(f"[ERROR] Plot image: Array must have 3 dimensions. Got {array.ndim}.")

    # Validate figsize first
    xlim, ylim, figsize = validate_params(x=x, y=y, xlim=xlim, ylim=ylim, figsize=figsize, factor=0)
    
    # Get channel count and adjust figsize
    channel_count = array.shape[iter_axis]
    figsize = (figsize[0], figsize[1] * channel_count / 2)
            
    # Create subplot object
    figure, axes = plt.subplots(channel_count, 1, figsize=figsize)
    if title:
        figure.suptitle(title)
    figure.subplots_adjust(hspace=0.15, top=0.9)
    cbar_ax = figure.add_axes([0.92, 0.15, 0.02, 0.7])

    # Plot image data for each channel
    for channel in range(channel_count):
        ax = axes[channel] if channel_count > 1 else axes
        
        # Make bottom image with x axis labels
        if channel != channel_count - 1:
            ax.get_xaxis().set_visible(False)
            
        # Save image plot
        img = ax.imshow(
            array[:, :, channel], 
            cmap=color_map, aspect="auto", vmin=-40, vmax=25, 
            extent=[xlim[0], xlim[1], y.shape[0], 0]
        )
        if channel_count > 1:
            ax.set_title(f"Channel {channel}")
            
        if xlab:
            ax.set_xlabel(xlab)
        if ylab:
            ax.set_ylabel(ylab)
        
        # Set tick labels for y axis
        tick_count = y.shape[0]
        tick_positions = np.linspace(0, tick_count-1, tick_count)
        ax.set_yticks(tick_positions, (10 * np.round(y.flatten() / 10)).astype(int))
        ax.locator_params(axis='y', nbins=y_bin_count)
        
    # Colorbar for decibel levels
    cbar = plt.colorbar(img, cax=cbar_ax)
    cbar.set_label(color_label)
        
    plt.show()