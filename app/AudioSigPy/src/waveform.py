from typing import Optional, Union, List, Tuple
import numpy as np

from . import utils as Utils
from . import wave as Wave
from . import array as Array
from . import plot as Plot
from . import transforms as Transforms
from . import frequency as Frequency
from . import spectral as Spectral
from . import generators as Generators


class Waveform():
    def __init__(self, signal_id: str = None):
        """
        Initializes an Waveform object with the specified signal ID.
        
        Args:
            signal_id (str, optional): Unique identifier for the audio signal.
        """
        
        self.set_signal_id(signal_id=signal_id)
        
        # Initialize parameters
        self.set_sample_rate(sample_rate=None)
        self.set_sample_count(sample_count=None)
        self.set_sample_width(sample_width=None)
        self.set_channel_count(channel_count=None)
        self.set_duration(duration=None)
        
        
    # Waveform methods
        
    def generate(
        self, 
        name: Union[str, List[str]], 
        duration: float, 
        args: Union[dict, List[dict]] = {}
    ) -> None:
        """
        Generate a waveform signal of simple waveform types (e.g., sine, square, sawtooth, etc.).

        Args:
            name (str): The waveform type name.
            duration (float): The duration of the signal in seconds.
            args (dict, optional): The additional parameters as needed based on the signal type. Defaults to {}.

        Raises:
            TypeError: If name is not a string or duration is not a float.
            ValueError: If parameters are out of bounds for mono or stereo audio generation.
        """
        
        # Ensure waveform parameters are in lists
        if isinstance(name, str):
            name = [name]
        if isinstance(args, dict):
            args = [args]
            
        if not isinstance(name, list):
            raise TypeError(f"[ERROR] Generate Wave: Name must be a string. Got {type(name).__name__}.")
        if not isinstance(args, list):
            raise TypeError(f"[ERROR] Generate Wave: Arguments must be a dictionary. Got {type(args).__name__}.")
        
        # Only support mono or stereo signal generation
        if len(name) > 2:
            raise ValueError(f"[ERROR] Generate Wave: Function only supports up to two channel generation. Got {len(name)} inputs for name.")
        if len(args) > 2:
            raise ValueError(f"[ERROR] Generate Wave: Function only supports up to two channel generation. Got {len(args)} inputs for args.")
        
        if len(name) == 0:
            raise ValueError(f"[ERROR] Generate Wave: Function needs to generate at least one channel. Got empty inputs for name.")
        
        # Validate parameters if generating stereo signal
        if len(name) == 2 or len(args) == 2:
            if len(name) == 1:
                name = [name[0]] * 2
            if len(args) == 1:
                args = [args[0]] * 2
                
            has_sample_rate = False
            if "sample_rate" in args[0].keys():
                has_sample_rate = args[0]["sample_rate"]
            if "sample_rate" in args[1].keys() and has_sample_rate:
                args[1]["sample_rate"] = has_sample_rate
            elif "sample_rate" in args[1].keys():
                args[0]["sample_rate"] = args[1]["sample_rate"]
                
        # Generate signal data
        signal = []
        signal_id = []
        for c in range(len(name)):
            # Validate sample rate
            sample_rate = args[c]["sample_rate"] if "sample_rate" in args[c].keys() else None
            sample_rate = Frequency.val_sample_rate(sample_rate=sample_rate)
            
            # Generate waveform
            gen_signal = Generators.choose_generator(duration=duration, name=name[c], args=args[c], mode="run")
            signal.append(gen_signal)
            
            # Create signal id
            signal_id.append(name[c].capitalize())
            if len(args[c]) > 0:
                params = ", ".join([k[:4] + "=" + str(v) for k, v in args[c].items()])
                signal_id[c] = signal_id[c] + " — " + params
                
        # Combine data
        signal = np.hstack(signal)
        signal_id = "; ".join(signal_id)
            
        # Set waveform parameters for object
        if self.get_signal_id() is None:
            self.set_signal_id(signal_id=signal_id)
        self.set_sample_count(sample_count=signal.shape[0])
        self.set_sample_rate(sample_rate=sample_rate)
        self.set_duration(duration=duration)
        self.set_sample_width(sample_width=2)
        self.set_channel_count(channel_count=len(name))
        self.set_signal_data(data=signal, sample_rate=sample_rate)
        
        
    def read(self, filename: str):
        """
        Reads a .wav file and extracts relevant information.
        
        Args:
            filename (str): Path to the .wav file.
        """
        
        # Open the .wav file in binary read mode
        self.__origin_filename = filename
        self.__wav_object = Wave.Wave(self.__origin_filename)
        self.__wav_object.open(mode="read")
        
        # Extract metadata
        self.set_sample_count(sample_count=self.__wav_object.get_sample_count())
        self.set_sample_rate(sample_rate=self.__wav_object.get_sample_rate())
        self.set_duration(duration=self.get_sample_count() / self.get_sample_rate())
        self.set_sample_width(sample_width=self.__wav_object.get_sample_width())
        self.set_channel_count(channel_count=self.__wav_object.get_channel_count())
        
        # Decode audio data and create time series array
        self.set_signal_data(data=self.__wav_object.decode(
            self.__wav_object.read_samples(self.__wav_object.get_sample_count())
            ), sample_rate=self.__wav_object.get_sample_rate())
        
        # Close the .wav file
        self.__wav_object.close()
        
        if self.get_signal_id() is None:
            self.set_signal_id(signal_id=self.__origin_filename)
        
        
    def write(self, filename: str):
        """
        Saves the generated waveform data to a .wav file.
        
        Args:
            filename (str): Name of the .wav file to save.
        """
        # Open the .wav file in binary write mode
        self.__target_filename = filename
        self.__wav_object = Wave.Wave(self.__target_filename)
        self.__wav_object.open(mode="write")
        
        # Set parameters for the .wav file
        self.__wav_object.set_sample_count(int(self.get_sample_count()))
        self.__wav_object.set_sample_rate(int(self.get_sample_rate()))
        self.__wav_object.set_sample_width(int(self.get_sample_width()))
        self.__wav_object.set_channel_count(int(self.get_channel_count()))
        
        # Check if the peak amplitude exceeds 1 and scale the data if needed
        signal_data = self.get_signal_data().copy()
        if np.max(np.abs(signal_data)) >= 1:
            signal_data = Transforms.scalar_transform(signal_data, 0.9)
            print("WARNING: The peak amplitude exceeds 1. The data will be scaled prior to writing to file.")
        
        # Write the waveform data to the .wav file
        self.__wav_object.write_samples(self.__wav_object.encode(signal_data))
        
        # Close the .wav file
        self.__wav_object.close()
        
        
    def frequency_series(self, size: Optional[int] = None, clip: bool = True) -> np.ndarray:
        """
        Generate a frequency series based on size and sample rate of the signal.

        Args:
            size (int, optional): The size of the frequency series. Defaults to None.
            clip (bool, optional): To clip the series in half. Defaults to True.

        Returns:
            np.ndarray: The frequency series.
        """
        
        if size is None:
            size = int(self.get_sample_count())
            
        # Generate and clip frequency series data
        frequency_series = Frequency.fft_frequencies(size=size, sample_rate=self.get_sample_rate())
        if clip:
            frequency_series = Array.subset(array=frequency_series, limits=[0.5], axes=[0])
        
        return frequency_series
    
    
    def time_series(self, size: Optional[int] = None) -> np.ndarray:
        """
        Generate a time series based on size and duration of the signal.

        Args:
            size (int, optional): The size of the time series. Defaults to None.

        Returns:
            np.ndarray: The time series.
        """
        
        if size is None:
            size = int(self.get_sample_count())
            
        # Generate time series data
        time_series = Array.linseries(start=0, end=self.get_duration(), size=size)
        
        return time_series
        

    # Plot methods

    def plot_waveform(
        self, 
        xlim: Optional[Union[List, Tuple]] = None, 
        ylim: Optional[Union[List, Tuple]] = None, 
        figsize: Optional[Union[List, Tuple]] = (16, 5), 
        overlay: bool = False
    ) -> None:
        """
        Plot the signal's waveform data.

        Args:
            xlim (tuple, optional): The x-axis limits for the plot. Defaults to None.
            ylim (tuple, optional): The y-axis limits for the plot. Defaults to None.
            figsize (tuple, optional): The figure size for the plot. Defaults to (16, 5).
            overlay (bool, optional): To overlay the channels on one plot. Defaults to False.
        """
        
        # Set the title for the plot
        title = "Waveform"
        if self.get_signal_id():
            title = f"\"{self.get_signal_id()}\" Waveform"
            
        # Plot the signal data
        Plot.plot_series(
            x=self.time_series(),
            y=self.get_signal_data(),
            xlim=xlim,
            ylim=ylim,
            figsize=figsize,
            xlab="Time (s)",
            ylab="Amplitude",
            title=title,
            overlay=overlay
        )
        
        
    def plot_spectra(
        self,
        spectrum: Optional[str] = None,
        xlim: Optional[Union[List, Tuple]] = None,
        ylim: Optional[Union[List, Tuple]] = None,
        figsize: Optional[Union[List, Tuple]] = (16, 5),
        clip: bool = True,
        log_freq: bool = True,
        decibel: bool = False,
        overlay: bool = False
    ) -> None:
        """
        Plot the signal's spectral data.

        Args:
            spectrum (str, optional): The spectrum to show in the plot. Defaults to None.
            xlim (tuple, optional): The x-axis limits for the plot. Defaults to None.
            ylim (tuple, optional): The y-axis limits for the plot. Defaults to None.
            figsize (tuple, optional): The figure size for the plot. Defaults to (16, 5).
            clip (bool, optional): To clip the frequency series. Defaults to True.
            log_freq (bool, optional): The apply log transform on frequency. Defaults to True.
            decibel (bool, optional): The apply log transform on the magnitude. Defaults to False.
            overlay (bool, optional): To overlay the channels on one plot. Defaults to False.

        Raises:
            TypeError: If log_freq or decibel is not boolean, or spectrum is not a string.
            ValueError: If spectrum is not in [magnitude, phase, None].
        """
        
        if not isinstance(log_freq, bool):
            raise TypeError(f"[ERROR] Plot Spectra: Log frequency must be a boolean. Got {type(log_freq).__name__}.")
        if not isinstance(decibel, bool):
            raise TypeError(f"[ERROR] Plot Spectra: Decibel must be a boolean. Got {type(decibel).__name__}.")
        
        # Select what spectra to show
        include_mag, include_phs = True, True
        if spectrum is not None:
            if not isinstance(spectrum, str):
                raise TypeError(f"[ERROR] Plot Spectra: Spectrum must be a string. Got {type(spectrum).__name__}.")
            spectrum = spectrum.lower()
            if spectrum not in ["magnitude", "phase"]:
                raise ValueError(f"[ERROR] Plot Spectra: Spectrum must be one of [magnitude, phase]. Got {spectrum}.")
            if spectrum == "magnitude":
                include_mag, include_phs = True, False
            elif spectrum == "phase":
                include_mag, include_phs = False, True
        
        # Compute the magnitude and phase spectra
        _, magnitude, phase = Spectral.calc_spectra(signal=self.get_signal_data(), clip=clip)
        # Compute the frequency axis series
        frequency_series = self.frequency_series(clip=clip)
        
        # Set axis labels
        frequency_label = "Frequency (Hz)"
        magnitude_label = "Magnitude"
        phase_label = "Phase (rad)"
        
        min_freq = np.min(frequency_series[frequency_series > 0])
        
        # Apply log transform on the spectra data
        if log_freq:
            frequency_label = "Frequency (log Hz)"
            frequency_series = Transforms.logn_transform(signal=frequency_series, coefficient=1, base=10, filter=min_freq)
            
        # Apply log transform if using decibels
        if decibel and include_mag:
            magnitude_label = "Magnitude (dB)"
            magnitude = Transforms.logn_transform(signal=magnitude, coefficient=20, base=10, filter=min_freq)
            
        # Inclusion logic for showing magnitude and/or phase spectra
        if include_mag and include_phs:
            data = np.stack([magnitude, phase], axis=-1)
            labels = [magnitude_label, phase_label]
        elif include_mag and not include_phs:
            data = magnitude
            labels = [magnitude_label]
        elif not include_mag and include_phs:
            data = phase
            labels = [phase_label]
            
        # Set the plot title
        title = "Spectra"
        if self.get_signal_id():
            title = f"\"{self.get_signal_id()}\" Spectra"
            
        # Plot the signal's spectral data
        Plot.plot_series(
            x=frequency_series,
            y=data,
            xlim=xlim,
            ylim=ylim,
            figsize=figsize,
            xlab=frequency_label,
            ylab=labels,
            title=title,
            overlay=overlay
        )
        
        
    def plot_spectrogram(
        self,
        alpha: float = 0.5,
        spread: float = 0.5,
        decibel: bool = True,
        xlim: Optional[Union[List, Tuple]] = None,
        ylim: Optional[Union[List, Tuple]] = None,
        figsize: Optional[Union[List, Tuple]] = (16, 8),
        channel: Optional[int] = None,
        clip: bool = True,
        window_size: float = 0.01,
        fft_size: int = 2048,
        overlap: float = 0.5,
        window: str = "hamming",
        color_map: str = "inferno"
    ) -> None:
        """
        Plot the signal's spectrogram data.

        Args:
            alpha (float, optional): Scaling parameter for frequency transform. Defaults to 0.5.
            spread (float, optional): Scaling parmaeter for y-axis labelling. Defaults to 0.5.
            decibel (bool, optional): Apply log transform on magnitude spectrum. Defaults to True.
            xlim (tuple, optional): The x-axis limits for the plot. Defaults to None.
            ylim (tuple, optional): The y-axis limits for the plot. Defaults to None.
            figsize (tuple, optional): The figure size for the plot. Defaults to (16, 5).
            channel (int, optional): Channels to view in the plot. Defaults to None.
            clip (bool, optional): To clip the data. Defaults to True.
            
        Raises:
            TypeError: If decibel is not boolean, spread is not float, or channel is not integer.
            ValueError: If spread is not in [0, 1] or channel is not in [0, N-1] for N channels.
        """
        
        if not isinstance(decibel, bool):
            raise TypeError(f"[ERROR] Plot Spectrogram: Decibel must be a boolean. Got {type(decibel).__name__}.")
        if not isinstance(spread, float):
            raise TypeError(f"[ERROR] Plot Spectrogram: Spread must be a float. Got {type(spread).__name__}.")
        
        if spread < 0 or spread > 1:
            raise ValueError(f"[ERROR] Plot Spectrogram: Spread must be in [0, 1]. Got {spread}.")
        
        # Calculate the spectrogram matrix of the signal
        spectrogram = Spectral.calc_spectrogram(
            signal=self.get_signal_data(), 
            sample_rate=self.get_sample_rate(), 
            clip=clip,
            window_size=window_size,
            fft_size=fft_size,
            overlap=overlap,
            window=window
        )
        
        # Validate the selected channels for viewing
        if not isinstance(channel, (int, type(None))):
            raise TypeError(f"[ERROR] Plot Spectrogram: Channel must be an integer or None. Got {type(channel).__name__}.")
        if channel is None:
            channels = Array.linseries(start=0, end=spectrogram.shape[2], size=spectrogram.shape[2], endpoint=False, coerce="int")
        elif channel < 0 or channel >= self.get_channel_count():
            raise ValueError(f"[ERROR] Plot Spectrogram: Channel must be in [0, {self.get_channel_count()-1}]. Got {channel}.")
        else:
            channels = Array.validate(array=channel, coerce="int")
        
        # Generate the frequency and time series
        frequency_series = self.frequency_series(clip=clip, size=spectrogram.shape[0] * 2)
        frequency_series = np.flip(frequency_series, axis=0)
        time_series = self.time_series(size=spectrogram.shape[1])
        
        # Get minimum frequency value for log thresholding
        min_mag = np.min(spectrogram[spectrogram > 0])
        
        # Apply log transformation on magnitudes
        if decibel:
            spectrogram = Transforms.logn_transform(signal=spectrogram, coefficient=20, base=10, filter=min_mag, ref_axes=(0, 2))
            
        # Pre-validate plotting parameters
        xlim, ylim, figsize = Plot.validate_params(x=time_series, y=frequency_series, xlim=xlim, ylim=ylim, figsize=figsize, factor=0)
        
        # Adjust limit values to account for some deviation from validation and subsetting operations
        delta = (np.abs(frequency_series[0] - frequency_series[1]) / 2)[0]
        ylim = (np.max([0.01, ylim[0] - delta]), np.min([self.get_nyquist_limit(), ylim[1] + delta]))
        delta = (np.abs(time_series[0] - time_series[1]) / 2)[0]
        xlim = (np.max([0.01, xlim[0] - delta]), np.min([self.get_duration(), xlim[1] + delta]))
        
        # Subset the spectrogram matrix and frequency series prior to plotting
        spectrogram = Array.subset(
            array=spectrogram,
            limits=[xlim, ylim],
            axes=[1, 0],
            x=[time_series, frequency_series],
            how="inner",
            method="value",
            ref_axes=(0, 2)
        )
        
        frequency_series = Array.subset(
            array=frequency_series,
            limits=[ylim],
            axes=[0],
            x=[frequency_series],
            how="inner",
            method="value",
            ref_axes=(0, 1)
        )
        
        # Generate and apply the box-cox scaled index series for image plotting
        scaled_index = Transforms.index_transform(size=spectrogram.shape[0], alpha=alpha).reshape((-1, ))
        spectrogram = spectrogram[scaled_index, :, :]
        frequency_series = frequency_series[scaled_index]
        
        # Subset spectrogram over channel selection
        spectrogram = spectrogram[:, :, channels.flatten()]
        
        # Set the plot title
        title = "Spectrogram"
        if self.get_signal_id():
            title = f"\"{self.get_signal_id()}\" Spectrogram"
            
        # Adjust spread parameter
        bin_count = int(4 + 16 * (1 - spread))
        
        # Plot the spectrogram matrix as an image plot
        Plot.plot_image(
            array=spectrogram,
            x=time_series,
            y=frequency_series,
            xlim=xlim, 
            ylim=ylim,
            figsize=figsize,
            xlab="Time (s)",
            ylab="Frequency (Hz)",
            title=title,
            y_bin_count=bin_count,
            color_map=color_map
        )
        
        
    # Setter methods
    
    def set_signal_id(self, signal_id: Optional[str] = None) -> None:
        """
        Updates the signal ID.
        
        Args:
            signal_id (str): The ID of the signal. Default to None.
            
        Raises:
            TypeError: If the signal id is not a string.
        """
        
        if not isinstance(signal_id, (str, type(None))):
            raise TypeError(f"[ERROR] Set Signal ID: Signal ID must be a string. Got {type(signal_id).__name__}.")
        
        self.__signal_id = signal_id
        
    
    def set_signal_data(
        self, 
        data: np.ndarray, 
        sample_rate: Optional[Union[int, float]] = None, 
        duration: Optional[Union[int, float]] = None
    ) -> None:
        """
        Updates the signal data array after generating or reading an external waveform array.
        
        Args:
            data (np.ndarray): The input signal data.
            sample_rate (int or float, optional): The sample rate for the signal. Default to None.
            duration (int or float, optional): The duration for the signal. Default to None.
            
        Raises:
            TypeError: If sample rate or duration is not int or float.
            ValueError: If data is not given or either sample rate or duration is not given.
        """
        
        if not isinstance(sample_rate, (int, float, type(None))):
            raise TypeError(f"[ERROR] Set Signal Data: Sample rate must be an integer or float. Got {type(sample_rate).__name__}.")
        if not isinstance(duration, (int, float, type(None))):
            raise TypeError(f"[ERROR] Set Signal Data: Duration must be an integer or float. Got {type(duration).__name__}.")
        
        if sample_rate is None and duration is None:
            raise ValueError(f"[ERROR] Set Signal Data: Sample rate or duration must be specified for new signal data.")
        
        if data is None:
            raise ValueError(f"[ERROR] Set Signal Data: Data must be non-empty.")
        
        self.__data = Array.validate(array=data)
        self.set_sample_count(sample_count=self.__data.shape[0])
        self.set_channel_count(channel_count=self.__data.shape[1])
        
        if sample_rate is not None:
            self.set_sample_rate(sample_rate=sample_rate)
            self.set_duration(duration=self.get_sample_count() / sample_rate)
            
        elif duration is not None:
            self.set_duration(duration=duration)
            self.set_sample_rate(sample_rate=int(self.get_sample_count() / duration))
    
    
    def set_duration(self, duration: Union[int, float]):
        """
        Updates the signal duration in seconds.
        
        Args:
            duration (int or float): The duration of the signal in seconds.
            
        Raises:
            TypeError: If duration is not int or float.
        """
        
        if not isinstance(duration, (int, float, type(None))):
            raise TypeError(f"[ERROR] Set Duration: Duration must be an integer or float. Got {type(duration).__name__}.")
        
        self.__duration = duration
    
    
    def set_sample_rate(self, sample_rate: Union[int, float]):
        """
        Updates the signal sampling rate in bits per second.
        
        Args:
            sample_rate (int or float): The sample rate of the signal in Hz per second.
            
        Raises:
            TypeError: If sample rate is not int or float.
        """
        
        if not isinstance(sample_rate, (int, float, type(None))):
            raise TypeError(f"[ERROR] Set Sample Rate: Sample rate must be an integer or float. Got {type(sample_rate).__name__}.")
        
        self.__sample_rate = sample_rate
    
    
    def set_sample_count(self, sample_count: Union[int, float]):
        """
        Updates the signal bit count.
        
        Args:
            sample_count (int or float): The sample count of the signal.
            
        Raises:
            TypeError: If sample count is not int or float.
        """
        
        if not isinstance(sample_count, (int, float, type(None))):
            raise TypeError(f"[ERROR] Set Sample Count: Sample count must be an integer or float. Got {type(sample_count).__name__}.")
        
        self.__sample_count = sample_count
    
    
    def set_sample_width(self, sample_width: Union[int, float] = 4):
        """
        Updates the signal sample width (bits per sample).
        
        Args:
            sample width (int or float): The sample width of the signal. Default is 16.
            
        Raises:
            TypeError: If sample width is not int or float.
        """
        
        if not isinstance(sample_width, (int, float, type(None))):
            raise TypeError(f"[ERROR] Set Sample Width: Sample width must be an integer or float. Got {type(sample_width).__name__}.")
        
        self.__sample_width = sample_width
    
    
    def set_channel_count(self, channel_count: Union[int, float] = 1):
        """
        Updates the signal channel count.
        
        Args:
            channel count (int or float): The channel count of the signal. Default is 1.
            
        Raises:
            TypeError: If channel count is not int or float.
        """
        
        if not isinstance(channel_count, (int, float, type(None))):
            raise TypeError(f"[ERROR] Set Channel Count: Channel count must be an integer or float. Got {type(channel_count).__name__}.")
        
        self.__channel_count = channel_count
        
        
    # Getter methods
        
    def get_signal_id(self) -> str:
        """
        Returns the signal ID.
        
        Raises:
            ValueError: If signal id is not set.
        """
        
        if self.__signal_id is None:
            raise ValueError(f"[ERROR] Get Signal ID: Signal ID not set.")
        
        return self.__signal_id
        
    
    def get_signal_data(self) -> np.ndarray:
        """
        Returns the signal data array after generating or reading a waveform.
        
        Raises:
            ValueError: If data is not set.
        """
        
        if self.__data is None:
            raise ValueError(f"[ERROR] Get Data: Data not set.")
        
        return self.__data
    
    
    def get_duration(self) -> float:
        """
        Returns the signal duration in seconds after generating or reading a waveform.
        
        Raises:
            ValueError: If duration is not set.
        """
        
        if self.__duration is None:
            raise ValueError(f"[ERROR] Get Duration: Duration not set.")
        
        return self.__duration
    
    
    def get_sample_rate(self) -> int:
        """
        Returns the signal sampling rate in bits per second after generating or reading a waveform.
        
        Raises:
            ValueError: If sample rate is not set.
        """
        
        if self.__sample_rate is None:
            raise ValueError(f"[ERROR] Get Sample Rate: Sample rate not set.")
        
        return self.__sample_rate
    
    
    def get_nyquist_limit(self) -> int:
        """
        Returns the signal nyquist limit after generating or reading a waveform.
        
        Raises:
            ValueError: If nyquist limit is not set.
        """
        
        if self.get_sample_rate() is None:
            raise ValueError(f"[ERROR] Get Nyquist Limit: Sample rate not set.")
        
        return self.__sample_rate // 2
    
    
    def get_sample_count(self) -> int:
        """
        Returns the signal bit count after generating or reading a waveform.
        
        Raises:
            ValueError: If sample coount is not set.
        """
        
        if self.__sample_count is None:
            raise ValueError(f"[ERROR] Get Sample Count: Sample count not set.")
        
        return self.__sample_count
    
    
    def get_sample_width(self) -> int:
        """
        Returns the signal sample width (bits per sample) after generating or reading a waveform.
        
        Raises:
            ValueError: If sample width is not set.
        """
        
        if self.__sample_width is None:
            self.set_sample_width()
        
        return self.__sample_width
    
    
    def get_channel_count(self) -> int:
        """
        Returns the signal channel count after generating or reading a waveform.
        
        Raises:
            ValueError: If channel count is not set.
        """
        
        if self.__channel_count is None:
            raise ValueError(f"[ERROR] Get Channel Count: Channel count not set.")
        
        return self.__channel_count
        
    
    # Representation methods
        
    def get_parameters(self) -> dict:
        """
        Return the parameters of the Waveform object.

        Returns:
            dict: Dictionary of parameters and their values.
            
        Raises:
            ValueError: If not all parameters are set.
        """
        
        parameters = {
            "id": self.get_signal_id(),
            "duration": self.get_duration(),
            "sample_rate": self.get_sample_rate(),
            "nyquist_limit": self.get_nyquist_limit(),
            "sample_count": self.get_sample_count(),
            "sample_width": self.get_sample_width(),
            "channel_count": self.get_channel_count()
        }
        
        if not all(parameters):
            raise ValueError("[ERROR] Get parameters: Not all parameters are set.")
        
        return parameters
    
        
    def __repr__(self) -> str:
        """
        Return a string representation of the Waveform object for debugging.

        Returns:
            str: String representation of the object.
        """
        
        developer_repr = Utils.base_repr(self)
        
        return developer_repr


    def __str__(self) -> str:
        """
        Return a string representation of the Waveform object for the end user.

        Returns:
            str: User-friendly string representation of the object.
        """
        
        user_repr = Utils.base_str(self)
        
        return user_repr