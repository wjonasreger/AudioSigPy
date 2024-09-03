import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import imageio.v2 as imageio
from typing import Tuple, Union, Optional

from . import frequency as Frequency
from . import filter as Filter
from . import sound as Sound
from . import array as Array
from . import generators as Generators
from . import utils as Utils
from . import windows as Windows

# TODO
# begin building Talkbox class

class Processor():
    def __init__(
        self,
        node_count: int = 22,
        band_count: int = 22,
        lowcut: int = 20,
        highcut: int = 20000,
        scale: str = "octave"
    ) -> None:
        """
        Initialize the Processor with the specified parameters.

        Args:
            node_count (int): Number of nodes. Defaults to 22.
            band_count (int): Number of bands. Defaults to 22.
            lowcut (int): Low cut frequency. Defaults to 20.
            highcut (int): High cut frequency. Defaults to 20000.
            scale (str): Scale type. Defaults to 'octave'.

        Raises:
            ValueError: If band_count is less than node_count.
        """
        
        # Filterbank must have the same or more bands than there are nodes
        if band_count < node_count:
            raise ValueError(
                f"[ERROR] Initialization: Band count cannot be less than the node count. Got {band_count} < {node_count}"
            )
        
        # Set parameters for an audio processor
        self.set_node_count(node_count=node_count)
        self.set_band_count(band_count=band_count)
        self.set_lowcut(lowcut=lowcut)
        self.set_highcut(highcut=highcut)
        self.set_scale(scale=scale)
        
    def calc_bands(
        self,
        band_count: int,
        scale: str,
        ftype: str
    ) -> np.ndarray:
        """
        Calculate frequency bands based on the given scale and bandwidth selection.

        Arguments:
            band_count (int): Number of bands to calculate.
            scale (str): The scale used for band calculation. Must be one of ['erb', 'octave'].
            ftype (str): The type of bands to return. Must be one of ['bounds', 'centre', 'full'].

        Returns:
            np.ndarray: An array of calculated frequency bands.

        Raises:
            TypeError: If the scale or ftype is not a string.
            ValueError: If the scale or ftype is not among the valid options.
        """
        
        # Parameter checking and validation
        if not isinstance(scale, str):
            raise TypeError(f"[ERROR] Calculate bands: Scale must be a string. Got {type(scale).__name__}.")
        
        if not isinstance(ftype, str):
            raise TypeError(f"[ERROR] Calculate bands: Ftype must be a string. Got {type(ftype).__name__}.")
        
        scale = scale.lower()
        ftype = ftype.lower()
        
        if scale not in ["erb", "octave"]:
            raise ValueError(f"[ERROR] Calculate bands: Scale must be one of [erb, octave]. Got {scale}.")
        
        if ftype not in ["bounds", "centre", "full"]:
            raise ValueError(f"[ERROR] Calculate bands: Ftype must be one of [bounds, centre, full]. Got {ftype}.")
        
        lowcut, highcut = self.get_lowcut(), self.get_highcut()
        
        # ERB scale is ideal for Human hearing processors
        if scale == "erb":
            bands = Frequency.erb_centre_freqs(
                lowcut=lowcut,
                highcut=highcut,
                band_count=band_count * 2 + 1
            )
            bands = np.column_stack((
                bands[0:-2:2], bands[1:-1:2], bands[2::2]
            ))
            
        # Octave scale is default scaling for general processors
        if scale == "octave":
            steps = band_count / np.log2(highcut / lowcut)
            bands = Frequency.calc_step_bands(
                lowcut=lowcut, 
                highcut=highcut,
                steps=steps
            )
            self.set_band_count(band_count=bands.shape[0])
            
        # Flip array so higher frequency bands are at the top
        bands = np.flip(bands, axis=0)
        
        # Select what info is passed from the bandwidths array
        if ftype == "bounds":
            bands = np.delete(bands, 1, axis=1)
        if ftype == "centre":
            bands = bands[:, 1].reshape((-1, 1))
        if ftype == "full":
            bands = bands
            
        return bands
    
    def filterbank(
        self,
        signal: np.ndarray,
        sample_rate: Union[float, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies a filter bank to the input signal and computes RMS values for each band

        Args:
            signal (np.ndarray): The input signal to be filtered
            sample_rate (Union[float, int]): The sample rate of the input signal

        Returns:
            Tuple[np.ndarray, np.ndarray]: The filtered signal and corresponding RMS values
        """
        
        # Validate and check signal data
        signal = Array.validate(array=signal)
        
        if signal.shape[1] != 1:
            raise ValueError(f"[ERROR] Filterbank: Signal must have only one channel. Got {signal.shape[1]}.")
        
        # Calculate the frequency bands
        bands = self.calc_bands(
            band_count=self.get_band_count(),
            scale=self.get_scale(),
            ftype="bounds"
        )
        
        streams = []
        rms = []
        
        # Iteratively filter the signal over different bands to build filter bank
        for i in range(bands.shape[0]):
            lowcut, highcut = bands[i]
            
            filt_signal = Filter.butter_filter(
                signal=signal,
                cutoffs=[lowcut, highcut],
                sample_rate=sample_rate,
                band_type="bandpass",
                order=2,
                method="filtfilt"
            )
            
            streams.append(filt_signal)
            # Calculate the energy for each band stream
            rms.append(Sound.calc_rms(signal=filt_signal).flatten())
            
        streams = np.hstack(streams)
        rms = np.array(rms).reshape((-1, 1))
        
        return streams, rms
    
    def rectify(
        self,
        streams: np.ndarray,
        method: str
    ) -> np.ndarray:
        """
        Rectifies the input streams based on the specified method to convert from AC to DC.

        Args:
            streams (np.ndarray): The input signal streams
            method (str): The rectification method, either 'half' or 'full' (i.e., clip negatives or take absolute values)

        Raises:
            TypeError: If the method is not a string
            ValueError: If the method is not 'half' or 'full'.

        Returns:
            np.ndarray: The rectified DC streams
        """
        
        if not isinstance(method, str):
            raise TypeError(f"[ERROR] Rectify: Method must be a string. Got {type(method).__name__}.")
        
        method = method.lower()
        
        if method not in ["half", "full"]:
            raise ValueError(f"[ERROR] Rectify: Method must be one of [half, full]. Got {method}.")
        
        # Clip negative values to 0
        if method == "half":
            up_streams = []
            for i in range(streams.shape[1]):
                stream = streams[:, i]
                stream[stream < 0] = 0
                up_streams.append(stream.reshape((-1, 1)))
                
        # Get absolute values of streams
        if method == "full":
            up_streams = []
            for i in range(streams.shape[1]):
                stream = streams[:, i]
                stream = np.abs(stream)
                up_streams.append(stream.reshape((-1, 1)))
                
        up_streams = np.hstack(up_streams)
        
        return up_streams
    
    def envelopes(
        self,
        streams: np.ndarray,
        sample_rate: Union[float, int],
        cutoff: Union[float, int]
    ) -> np.ndarray:
        """
        Applies a lowpass Butterworth filter to each input stream to retrieve slow-moving variations in energy.

        Args:
            streams (np.ndarray): The input signal streams
            sample_rate (Union[float, int]): The sample rate of the input signal
            cutoff (Union[float, int]): The cutoff frequency for the lowpass filter

        Returns:
            np.ndarray: The extracted energy envelopes of the signal
        """
        
        up_streams = []
        
        # Get lowpass filters to extract slow-moving varying levels for envelopes
        for i in range(streams.shape[1]):
            stream = Filter.butter_filter(
                signal=streams[:, i],
                cutoffs=cutoff,
                sample_rate=sample_rate,
                band_type="lowpass",
                order=2,
                method="filtfilt"
            )
            up_streams.append(stream)
            
        up_streams = np.hstack(up_streams)
        
        return up_streams
    
    def normalize(
        self,
        streams: np.ndarray,
        rms: np.ndarray
    ) -> np.ndarray:
        """
        Normalize the audio streams based on the RMS values

        Args:
            streams (np.ndarray): The input signal streams
            rms (np.ndarray): The RMS values for each stream

        Returns:
            np.ndarray: The normalized streams
        """
        
        up_rms = []
        for i in range(streams.shape[1]):
            up_rms.append(
                Sound.calc_rms(signal=streams[:, i]).flatten()
            )
        
        up_rms = np.array(up_rms).reshape((-1, 1))
        
        # Calculate energy gain across streams for correction
        gains = np.mean(up_rms) / up_rms * rms
        
        # Correct streams with gains
        up_streams = []
        for i in range(streams.shape[1]):
            stream = streams[:, i] * gains[i][0]
            up_streams.append(stream.reshape((-1, 1)))
            
        up_streams = np.hstack(up_streams)
        
        return up_streams
    
    def steer(
        self,
        streams: np.ndarray
    ) -> np.ndarray:
        """
        Steers energy currents from origin streams (frequencies) to target streams (nodes)

        Args:
            streams (np.ndarray): The input streams

        Returns:
            np.ndarray: The steered energy currents across the target streams (nodes)
        """
        
        # Calculate bands for input and output streams
        freq_bands = self.calc_bands(
            band_count=self.get_band_count(),
            scale=self.get_scale(),
            ftype="bounds"
        )
        
        node_bands = self.calc_bands(
            band_count=self.get_node_count(),
            scale=self.get_scale(),
            ftype="bounds"
        )
        
        # Convert bands to log space
        freq_bands = np.log10(freq_bands)
        node_bands = np.log10(node_bands)
        
        # Compute bounds of different bands
        bounds = set(freq_bands.flatten()).union(set(node_bands.flatten()))
        bounds = np.sort(list(bounds))[::-1]
        bounds = np.column_stack((bounds[1:], bounds[:-1]))
        bounds = Array.validate(bounds)
        
        # Frequency interval lengths
        lengths = bounds[:, 1] - bounds[:, 0]
        lengths = Array.validate(array=lengths)
        
        # Complete the steering matrix
        origins = []
        targets = []
        for i in range(bounds.shape[0]):
            # Midpoint between interval bounds
            midpoint = np.mean(bounds[i])
            
            # Getting the origin index
            freq_idx = np.where(
                (freq_bands[:, 0] <= midpoint) & (freq_bands[:, 1] >= midpoint)
            )[0][0]
            
            # Getting the target index
            node_idx = np.where(
                (node_bands[:, 0] <= midpoint) & (node_bands[:, 1] >= midpoint)
            )[0][0]
            
            origins.append(freq_idx)
            targets.append(node_idx)
        
        origins = Array.validate(array=origins)
        targets = Array.validate(array=targets)
        
        steer_matrix = np.hstack([bounds, lengths, origins, targets])
        
        # Steer energy currents from origin streams to target streams (freq to nodes)
        up_streams = np.zeros((streams.shape[0], self.get_node_count()))
        weights = np.zeros((1, self.get_node_count()))
        
        for i in range(steer_matrix.shape[0]):
            origin = int(steer_matrix[i, 3])
            target = int(steer_matrix[i, 4])
            length = steer_matrix[i, 2]
            total = steer_matrix[steer_matrix[:, 3] == origin].sum()
            weight = length / total
            
            up_streams[:, target] += streams[:, origin] * weight
            weights[0, target] += weight
            
        # Normalize redirected streams by weights
        up_streams /= weights
        
        return up_streams
    
    def compress(
        self,
        frequencies: np.ndarray,
        lowcut: Union[float, int],
        highcut: Union[float, int]
    ) -> np.ndarray:
        """
        Compresses the frequency range by applying logarithmic scaling to specified bounds

        Args:
            frequencies (np.ndarray): The array of frequencies to be compressed
            lowcut (Union[float, int]): The lower cutoff frequency
            highcut (Union[float, int]): The upper cutoff frequency

        Returns:
            np.ndarray: The compressed array of frequencies
        """
        
        # Transform frequency data to log space
        log_freq = np.log10(frequencies)
        minfreq, maxfreq = np.min(log_freq), np.max(log_freq)
        lowcut, highcut = np.log10(lowcut), np.log10(highcut)
        
        # Scale frequency data to fit in target bounds
        alpha = (highcut - lowcut) / (maxfreq - minfreq)
        scaled_freqs = lowcut + alpha * (log_freq - minfreq)
        
        # Transform back to normal space
        frequencies = 10 ** scaled_freqs
        
        return frequencies
    
    def balance(
        self,
        streams: np.ndarray,
        delta: Union[float, int]
    ) -> np.ndarray:
        """
        Adjusts the levels of the input streams by applying a gamma distribution based on the calculated bands.

        Args:
            streams (np.ndarray): The inpust signal streams
            delta (Union[float, int]): Adjust balancing of bandwidth energy levels. 1 is most mild and >1 increases in extremity.

        Returns:
            np.ndarray: Adjusted streams after balancing
        """
        
        # Calculate bands for each node
        bands = self.calc_bands(
            band_count=self.get_node_count(),
            scale=self.get_scale(),
            ftype="centre"
        )
        
        # Calculate gamma curve for balance correction of energy levels across streams
        levels = scipy.stats.gamma.pdf(
            x=bands,
            a=1,
            loc=0,
            scale=2000
        )
        
        # Adjust gamma levels with delta parameter
        levels /= np.max(levels)
        levels += delta
        levels /= delta + 1
        
        # Apply gamma adjustments
        streams *= levels.flatten()
        
        return streams
            
    # Getter methods
    
    def get_node_count(self) -> int:
        """
        Returns:
            int: The number of nodes
        """
        
        return self.__node_count
    
    def get_band_count(self) -> int:
        """
        Returns:
            int: The number of bands
        """
        
        return self.__band_count
    
    def get_lowcut(self) -> Union[float, int]:
        """
        Returns:
            int: The lower frequency cutoff
        """
        
        return self.__lowcut
    
    def get_highcut(self) -> Union[float, int]:
        """
        Returns:
            int: The upper frequency cutoff
        """
        
        return self.__highcut
    
    def get_scale(self) -> str:
        """
        Returns:
            str: The frequency bandwidth scale
        """
        
        return self.__scale
    
    # Setter methods
    
    def set_node_count(self, node_count: int):
        """
        Args:
            node_count (int): The number of nodes to set.
        """
        
        self.__node_count = node_count
        
    def set_band_count(self, band_count: int):
        """
        Args:
            band_count (int): The number of bands to set.
        """
        
        self.__band_count = band_count
        
    def set_lowcut(self, lowcut: Union[float, int]):
        """
        Args:
            lowcut (float, int): The lower frequency cutoff
        """
        
        self.__lowcut = lowcut
        
    def set_highcut(self, highcut: Union[float, int]):
        """
        Args:
            highcut (float, int): The upper frequency cutoff
        """
        
        self.__highcut = highcut
        
    def set_scale(self, scale: str):
        """
        Args:
            scale (str): The frequency bandwidth scale
        """
        
        self.__scale = scale
        

class Cochlear(Processor):
    def __init__(
        self, 
        node_count: int = 22, 
        band_count: int = 22, 
        lowcut: int = 20,
        highcut: int = 20000,
        scale: str = "erb"
    ) -> None:
        super().__init__(node_count, band_count, lowcut, highcut, scale)
        
    def process(
        self,
        signal: np.ndarray,
        sample_rate: Union[float, int],
        rectify: str = "full",
        cutoff: Union[float, int] = 300,
        balance: Union[float, int] = 1
    ) -> np.ndarray:
        """
        Process the input signal through various stages of audio processing for a simple cochlear implant processor.

        Args:
            signal (np.ndarray): The input audio signal
            sample_rate (Union[float, int]): The sample rate of the signal
            rectify (str, optional): The method for rectifying the signal to DC. Defaults to "full".
            cutoff (Union[float, int], optional): Cutoff frequency for lowpass filter envelope extraction. Defaults to 300.
            balance (Union[float, int], optional): Balance factor of energy distribution. Defaults to 1.

        Returns:
            np.ndarray: The processed envelopes of the signal
        """
        
        signal = Array.validate(array=signal)
        up_signal = []
        
        # Iterate over each channel of signal
        for channel in range(signal.shape[1]):
        
            # Generate filterbank on input audio signal
            streams, energies = self.filterbank(signal=signal[:, channel], sample_rate=sample_rate)
            
            # Rectify amplitudes of node streams
            streams = self.rectify(streams=streams, method=rectify)
            
            # Extract slow-varying envelopes for each node stream
            streams = self.envelopes(streams=streams, sample_rate=sample_rate, cutoff=cutoff)
            
            # Normalize envelope energy levels to account for gain
            streams = self.normalize(streams=streams, rms=energies)
            
            # Steer current energy from frequency band partitions to node streams
            streams = self.steer(streams=streams)
            
            # Apply frequency envelopes to balance energy levels between streams
            streams = self.balance(streams=streams, delta=balance)
            
            up_signal.append(streams)
            
        up_signal = np.stack(up_signal, axis=1)
        
        return up_signal
    
    def signal(
        self,
        signal: np.ndarray,
        sample_rate: Union[float, int],
        target_spl: Union[float, int] = 70,
        frequency: Union[float, int] = 2000,
        carrier: str = "sine"
    ) -> np.ndarray:
        """
        Apply the signal envelope to a carrier signal and adjusting its SPL level.

        Args:
            signal (np.ndarray): The processed signal envelopes
            sample_rate (Union[float, int]): The sample rate of the signal
            target_spl (Union[float, int], optional): The target sound pressure level. Defaults to 70.
            frequency (Union[float, int], optional): The frequency of the carrier signal. Defaults to 2000.
            carrier (str, optional): The carrier signal type. Defaults to "sine".

        Returns:
            np.ndarray: The processed signal for cochlear implant electrode array
            
        Raises:
            TypeError: If carrier is not a string
            ValueError: If an invalid carrier type is provided
        """
        
        if not isinstance(carrier, str):
            raise TypeError(f"[ERROR] Signal: Carrier must be a string. Got {type(carrier).__name__}.")
        
        carrier = carrier.lower()
        if carrier not in ["sine", "pulse"]:
            raise ValueError(f"[ERROR] Signal: Carrier must be one of [sine, pulse]. Got {carrier}.")
        
        # Create carrier signal for envelopes
        if carrier == "sine":
            carrier_signal = Generators.sine(
                frequency=frequency,
                duration=signal.shape[0] / sample_rate,
                sample_rate=sample_rate
            )
        if carrier == "pulse":
            carrier_signal = Generators.pulse(
                frequency=frequency,
                duration=signal.shape[0] / sample_rate,
                sample_rate=sample_rate
            )
            
        # Apply envelopes on carrier signal
        carrier_signal = np.expand_dims(carrier_signal, axis=2)
        carrier_signal = np.repeat(carrier_signal, signal.shape[1], axis=1)
        carrier_signal = np.repeat(carrier_signal, signal.shape[2], axis=2)
        signal = signal * carrier_signal
        
        # Adjust energy levels of signal
        signal, _ = Sound.spl_transform(
            signal=signal, target_spl=target_spl
        )
        
        return signal
    
    def simulate(
        self, 
        signal: np.ndarray, 
        sample_rate: Union[float, int], 
        target_spl: Union[float, int] = 70, 
        carrier: str = "sine", 
        merge: bool = True, 
        lowcut: Optional[Union[float, int]] = None, 
        highcut: Optional[Union[float, int]] = None
    ) -> np.ndarray:
        """
        Apply the signal envelope to a carrier signal and adjusting its SPL level.

        Args:
            signal (np.ndarray): The processed signal envelopes
            sample_rate (Union[float, int]): The sample rate of the signal
            target_spl (Union[float, int], optional): The target sound pressure level. Defaults to 70.
            carrier (str, optional): The carrier signal type. Defaults to "sine".
            merge (bool): Option to sum the carrier energy levels in one stream per channel.
            lowcut (float, int): Lower frequency cutoff for frequency scale.
            highcut (float, int): Upper frequency cutoff for frequency scale.

        Returns:
            np.ndarray: The processed signal for cochlear implant electrode array
            
        Raises:
            TypeError: If carrier is not a string
            ValueError: If an invalid carrier type is provided
        """
        
        if not isinstance(merge, bool):
            raise TypeError(f"[ERROR] Simulate: Merge must be a boolean. Got {type(merge).__name__}.")
        
        if not isinstance(carrier, str):
            raise TypeError(f"[ERROR] Simulate: Carrier must be a string. Got {type(carrier).__name__}.")
        
        carrier = carrier.lower()
        if carrier not in ["sine", "pulse", "noise", "none"]:
            raise ValueError(f"[ERROR] Simulate: Carrier must be one of [sine, pulse]. Got {carrier}.")
        
        frequencies = np.ones_like(signal[:, 0, :])
        
        # Create frequency bands for carrier signal bands
        frequencies *= self.calc_bands(
            band_count=self.get_node_count(),
            scale=self.get_scale(),
            ftype="centre"
        ).flatten()
        
        if lowcut is None:
            lowcut = np.min(frequencies)
        if highcut is None:
            highcut = np.max(frequencies)
            
        # Compress frequencies to desired range
        frequencies = self.compress(
            frequencies=frequencies,
            lowcut=lowcut,
            highcut=highcut
        )
        
        # Create carrier signal - sine, pulse, noise, or none
        if carrier == "sine":
            carrier_signal = Generators.sine(
                frequency=frequencies,
                duration=signal.shape[0] / sample_rate,
                sample_rate=sample_rate
            )
            
        if carrier == "pulse":
            carrier_signal = Generators.pulse(
                frequency=frequencies,
                duration=signal.shape[0] / sample_rate,
                sample_rate=sample_rate
            )
            
        if carrier == "noise":
            frequencies = self.calc_bands(
                band_count=self.get_node_count(),
                scale=self.get_scale(),
                ftype="bounds"
            )
            
            carrier_signal = []
            for i in range(frequencies.shape[0]):
                lowcut, highcut = frequencies[i, :]
                
                carrier_source = Generators.white_noise(
                    duration=signal.shape[0] / sample_rate,
                    sample_rate=sample_rate
                )

                # Filter white noise for desired band
                carrier_stream = Filter.butter_filter(
                    signal=carrier_source,
                    cutoffs=[lowcut, highcut],
                    sample_rate=sample_rate,
                    band_type="bandpass",
                    order=2,
                    method="filtfilt"
                )
                carrier_signal.append(carrier_stream)
            carrier_signal = np.hstack(carrier_signal)
                
        if carrier == "none":
            carrier_signal = np.ones_like(frequencies)
            
        # Apply envelopes to carrier signal
        for channel in range(signal.shape[1]):
            signal[:, channel, :] *= carrier_signal
        
        # Adjust energy levels of signal
        signal, _ = Sound.spl_transform(
            signal=signal, target_spl=target_spl
        )
        
        # Sum energies across streams for each channel
        if merge:
            signal = signal.sum(axis=2)
        
        return signal
    
    def visualize(
        self,
        streams: np.ndarray,
        sample_rate: Union[float, int],
        frame_rate: Union[float, int],
        path: str,
        figsize: Tuple[int, int] = (16, 4)
    ) -> None:
        """
        Visualize the stream data for a given channel as a series of frames saved as a video

        Args:
            streams (np.ndarray): The input stream array
            sample_rate (Union[float, int]): The sample rate of the signal
            frame_rate (Union[float, int]): The video frame rate
            path (str): The pathway for the images and video
            figsize (Tuple[int, int], optional): The figure size of the plots. Defaults to (16, 4).

        Raises:
            ValueError: If the streams array is not the expected shape size (SAMPLE_COUNT, NODE_COUNT).
        """
        
        if len(streams.shape) > 2:
            raise ValueError(f"[ERROR] Visualize: Streams must be a stream array, not an entire signal array. Try signal[:, c, :] to select a channel for stream data.")
        
        # Plot signal across streams for a given channel and instance in time
        def plot(signal, amplitude=1):
            signal = np.flip(signal)
            
            # Generate window curve for energy amplitudes
            L = int(1000 / len(signal))
            if L % 2 == 0:
                L += 1
            peak = Windows.generalized_gaussian(length=L, mu=L/2, sigma=L/8)
            
            # Create energy amplitude curves for each node
            peaks = []
            for i in range(len(signal)):
                peaks.append(signal[i] * peak)
                
            peaks = np.hstack(peaks)
            
            # Set plot parameters
            plt.figure(figsize=figsize)
            xlim = (0, L-1 + np.ceil(L * (len(signal)-1) / 2))
            plt.hlines(0, xlim[0], xlim[1], colors="k", alpha=0.4)
            
            # Create grid lines
            delta = amplitude / 3
            plt.hlines((np.arange(2) + 1) * delta, xlim[0], xlim[1], colors="k", alpha=0.1)
            plt.hlines((np.arange(2) + 1) * -delta, xlim[0], xlim[1], colors="k", alpha=0.1)
            
            for stream in range(len(signal)):
                # Calculate x-axis locations for each node
                x = np.arange(L) + np.ceil(L * stream / 2)
                # Plot grid lines
                plt.vlines(np.ceil(L//2) + np.ceil(L * stream / 2), -amplitude, amplitude, colors="k", alpha=0.1)
                
                # Plot energy levels for each node
                plt.plot(x, peaks[:, stream], label=f'Node {stream}', alpha=0.3)
                plt.plot(x, -peaks[:, stream], label=f'Node {stream}', alpha=0.3)
                # Fill curves with shading
                plt.fill_between(x, peaks[:, stream], 0, alpha=0.3)
                plt.fill_between(x, -peaks[:, stream], 0, alpha=0.3)
                
            # Add plot labels
            plt.xlabel('Node (ID)', fontsize=14)
            plt.ylabel('Amplitude (V)', fontsize=14)
            plt.title('Modulated Signals for Each Node', fontsize=16)
            index = np.arange(len(signal))
            plt.xticks(np.ceil(index * L / 2) + np.floor(L/2), np.flip(index))
            
            # Set plot limit bounds
            plt.xlim(xlim)
            plt.ylim(-amplitude, amplitude)
            plt.tight_layout()
            
        # Get target indices for rate conversion from audio to video
        index = np.arange(streams.shape[0])
        index = index[index % (sample_rate // frame_rate) == 0][1:]
        
        # Get max peaks in each time interval of audio data to one video frame
        frame_streams = np.zeros((len(index), streams.shape[1]))
        for i in range(1, len(index)):
            frame_streams[i] = np.max(streams[index[i-1]:index[i]], axis=0)
            
        # Setup for plot image generation process
        Utils.manage_directory(directory_path=path, delete_if_exists=True)
        num_digits = len(str(len(index)))
        string_format = os.path.join(path, f"frame_{{i:0{num_digits}d}}.png")
            
        # Generate and save plots
        max_val = np.max(np.abs(frame_streams))
        for i in range(len(index)):
            p = plot(signal=frame_streams[i, :], amplitude=max_val)
            plt.savefig(string_format.format(i=i), dpi=120)
            plt.close()
            
        # Load all plot images
        image_files = [string_format.format(i=i) for i in np.arange(len(index))]
        images = [imageio.imread(img) for img in image_files]
        
        # Delete plot images from storage, convert on memory images to a video and save
        Utils.manage_directory(directory_path=path, delete_if_exists=True)
        imageio.mimwrite(os.path.join(path, "civis.mp4"), images, fps=frame_rate)
    
    
class Talkbox(Processor):
    def __init__(
        self, 
        band_count: int = 22, 
        lowcut: int = 20, 
        highcut: int = 20000, 
        scale: str = "octave"
    ) -> None:
        super().__init__(band_count, band_count, lowcut, highcut, scale)
        
    def process(
        self,
        signal: np.ndarray,
        sample_rate: Union[float, int],
        rectify: str = "full",
        cutoff: Union[float, int] = 300,
        balance: Union[float, int] = 1
    ) -> np.ndarray:
        """
        Process the input signal through various stages of audio processing for a simple talkbox processor.

        Args:
            signal (np.ndarray): The input audio signal
            sample_rate (Union[float, int]): The sample rate of the signal
            rectify (str, optional): The method for rectifying the signal to DC. Defaults to "full".
            cutoff (Union[float, int], optional): Cutoff frequency for lowpass filter envelope extraction. Defaults to 300.
            balance (Union[float, int], optional): Balance factor of energy distribution. Defaults to 1.

        Returns:
            np.ndarray: The processed envelopes of the signal
        """
        
        signal = Array.validate(array=signal)
        up_signal = []
        
        # Iterate over each channel of signal
        for channel in range(signal.shape[1]):
        
            # Generate filterbank on input audio signal
            streams, energies = self.filterbank(signal=signal[:, channel], sample_rate=sample_rate)
            
            # Rectify amplitudes of node streams
            streams = self.rectify(streams=streams, method=rectify)
            
            # Extract slow-varying envelopes for each node stream
            streams = self.envelopes(streams=streams, sample_rate=sample_rate, cutoff=cutoff)
            
            # Normalize envelope energy levels to account for gain
            streams = self.normalize(streams=streams, rms=energies)
            
            # Apply frequency envelopes to balance energy levels between streams
            streams = self.balance(streams=streams, delta=balance)
            
            up_signal.append(streams)
            
        up_signal = np.stack(up_signal, axis=1)
        
        return up_signal
    
    def signal(
        self,
        carrier: np.ndarray,
        modulator: np.ndarray,
        sample_rate: Union[float, int],
        target_spl: Union[float, int] = 70
    ) -> np.ndarray:
        """
        Generate a talkbox signal effect

        Args:
            carrier (np.ndarray): Carrier signal
            modulator (np.ndarray): Modulator signal from process
            sample_rate (Union[float, int]): Sample rate of both signals (must be the same)
            target_spl (Union[float, int], optional): Target sound pressure level. Defaults to 70.

        Raises:
            ValueError: If both signals do not have the same number of channels

        Returns:
            np.ndarray: The talkbox signal
        """
        
        sample_count = np.min([carrier.shape[0], modulator.shape[0]])
        carrier = Array.validate(array=carrier[:sample_count])
        modulator = Array.validate(array=modulator[:sample_count])
        
        if modulator.shape[1] != carrier.shape[1]:
            raise ValueError(f"[ERROR] Signal: Carrier and Modulator must have the same number of channels. Got {carrier.shape[1]} != {modulator.shape[1]}.")
        
        talkbox_signal = []
        for channel in range(carrier.shape[1]):
            streams, _ = self.filterbank(
                signal=carrier[:, channel], sample_rate=sample_rate
            )
            streams *= modulator[:, channel]
            talkbox_signal.append(streams)
            
        talkbox_signal = np.stack(talkbox_signal, axis=1).sum(axis=2)
        
        talkbox_signal, _ = Sound.spl_transform(
            signal=talkbox_signal, target_spl=target_spl
        )
        
        return talkbox_signal