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
# Add error handling
# Add comments/docs
# review variable names for clarity/consistency
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
        
        if band_count < node_count:
            raise ValueError()
        
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
        
        lowcut, highcut = self.get_lowcut(), self.get_highcut()
        
        if scale == "erb":
            bands = Frequency.erb_centre_freqs(
                lowcut=lowcut,
                highcut=highcut,
                band_count=band_count * 2 + 1
            )
            bands = np.column_stack((
                bands[0:-2:2], bands[1:-1:2], bands[2::2]
            ))
            
        if scale == "octave":
            steps = band_count / np.log2(highcut / lowcut)
            bands = Frequency.calc_step_bands(
                lowcut=lowcut, 
                highcut=highcut,
                steps=steps
            )
            self.set_band_count(band_count=bands.shape[0])
            
        bands = np.flip(bands, axis=0)
        
        if ftype == "bounds":
            bands = np.delete(bands, 1, axis=1)
        if ftype == "centre":
            bands = bands[:, 1].reshape((-1, 1))
            
        return bands
    
    def filterbank(
        self,
        signal: np.ndarray,
        sample_rate: Union[float, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        bands = self.calc_bands(
            band_count=self.get_band_count(),
            scale=self.get_scale(),
            ftype="bounds"
        )
        
        channels = []
        rms = []
        
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
            
            channels.append(filt_signal)
            rms.append(Sound.calc_rms(signal=filt_signal).flatten())
            
        channels = np.hstack(channels)
        rms = np.array(rms).reshape((-1, 1))
        
        return channels, rms
    
    def rectify(
        self,
        channels: np.ndarray,
        method: str
    ) -> np.ndarray:
        if method == "half":
            up_channels = []
            for i in range(channels.shape[1]):
                channel = channels[:, i]
                channel[channel < 0] = 0
                up_channels.append(channel.reshape((-1, 1)))
                
        if method == "full":
            up_channels = []
            for i in range(channels.shape[1]):
                channel = channels[:, i]
                channel = np.abs(channel)
                up_channels.append(channel.reshape((-1, 1)))
                
        up_channels = np.hstack(up_channels)
        
        return up_channels
    
    def envelopes(
        self,
        channels: np.ndarray,
        sample_rate: Union[float, int],
        cutoff: Union[float, int]
    ) -> np.ndarray:
        env_channels = []
        
        for i in range(channels.shape[1]):
            filt_channel = Filter.butter_filter(
                signal=channels[:, i],
                cutoffs=cutoff,
                sample_rate=sample_rate,
                band_type="lowpass",
                order=2,
                method="filtfilt"
            )
            env_channels.append(filt_channel)
            
        env_channels = np.hstack(env_channels)
        
        return env_channels
    
    def normalize(
        self,
        channels: np.ndarray,
        rms: np.ndarray
    ) -> np.ndarray:
        
        env_rms = []
        for i in range(channels.shape[1]):
            env_rms.append(
                Sound.calc_rms(signal=channels[:, i]).flatten()
            )
        
        env_rms = np.array(env_rms).reshape((-1, 1))
        
        gains = np.mean(env_rms) / env_rms * rms
        
        norm_channels = []
        for i in range(channels.shape[1]):
            norm_channel = channels[:, i] * gains[i][0]
            norm_channels.append(norm_channel.reshape((-1, 1)))
            
        norm_channels = np.hstack(norm_channels)
        
        return norm_channels
    
    def steer(
        self,
        channels: np.ndarray
    ) -> np.ndarray:
        # Calculate bands for input and output channels
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
        
        lengths = bounds[:, 1] - bounds[:, 0]
        lengths = Array.validate(array=lengths)
        
        # Complete the steering matrix
        origins = []
        targets = []
        for i in range(bounds.shape[0]):
            midpoint = np.mean(bounds[i])
            
            freq_idx = np.where(
                (freq_bands[:, 0] <= midpoint) & (freq_bands[:, 1] >= midpoint)
            )[0][0]
            
            node_idx = np.where(
                (node_bands[:, 0] <= midpoint) & (node_bands[:, 1] >= midpoint)
            )[0][0]
            
            origins.append(freq_idx)
            targets.append(node_idx)
        
        origins = Array.validate(array=origins)
        targets = Array.validate(array=targets)
        
        steer_matrix = np.hstack([bounds, lengths, origins, targets])
        
        # Steer energy currents from origin channels to target channels (freq to nodes)
        currents = np.zeros((channels.shape[0], self.get_node_count()))
        weights = np.zeros((1, self.get_node_count()))
        
        for i in range(steer_matrix.shape[0]):
            origin = int(steer_matrix[i, 3])
            target = int(steer_matrix[i, 4])
            length = steer_matrix[i, 2]
            total = steer_matrix[steer_matrix[:, 3] == origin].sum()
            weight = length / total
            
            currents[:, target] += channels[:, origin] * weight
            weights[0, target] += weight
            
        currents /= weights
        
        return currents
    
    def compress(
        self,
        frequencies: np.ndarray,
        lowcut: Union[float, int],
        highcut: Union[float, int]
    ) -> np.ndarray:
        
        log_freq = np.log10(frequencies)
        minfreq, maxfreq = np.min(log_freq), np.max(log_freq)
        lowcut, highcut = np.log10(lowcut), np.log10(highcut)
        
        alpha = (highcut - lowcut) / (maxfreq - minfreq)
        scaled_freqs = lowcut + alpha * (log_freq - minfreq)
        
        frequencies = 10 ** scaled_freqs
        
        return frequencies
    
    def balance(
        self,
        channels: np.ndarray,
        delta: Union[float, int]
    ) -> np.ndarray:
        bands = self.calc_bands(
            band_count=self.get_node_count(),
            scale=self.get_scale(),
            ftype="centre"
        )
        levels = scipy.stats.gamma.pdf(
            x=bands,
            a=1,
            loc=0,
            scale=2000
        )
        
        levels /= np.max(levels)
        levels += delta
        levels /= delta + 1
        
        channels *= levels.flatten()
        
        return channels
            
    # Getter methods
    
    def get_node_count(self) -> int:
        return self.__node_count
    
    def get_band_count(self) -> int:
        return self.__band_count
    
    def get_lowcut(self) -> Union[float, int]:
        return self.__lowcut
    
    def get_highcut(self) -> Union[float, int]:
        return self.__highcut
    
    def get_scale(self) -> str:
        return self.__scale
    
    # Setter methods
    
    def set_node_count(self, node_count: int):
        self.__node_count = node_count
        
    def set_band_count(self, band_count: int):
        self.__band_count = band_count
        
    def set_lowcut(self, lowcut: Union[float, int]):
        self.__lowcut = lowcut
        
    def set_highcut(self, highcut: Union[float, int]):
        self.__highcut = highcut
        
    def set_scale(self, scale: str):
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
        
        if self.get_band_count() < self.get_node_count():
            raise ValueError()
        
        # Generate filterbank on input audio signal
        channels, energies = self.filterbank(
            signal=signal, sample_rate=sample_rate
        )
        
        # Rectify amplitudes of node channels
        channels = self.rectify(
            channels=channels,
            method=rectify
        )
        
        # Extract slow-varying envelopes for each node channel
        channels = self.envelopes(
            channels=channels,
            sample_rate=sample_rate,
            cutoff=cutoff
        )
        
        # Normalize envelope energy levels to account for gain
        channels = self.normalize(
            channels=channels,
            rms=energies
        )
        
        # Steer current energy from frequency band partitions to node channels
        channels = self.steer(
            channels=channels
        )
        
        # Apply frequency envelopes to balance energy levels between channels
        channels = self.balance(
            channels=channels,
            delta=balance
        )
        
        return channels
    
    def signal(
        self,
        channels: np.ndarray,
        sample_rate: Union[float, int],
        target_spl: Union[float, int] = 70,
        frequency: Union[float, int] = 2000,
        carrier: str = "sine"
    ) -> np.ndarray:
        
        if carrier == "sine":
            carrier_signal = Generators.sine(
                frequency=frequency,
                duration=channels.shape[0] / sample_rate,
                sample_rate=sample_rate
            )
        if carrier == "pulse":
            carrier_signal = Generators.pulse(
                frequency=frequency,
                duration=channels.shape[0] / sample_rate,
                sample_rate=sample_rate
            )
            
        channels = channels * carrier_signal
        
        channels, _ = Sound.spl_transform(
            signal=channels, target_spl=target_spl
        )
        
        return channels
    
    def simulate(
        self, 
        channels: np.ndarray, 
        sample_rate: Union[float, int], 
        target_spl: Union[float, int] = 70, 
        carrier: str = "sine", 
        merge: bool = True, 
        lowcut: Optional[Union[float, int]] = None, 
        highcut: Optional[Union[float, int]] = None
    ) -> np.ndarray:
        frequencies = np.ones_like(channels)
        
        frequencies *= self.calc_bands(
            band_count=self.get_node_count(),
            scale=self.get_scale(),
            ftype="centre"
        ).flatten()
        
        if lowcut is None:
            lowcut = np.min(frequencies)
        if highcut is None:
            highcut = np.max(frequencies)
            
        frequencies = self.compress(
            frequencies=frequencies,
            lowcut=lowcut,
            highcut=highcut
        )
        
        if carrier == "sine":
            carrier_signal = Generators.sine(
                frequency=frequencies,
                duration=channels.shape[0] / sample_rate,
                sample_rate=sample_rate
            )
            
        if carrier == "pulse":
            carrier_signal = Generators.pulse(
                frequency=frequencies,
                duration=channels.shape[0] / sample_rate,
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
                    duration=channels.shape[0] / sample_rate,
                    sample_rate=sample_rate
                )
            
                carrier_channel = Filter.butter_filter(
                    signal=carrier_source,
                    cutoffs=[lowcut, highcut],
                    sample_rate=sample_rate,
                    band_type="bandpass",
                    order=2,
                    method="filtfilt"
                )
                carrier_signal.append(carrier_channel)
            carrier_signal = np.hstack(carrier_signal)
                
        if carrier == "none":
            carrier_signal = np.ones_like(channels)
            
        signal = channels * carrier_signal
        
        signal, _ = Sound.spl_transform(
            signal=signal, target_spl=target_spl
        )
        
        if merge:
            signal = signal.sum(axis=1).reshape((-1, 1))
        
        return signal
    
    def visualize(
        self,
        channels: np.ndarray,
        sample_rate: Union[float, int],
        frame_rate: Union[float, int],
        path: str,
        figsize: Tuple[int, int] = (16, 4)
    ) -> None:
        def plot(signal, amplitude=1):
            signal = np.flip(signal)
            
            L = int(1000 / len(signal))
            if L % 2 == 0:
                L += 1
            peak = Windows.generalized_gaussian(length=L, mu=L/2, sigma=L/8)
            
            peaks = []
            for i in range(len(signal)):
                peaks.append(signal[i] * peak)
                
            peaks = np.hstack(peaks)
            
            plt.figure(figsize=figsize)
            
            xlim = (0, L-1 + np.ceil(L * (len(signal)-1) / 2))
            plt.hlines(0, xlim[0], xlim[1], colors="k", alpha=0.4)
            
            delta = amplitude / 3
            plt.hlines((np.arange(2) + 1) * delta, xlim[0], xlim[1], colors="k", alpha=0.1)
            plt.hlines((np.arange(2) + 1) * -delta, xlim[0], xlim[1], colors="k", alpha=0.1)
            
            for channel in range(len(signal)):
                x = np.arange(L) + np.ceil(L * channel / 2)
                
                plt.vlines(np.ceil(L//2) + np.ceil(L * channel / 2), -amplitude, amplitude, colors="k", alpha=0.1)
                
                plt.plot(x, peaks[:, channel], label=f'Node {channel}', alpha=0.3)
                plt.plot(x, -peaks[:, channel], label=f'Node {channel}', alpha=0.3)
                
                plt.fill_between(x, peaks[:, channel], 0, alpha=0.3)
                plt.fill_between(x, -peaks[:, channel], 0, alpha=0.3)
                
            plt.xlabel('Node (ID)', fontsize=14)
            plt.ylabel('Amplitude (V)', fontsize=14)
            plt.title('Modulated Signals for Each Node', fontsize=16)
            
            index = np.arange(len(signal))
            plt.xticks(np.ceil(index * L / 2) + np.floor(L/2), np.flip(index))
            
            plt.xlim(xlim)
            plt.ylim(-amplitude, amplitude)
            plt.tight_layout()
    
            
        index = np.arange(channels.shape[0])
        index = index[index % (sample_rate // frame_rate) == 0][1:]
        
        frame_channels = np.zeros((len(index), channels.shape[1]))
        for i in range(1, len(index)):
            j = index[i]
            frame_channels[i] = np.max(channels[index[i-1]:index[i]], axis=0)
            
        Utils.manage_directory(directory_path=path, delete_if_exists=True)
        num_digits = len(str(len(index)))
        string_format = os.path.join(path, f"frame_{{i:0{num_digits}d}}.png")
            
        max_val = np.max(np.abs(frame_channels))
        for i in range(len(index)):
            p = plot(signal=frame_channels[i, :], amplitude=max_val)
            plt.savefig(string_format.format(i=i), dpi=120)
            plt.close()
            
        image_files = [string_format.format(i=i) for i in np.arange(len(index))]
        images = [imageio.imread(img) for img in image_files]
        
        Utils.manage_directory(directory_path=path, delete_if_exists=True)
        imageio.mimwrite(os.path.join(path, "civis.mp4"), images, fps=frame_rate)
    
    
class Talkbox(Processor):
    def __init__(
        self, 
        node_count: int = 22, 
        band_count: int = 22, 
        lowcut: int = 20, 
        highcut: int = 20000, 
        scale: str = "octave"
    ) -> None:
        super().__init__(node_count, band_count, lowcut, highcut, scale)