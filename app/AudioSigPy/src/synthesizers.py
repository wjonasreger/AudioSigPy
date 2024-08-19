import numpy as np
from typing import Optional, Union, List, Tuple, Callable
from . import frequency as Frequency
from . import filter as Filter
from . import array as Array
from . import transforms as Transforms
from . import generators as Generators
from . import waveform as Waveform

# TODO
# - Add tom drums
# - Add error handling
# - Add docs/comments
# - Add compose method
# - Add piano stuff
        

class Synthesizer():
    def __init__(self, sample_rate: Optional[int] = None) -> None:
        """
        Initializes a Synthesizer tool.

        Args:
            sample_rate (float, int): Sample rate of the signal. Defaults to None.
        """
        
        self.set_sample_rate(sample_rate=sample_rate)
        
    def time_series(self, duration: Union[float, int]) -> np.ndarray:
        """
        Generate time series array for the signal

        Args:
            duration (float, int): Duration of signal in seconds.

        Returns:
            np.ndarray: Time series array
        """
        
        ts = Array.linseries(
            start=0,
            end=duration,
            size=int(self.get_sample_rate() * duration),
            endpoint=False
        )
        
        return ts
    
    def envelope(
        self,
        signal: np.ndarray,
        decay: Union[float, int],
        alpha: Union[float, int] = 1,
        delta: Union[float, int] = 0,
        inverse: bool = False,
        scale: bool = False
    ) -> np.ndarray:
        """
        Generate energy envelope for a signal

        Args:
            signal (np.ndarray): The input signal
            decay (float, int): The exponential decay of the envelope
            alpha (float, int, optional): The multiplicative factor. Defaults to 1.
            delta (float, int, optional): The additive factor. Defaults to 0.
            inverse (bool, optional): Option to invert the envelope. Defaults to False.
            scale (bool, optional): Option to scale the envelope to 1. Defaults to False.

        Raises:
            TypeError: If any inputs have unexpected type.

        Returns:
            np.ndarray: The envelope signal
        """
        
        if not isinstance(decay, (float, int)):
            raise TypeError(f"[ERROR] Envelope: Decay must be a number. Got {type(decay).__name__}.")
        if not isinstance(alpha, (float, int)):
            raise TypeError(f"[ERROR] Envelope: Alpha must be a number. Got {type(alpha).__name__}.")
        if not isinstance(delta, (float, int)):
            raise TypeError(f"[ERROR] Envelope: Delta must be a number. Got {type(delta).__name__}.")
        if not isinstance(inverse, bool):
            raise TypeError(f"[ERROR] Envelope: Inverse must be a boolean. Got {type(inverse).__name__}.")
        if not isinstance(scale, bool):
            raise TypeError(f"[ERROR] Envelope: Scale must be a boolean. Got {type(scale).__name__}.")
        
        signal = Array.validate(array=signal)
        
        # Generate envelope
        env = np.exp(-decay * signal)
        
        # Apply transformations
        if inverse:
            env = 1 / env
        if scale:
            env = self.scale(signal=env)
            
        env = alpha * env + delta
        
        return env
    
    def noise(
        self,
        duration: Union[float, int],
        filter: bool = True,
        cutoffs: Union[float, int] = 9000,
        band: str = "lowpass",
        dist: str = "gaussian"
    ) -> np.ndarray:
        """
        Generate a white noise signal

        Args:
            duration (float, int): Duration of signal in seconds
            filter (bool, optional): Option to filter noise. Defaults to True.
            cutoffs (float, int, optional): Filter cutoff frequencies. Defaults to 9000.
            band (str, optional): Frequency band type. Defaults to "lowpass".
            dist (str, optional): Sampling distribution for noise generation. Defaults to "gaussian".

        Raises:
            TypeError: If filter is not a boolean

        Returns:
            np.ndarray: Noise signal
        """
        
        if not isinstance(filter, bool):
            raise TypeError(f"[ERROR] Noise: filter must be a boolean. Got {type(filter).__name__}.")
        
        # Generate white noise signal
        source = Generators.white_noise(
            duration=duration,
            dist=dist,
            sample_rate=self.get_sample_rate()
        )
        
        # Apply filter to noise
        if filter:
            source = Filter.butter_filter(
                signal=source,
                cutoffs=cutoffs,
                sample_rate=self.get_sample_rate(),
                band_type=band
            )
            
        return source
    
    def resonator(
        self,
        signal: np.ndarray,
        bands: List[Tuple[int, int]],
        filt_fn: Callable[..., np.ndarray] = Filter.butter_filter
    ) -> np.ndarray:
        """
        Process the input signal by applying a bandpass filter across specified frequency bands.

        Args:
            signal (np.ndarray): The input signal for filtering
            bands (List[Tuple[int, int]]): The frequency bands
            filt_fn (Callable[...], optional): The filtering function. Defaults to Filter.butter_filter.

        Raises:
            Exception: Catches errors if bad function or data is passed in

        Returns:
            np.ndarray: The resonance signal after filtering is applied
        """
        
        try:
            res = np.zeros_like(signal)
            # Get filtered components of input signal and sum
            for band in bands:
                filt_signal = filt_fn(
                    signal=signal,
                    cutoffs=band,
                    sample_rate=self.get_sample_rate(),
                    band_type="bandpass"
                )
                res += filt_signal
        except Exception as e:
            raise Exception(f"[ERROR] Resonator: Filtering failed due to {e}.")
        
        return res
    
    def oscillator(
        self,
        frequency: Union[float, int, np.ndarray, list],
        duration: Union[float, int],
        sum: bool = True
    ) -> np.ndarray:
        """
        Generate a sinusoidal oscillating signal with input frequencies

        Args:
            frequency (float, np.ndarray): Frequency(s) in Hz.
            duration (float, int): Duration in seconds.
            sum (bool, optional): Option to sum signals in one complex signal. Defaults to True.

        Raises:
            TypeError: If frequency is not numeric
            ValueError: If frequency array has a bad shape

        Returns:
            np.ndarray: THe oscillating signal
        """
        
        if not isinstance(frequency, (int, float, list, np.ndarray)):
            raise TypeError(f"[ERROR] Oscillator: Frequency must a number or numeric array. Got {type(frequency).__name__}")
        
        frequency = Array.validate(array=frequency)
        sample_count = int(duration * self.get_sample_rate())
        
        # Check and validate frequency array shape
        if frequency.shape[0] != sample_count:
            if frequency.shape[1] == 1:
                frequency = frequency.transposed()
            else:
                raise ValueError(f"[ERROR] Oscillator: Frequency array has a bad shape. Got {frequency.shape}.")
            
        oscillators = []
        for i in range(frequency.shape[1]):
            # Prepare frequency data for new sine signal
            freq = frequency[:, i]
            if frequency.shape[0] == 1:
                freq = freq[0]
                
            # Generate sine signal
            signal = Generators.sine(
                frequency=freq,
                duration=duration,
                sample_rate=self.get_sample_rate()
            )
            
            oscillators.append(signal)
            
        # Combine sine signals as signal bank or complex signal
        complex_signal = np.hstack(oscillators)
        if sum:
            complex_signal = complex_signal.sum(axis=1)
            
        complex_signal = Array.validate(array=complex_signal)
        
        return complex_signal
    
    def smooth(
        self, 
        signal: np.ndarray, 
        T: int = 5
    ) -> np.ndarray:
        """
        Apply exponential smoothing to signal

        Args:
            signal (np.ndarray): Input signal
            T (int, optional): Size of smoothing window. Defaults to 5.

        Raises:
            TypeError: If T is not an integer
            ValueError: If T is not positive

        Returns:
            np.ndarray: The smoothed signal
        """
        
        if not isinstance(T, int):
            raise TypeError(f"[ERROR] Smooth: T must be an integer. Got {type(T).__name__}.")
        if T < 1:
            raise ValueError(f"[ERROR] Smooth: T must be a positive integer. Got {T}.")
        
        # Apply smoothing and convolution
        smoothing = np.array([0.1 * (1 - 0.1) ** n for n in range(T)])
        signal = Filter.convolve(signal=signal, kernel=smoothing)
        
        return signal
            
    def scale(
        self,
        signal: np.ndarray,
        scalar: Union[float, int] = 1.0
    ) -> np.ndarray:
        """
        Scale the input signal with scalar transformation

        Args:
            signal (np.ndarray): The input signal
            scalar (float, int, optional): The scalar value. Defaults to 1.0.

        Returns:
            np.ndarray: The scaled signal
        """
        
        signal = Transforms.scalar_transform(signal=signal, scalar=scalar)
        
        return signal
    
    def release(
        self,
        signal: np.ndarray,
        duration: Union[float, int],
        decay: Union[float, int] = 20
    ) -> np.ndarray:
        """
        Apply release envelope to quickly suppress signal levels

        Args:
            signal (np.ndarray): The input signal
            duration (float, int): The duration in seconds
            decay (float, int, optional): The exponential decay of the envelope. Defaults to 20.

        Returns:
            np.ndarray: The suppressed signal
        """
        
        signal = Array.validate(array=signal)
        
        ts = self.time_series(duration=duration)
        env = self.envelope(signal=ts, decay=decay)
        
        # Apply release envelope to end of signal
        signal[-env.shape[0]:] = signal[-env.shape[0]:] * env
        
        return signal
    
    def attack(
        self,
        signal: np.ndarray,
        duration: Union[float, int],
        decay: Union[float, int] = 20
    ) -> np.ndarray:
        """
        Apply attack envelope to quickly amplify signal levels

        Args:
            signal (np.ndarray): The input signal
            duration (float, int): The duration in seconds
            decay (float, int, optional): The exponential decay of the envelope. Defaults to 20.

        Returns:
            np.ndarray: The amplified signal
        """
        
        signal = Array.validate(array=signal)
        
        ts = self.time_series(duration=duration)
        env = np.flip(self.envelope(signal=ts, decay=decay))
        
        # Apply attack envelope to start of signal
        signal[:env.shape[0]] = signal[:env.shape[0]] * env
        
        return signal
    
    def get_sample_rate(self) -> float:
        """
        Returns:
            float: The sample rate of the synnthesizer
        """
        
        return self.__sample_rate
    
    def set_sample_rate(self, sample_rate: Union[int, float]) -> None:
        """
        Args:
            sample_rate (float, int): The sample rate for the synthesizer
        """
        
        sample_rate = Frequency.val_sample_rate(sample_rate=sample_rate)
        self.__sample_rate = sample_rate
        
        
class Drumset(Synthesizer):
    def __init__(self, sample_rate: Optional[int] = None) -> None:
        super().__init__(sample_rate)
        
    def kick_drum(
        self, 
        id: str,
        amplitude: Union[float, int] = 0.5
    ) -> Waveform.Waveform:
        
        # Frequency bands for resonance filtering
        freq_bands = [
            (20, 220), (220, 420), 
            (20, 220), (220, 420), (420, 620), (620, 820), 
            (20, 220), (220, 420), (420, 620), (620, 820), 
            (820, 1020), (1020, 1220)
        ]
        
        duration = 1.0
    
        # Generate the time and envelope series
        ts = self.time_series(duration=duration)
        env = self.envelope(signal=ts, decay=8)
        
        # Special envelope for burst component
        bts = self.time_series(duration=0.02)
        benv = self.envelope(signal=bts, decay=50)
        
        # Enveloped frequencies for tonal component
        tonal_freqs = np.hstack([
            60 * self.envelope(signal=ts, decay=3),
            90 * self.envelope(signal=ts, decay=5),
            120 * self.envelope(signal=ts, decay=8)
        ])
        
        # Generate signal components
        noise = self.noise(duration=0.02, filter=True, cutoffs=4000)
        resonance = self.resonator(signal=noise, bands=freq_bands)
        tone = self.oscillator(
            frequency=tonal_freqs, 
            duration=duration
        )
        
        # Apply envelopes and scaling
        burst = self.scale(signal=noise * benv)
        burst += self.scale(signal=resonance * benv)
        tone = self.scale(signal=tone * env, scalar=3)
        signal = tone
        signal[:bts.shape[0]] += burst
        signal = np.tanh(signal * 5)
        
        # Apply smoothing and scaling over the final signal
        signal = self.smooth(signal=signal)
        signal = self.scale(signal=signal, scalar=amplitude)
        
        # Create a Waveform object for kick drum signal
        wf = Waveform.Waveform(signal_id=id)
        wf.set_signal_data(data=signal, sample_rate=self.get_sample_rate())
        
        return wf
    
    def floor_tom():
        pass
    
    def tom_drum1():
        pass
    
    def tom_drum2():
        pass
    
    def snare_drum(
        self, 
        id, 
        duration=0.2,
        amplitude=0.5
    ) -> Waveform.Waveform:
        
        # Frequency bands for resonance filtering
        freq_bands = [
            (200, 400), (400, 600), (600, 800), (800, 1000),
            (1000, 1500), (1500, 2000), (2000, 2500), (2500, 3500),
            (3000, 4000), (4000, 5000), (5000, 6000),
            (6000, 7000), (7000, 8000), (8000, 9000)
        ]
        
        if duration < 0.15:
            duration = 0.15
        if duration > 0.3:
            duration = 0.3
            
        decay = 18 + 12 * (0.3 - duration) / 0.15
        
        # Generate the time and envelope series
        ts = self.time_series(duration=duration)
        env = self.envelope(signal=ts, decay=decay)
        
        # Generate signal components
        source = self.noise(duration=duration, filter=False)
        resonance = self.resonator(signal=source, bands=freq_bands)
        tone = self.oscillator(frequency=150, duration=duration)
        
        # Apply envelopes and scaling
        signal = self.scale(signal=source * env)
        signal += self.scale(signal=resonance * env)
        signal += self.scale(signal=tone * env)
        
        # Apply smoothing and scaling over the final signal
        signal = self.smooth(signal=signal)
        signal = self.scale(signal=signal, scalar=amplitude)
        
        # Create a Waveform object for snare signal
        wf = Waveform.Waveform(signal_id=id)
        wf.set_signal_data(data=signal, sample_rate=self.get_sample_rate())
        
        return wf
    
    def ride_cymbal(
        self, 
        id,
        duration=10.0,
        amplitude=0.5
    ) -> Waveform.Waveform:
            
        # Frequency bands for resonance filtering
        freq_bands = [
            (500, 1500), (500, 1500), (500, 1500), (1500, 2500), 
            (2500, 3500), (2500, 3500), (3000, 4000),
            (3500, 4500), (3500, 4500), 
            (4500, 5500), (5500, 6500)
        ]
        
        if duration < 0.4:
            duration = 0.4
        if duration > 10.0:
            duration = 10.0
    
        # Generate the time and envelope series
        ts = self.time_series(duration=duration)
        env = self.envelope(signal=ts, decay=0.8)
        
        # Generate signal components
        source = self.noise(duration=duration)
        resonance = self.resonator(signal=source, bands=freq_bands)
        
        # Apply envelopes and scaling
        signal = self.scale(signal=source * env)
        signal += self.scale(signal=resonance * env)
        
        # If cymbal is cut short, apply quick release
        if duration <= 7.0:
            signal = self.release(signal=signal)
        
        # Apply smoothing and scaling over the final signal
        signal = self.smooth(signal=signal)
        signal = self.scale(signal=signal, scalar=amplitude)
        
        # Create a Waveform object for ride cymbal signal
        wf = Waveform.Waveform(signal_id=id)
        wf.set_signal_data(data=signal, sample_rate=self.get_sample_rate())
        
        return wf
    
    def closed_hihat(
        self, 
        id, 
        amplitude=0.2
    ) -> Waveform.Waveform:
        
        # Frequency bands for resonance filtering
        freq_bands = [
            (3500, 4500), (4500, 5500), (5500, 6500), (6500, 7500)
        ]
        
        duration = 0.15
    
        # Generate the time and envelope series
        ts = self.time_series(duration=duration)
        env = self.envelope(signal=ts, decay=40)
        reso_env = self.envelope(signal=ts, decay=15)
        tone_env = self.envelope(signal=ts, decay=20)
        
        # Special envelope for shifting amplitudes of short signal
        dist_env = self.envelope(signal=ts, decay=-30) + 10
        dist_env *= Waveform.generalized_hann(length=ts.shape[0], delta=0.6)
        
        # Generate signal components
        source = self.noise(duration=duration, cutoffs=7500)
        resonance = self.resonator(signal=source, bands=freq_bands)
        tone = self.oscillator(frequency=60, duration=duration)
        
        # Apply envelopes and scaling
        signal = self.scale(signal=source * env)
        signal += self.scale(signal=resonance * reso_env, scalar=1/20)
        signal += self.scale(signal=tone * tone_env)
        
        # Apply smoothing and scaling over the final signal
        signal *= dist_env
        signal = self.smooth(signal=signal)
        signal = self.scale(signal=signal, scalar=amplitude)
        
        # Create a Waveform object for closed hi-hat signal
        wf = Waveform.Waveform(signal_id=id)
        wf.set_signal_data(data=signal, sample_rate=self.get_sample_rate())
        
        return wf
    
    def open_hihat(
        self, 
        id, 
        duration=0.2,
        amplitude=0.5,
        pedal=True
    ) -> Waveform.Waveform:
        
        # Frequency bands for resonance filtering
        freq_bands = [
            (500, 1500), (500, 1500), 
            (1500, 2500), (2500, 3500), 
            (3500, 4500), (3500, 4500), (3500, 4500),
            (4500, 5500), (4500, 5500), (4500, 5500),
            (5500, 6500), (6500, 7500)
        ]
        
        if duration < 0.01:
            duration = 0.01
        if duration > 8.0:
            duration = 8.0
        
        if duration <= 6:
            pedal = True
            
        if pedal:
            # Add 0.2 seconds for release component
            duration += 0.2
        
        # Generate the time and envelope series
        ts = self.time_series(duration=duration)
        env = self.envelope(signal=ts, decay=0.8)
        
        # Generate signal components
        source = self.noise(duration=duration)
        resonance = self.resonator(signal=source, bands=freq_bands)
        
        # Apply envelopes and scaling
        signal = self.scale(signal=source * env)
        signal += self.scale(signal=resonance * env)
        
        # When pedal is applied to hi-hat
        if pedal:
            signal = self.release(signal=signal, duration=0.21, decay=100)
        
            # Generate pedal signal
            release = self.hihat_pedal(
                id="release",
                amplitude=amplitude
            ).get_signal_data()
            
            # Add pedal release
            signal[-release.shape[0]:] += release

        # Apply smoothing and scaling over the final signal
        signal = self.smooth(signal=signal)
        signal = self.scale(signal=signal, scalar=amplitude)
        
        # Create a Waveform object for closed hi-hat signal
        wf = Waveform.Waveform(signal_id=id)
        wf.set_signal_data(data=signal, sample_rate=self.get_sample_rate())
        
        return wf
    
    def hihat_pedal(
        self, 
        id,
        amplitude=0.5
    ) -> Waveform.Waveform:
        
        # Frequency bands for resonance filtering
        freq_bands = [
            (500, 1500), (500, 1500), 
            (1500, 2500), (2500, 3500), 
            (3500, 4500), (3500, 4500), (3500, 4500),
            (4500, 5500), (4500, 5500), (4500, 5500),
            (5500, 6500), (6500, 7500)
        ]
        
        duration = 0.2
        
        # Generate primary signal
        # Generate the time and envelope series
        ts = self.time_series(duration=duration)
        env = self.envelope(signal=ts, decay=30)
        
        # Generate signal components
        source = self.noise(duration=duration)
        resonance = self.resonator(signal=source, bands=freq_bands)
        
        # Apply envelopes and scaling
        signal = self.scale(signal=source * env)
        signal += self.scale(signal=resonance * env)

        # Apply smoothing and scaling over the final signal
        signal = self.smooth(signal=signal)
        signal = self.scale(signal=signal, scalar=amplitude)
        
        # Create a Waveform object for closed hi-hat signal
        wf = Waveform.Waveform(signal_id=id)
        wf.set_signal_data(data=signal, sample_rate=self.get_sample_rate())
        
        return wf
    
    def crash_cymbal(
        self, 
        id,
        duration=5.0,
        amplitude=0.5
    ) -> Waveform.Waveform:
        
        # Frequency bands for resonance filtering
        freq_bands = [
            (500, 1000), (500, 1000),
            (2500, 3000),
            (3500, 4000), (3500, 4000),
            (3500, 4500),
            (4000, 4500), (4000, 4500),
            (4500, 5000), (4500, 5000),
            (5000, 5500), (5000, 5500),
            (5500, 6000), (6500, 7000), (7500, 8000)
        ]
        
        if duration < 0.4:
            duration = 0.4
        if duration > 5.0:
            duration = 5.0
    
        # Generate the time and envelope series
        ts = self.time_series(duration=duration)
        env = self.envelope(signal=ts, decay=1.4)
        
        # Generate signal components
        source = self.noise(duration=duration)
        resonance = self.resonator(signal=source, bands=freq_bands)
        
        # Apply envelopes and scaling
        signal = self.scale(signal=source * env)
        signal += self.scale(signal=resonance * env, scalar=1/20)
        
        # If cymbal is cut short, apply quick release
        if duration <= 4.5:
            signal = self.release(signal=signal)
        
        # Apply smoothing and scaling over the final signal
        signal = self.smooth(signal=signal)
        signal = self.scale(signal=signal, scalar=amplitude)
        
        # Create a Waveform object for crash cymbal signal
        wf = Waveform.Waveform(signal_id=id)
        wf.set_signal_data(data=signal, sample_rate=self.get_sample_rate())
        
        return wf
        
        
class Piano(Synthesizer):
    def __init__(self, sample_rate: Optional[int] = None) -> None:
        super().__init__(sample_rate)