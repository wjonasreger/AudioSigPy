import numpy as np
from typing import Optional, Union, List, Tuple, Callable
from . import frequency as Frequency
from . import filter as Filter
from . import array as Array
from . import transforms as Transforms
from . import generators as Generators
from . import waveform as Waveform
from . import windows as Windows
from . import sound as Sound


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
                frequency = frequency.transpose()
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
        amplitude: Union[float, int] = 0.5,
        waveform: bool = True
    ) -> Union[Waveform.Waveform, np.ndarray]:
        """
        Generate a kick drum signal.

        Args:
            id (str): Name of signal
            amplitude (Union[float, int], optional): Volume level of signal. Defaults to 0.5.
            waveform (bool, optional): Return as waveform or numpy array. Defaults to True.

        Raises:
            TypeError: If waveform is not boolean

        Returns:
            Union[Waveform.Waveform, np.ndarray]: A hit signal for kick drum
        """
        
        if not isinstance(waveform, bool):
            raise TypeError(f"[ERROR] Kick Drum: Waveform must be a boolean. Got {type(waveform).__name__}.")
        
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
        
        if waveform:
            # Create a Waveform object for signal
            wf = Waveform.Waveform(signal_id=id)
            wf.set_signal_data(data=signal, sample_rate=self.get_sample_rate())
            return wf
        else:
            # Return numpy array instead
            return signal
    
    def snare_drum(
        self, 
        id: str, 
        duration: Union[float, int] = 0.2,
        amplitude: Union[float, int] = 0.5,
        waveform: bool = True
    ) -> Union[Waveform.Waveform, np.ndarray]:
        """
        Generate a snare drum signal.

        Args:
            id (str): Name of signal
            duration (Union[float, int], optional): Time in seconds. Defaults to 0.2.
            amplitude (Union[float, int], optional): Volume level of signal. Defaults to 0.5.
            waveform (bool, optional): Return as waveform or numpy array. Defaults to True.

        Raises:
            TypeError: If waveform is not boolean, and duration is not number

        Returns:
            Union[Waveform.Waveform, np.ndarray]: A hit signal for snare drum
        """
        
        if not isinstance(waveform, bool):
            raise TypeError(f"[ERROR] Snare Drum: Waveform must be a boolean. Got {type(waveform).__name__}.")
        
        if not isinstance(duration, (float, int)):
            raise TypeError(f"[ERROR] Snare Drum: Duration must be a number. Got {type(duration).__name__}.")
        
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
        
        if waveform:
            # Create a Waveform object for signal
            wf = Waveform.Waveform(signal_id=id)
            wf.set_signal_data(data=signal, sample_rate=self.get_sample_rate())
            return wf
        else:
            # Return numpy array instead
            return signal
    
    def ride_cymbal(
        self, 
        id: str,
        duration: Union[float, int] = 10.0,
        amplitude: Union[float, int] = 0.5,
        waveform: bool = True
    ) -> Union[Waveform.Waveform, np.ndarray]:
        """
        Generate a ride cymbal signal.

        Args:
            id (str): Name of signal
            duration (Union[float, int], optional): Time in seconds. Defaults to 10.0.
            amplitude (Union[float, int], optional): Volume level of signal. Defaults to 0.5.
            waveform (bool, optional): Return as waveform or numpy array. Defaults to True.

        Raises:
            TypeError: If waveform is not boolean, and duration is not number

        Returns:
            Union[Waveform.Waveform, np.ndarray]: A hit signal for ride cymbal
        """
        
        if not isinstance(waveform, bool):
            raise TypeError(f"[ERROR] Ride Cymbal: Waveform must be a boolean. Got {type(waveform).__name__}.")
        
        if not isinstance(duration, (float, int)):
            raise TypeError(f"[ERROR] Ride Cymbal: Duration must be a number. Got {type(duration).__name__}.")
            
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
            signal = self.release(signal=signal, duration=0.2)
        
        # Apply smoothing and scaling over the final signal
        signal = self.smooth(signal=signal)
        signal = self.scale(signal=signal, scalar=amplitude)
        
        if waveform:
            # Create a Waveform object for signal
            wf = Waveform.Waveform(signal_id=id)
            wf.set_signal_data(data=signal, sample_rate=self.get_sample_rate())
            return wf
        else:
            # Return numpy array instead
            return signal
    
    def closed_hihat(
        self, 
        id: str, 
        amplitude: Union[float, int] = 0.2,
        waveform: bool = True
    ) -> Union[Waveform.Waveform, np.ndarray]:
        """
        Generate a closed hihat signal.

        Args:
            id (str): Name of signal
            amplitude (Union[float, int], optional): Volume level of signal. Defaults to 0.2.
            waveform (bool, optional): Return as waveform or numpy array. Defaults to True.

        Raises:
            TypeError: If waveform is not boolean

        Returns:
            Union[Waveform.Waveform, np.ndarray]: A hit signal for closed hihat
        """
        
        if not isinstance(waveform, bool):
            raise TypeError(f"[ERROR] Closed Hi-Hat: Waveform must be a boolean. Got {type(waveform).__name__}.")
        
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
        dist_env *= Windows.generalized_hann(length=ts.shape[0], delta=0.6)
        
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
        
        if waveform:
            # Create a Waveform object for signal
            wf = Waveform.Waveform(signal_id=id)
            wf.set_signal_data(data=signal, sample_rate=self.get_sample_rate())
            return wf
        else:
            # Return numpy array instead
            return signal
    
    def open_hihat(
        self, 
        id: str, 
        duration: Union[float, int] = 0.2,
        amplitude: Union[float, int] = 0.5,
        pedal: bool = True,
        waveform: bool = True
    ) -> Union[Waveform.Waveform, np.ndarray]:
        """
        Generate a open hihat signal.

        Args:
            id (str): Name of signal
            duration (Union[float, int], optional): Time in seconds. Defaults to 0.2.
            amplitude (Union[float, int], optional): Volume level of signal. Defaults to 0.5.
            pedal (bool, optional): Apply the pedal stop. Defaults to True.
            waveform (bool, optional): Return as waveform or numpy array. Defaults to True.

        Raises:
            TypeError: If waveform or pedal is not boolean, and duration is not number

        Returns:
            Union[Waveform.Waveform, np.ndarray]: A hit signal for open hihat
        """
        
        if not isinstance(waveform, bool):
            raise TypeError(f"[ERROR] Open Hi-Hat: Waveform must be a boolean. Got {type(waveform).__name__}.")
        
        if not isinstance(pedal, bool):
            raise TypeError(f"[ERROR] Open Hi-Hat: Pedal must be a boolean. Got {type(pedal).__name__}.")
        
        if not isinstance(duration, (float, int)):
            raise TypeError(f"[ERROR] Open Hi-Hat: Duration must be a number. Got {type(duration).__name__}.")
        
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
        
        if waveform:
            # Create a Waveform object for signal
            wf = Waveform.Waveform(signal_id=id)
            wf.set_signal_data(data=signal, sample_rate=self.get_sample_rate())
            return wf
        else:
            # Return numpy array instead
            return signal
    
    def hihat_pedal(
        self, 
        id: str,
        amplitude: Union[float, int] = 0.5,
        waveform: bool = True
    ) -> Union[Waveform.Waveform, np.ndarray]:
        """
        Generate a hihat pedal signal.

        Args:
            id (str): Name of signal
            amplitude (Union[float, int], optional): Volume level of signal. Defaults to 0.5.
            waveform (bool, optional): Return as waveform or numpy array. Defaults to True.

        Raises:
            TypeError: If waveform is not boolean

        Returns:
            Union[Waveform.Waveform, np.ndarray]: A hit signal for hihat pedal
        """
        
        if not isinstance(waveform, bool):
            raise TypeError(f"[ERROR] Hi-Hat Pedal: Waveform must be a boolean. Got {type(waveform).__name__}.")
        
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
        
        if waveform:
            # Create a Waveform object for signal
            wf = Waveform.Waveform(signal_id=id)
            wf.set_signal_data(data=signal, sample_rate=self.get_sample_rate())
            return wf
        else:
            # Return numpy array instead
            return signal
    
    def crash_cymbal(
        self, 
        id: str,
        duration: Union[float, int] = 5.0,
        amplitude: Union[float, int] = 0.5,
        waveform: bool = True
    ) -> Union[Waveform.Waveform, np.ndarray]:
        """
        Generate a crash cymbal signal.

        Args:
            id (str): Name of signal
            duration (Union[float, int], optional): Time in seconds. Defaults to 5.0.
            amplitude (Union[float, int], optional): Volume level of signal. Defaults to 0.5.
            waveform (bool, optional): Return as waveform or numpy array. Defaults to True.

        Raises:
            TypeError: If waveform is not boolean, and duration is not number

        Returns:
            Union[Waveform.Waveform, np.ndarray]: A hit signal for crash cymbal
        """
        
        if not isinstance(waveform, bool):
            raise TypeError(f"[ERROR] Crash Cymbal: Waveform must be a boolean. Got {type(waveform).__name__}.")
        
        if not isinstance(duration, (float, int)):
            raise TypeError(f"[ERROR] Crash Cymbal: Duration must be a number. Got {type(duration).__name__}.")
        
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
            signal = self.release(signal=signal, duration=0.2)
        
        # Apply smoothing and scaling over the final signal
        signal = self.smooth(signal=signal)
        signal = self.scale(signal=signal, scalar=amplitude)
        
        if waveform:
            # Create a Waveform object for signal
            wf = Waveform.Waveform(signal_id=id)
            wf.set_signal_data(data=signal, sample_rate=self.get_sample_rate())
            return wf
        else:
            # Return numpy array instead
            return signal
    
    def process(
        self,
        data: dict,
        id: str = "drum",
        waveform: bool = True
    ) -> Union[Waveform.Waveform, np.ndarray]:
        """
        Process many hit signal parameters to generate compositions for more complex signals (i.e., generate a song, rhythm, etc.)

        Args:
            data (dict): The hit parameter dataset.
            id (str, optional): An id name. Defaults to "drum".
            waveform (bool, optional): Return as waveform or numpy array. Defaults to True.

        Raises:
            ValueError: If data has mismatched sizes or a bad base unit measure
            TypeError: If base unit is not a number

        Returns:
            Union[Waveform.Waveform, np.ndarray]: The composed signal
        """
        
        # Check and validate song data
        n = len(data["beats"])
        attributes = ["types", "beats", "durations", "amplitudes", "tempos"]
        sorts = ["durations", "beats"]
        
        if not np.all(np.array([len(data[att]) for att in attributes]) == n):
            raise ValueError(f"[ERROR] Process: All song data attributes must be of same length. Got {[len(data[att]) for att in attributes]}.")
        
        if not isinstance(data["base_unit"], float):
            raise TypeError(f"[ERROR] Process: Base unit must be float. Got {type(data['base_unit']).__name__}.")
        
        if data["base_unit"] <= 0:
            raise ValueError(f"[ERROR] Process: Base unit must be a positive number. Got {data['base_unit']}.")
        
        # Sort incoming data
        for att_sort in sorts:
            idx_sorts = sorted(range(n), key=lambda k: data[att_sort][k])
            for att in attributes:
                data[att] = [data[att][i] for i in idx_sorts]
        
        # Create beat tempo mapping
        beat_tempo_map = {}
        max_time = 0
        
        for i in range(n):
            beat = data["beats"][i]
            tempo = data["tempos"][i]
            duration = data["durations"][i]
            
            beats = beat_tempo_map.keys()
            if beat not in beats:
                beat_tempo_map[beat] = tempo
            elif beat in beats and beat_tempo_map[beat] != tempo:
                raise ValueError(f"[ERROR] Process: Only one tempo can be defined at one time. Got {beat_tempo_map[beat]} and {tempo} on beat {beat}.")
            
            if beat + duration > max_time:
                max_time = beat + duration
                
        # Create temporal array
        beats = list(beat_tempo_map.keys())
        beats.append(max_time)
        tempos = list(beat_tempo_map.values())
        
        temporal = []
        for i in range(len(tempos)):
            duration = 4 * data["base_unit"] * (60 / tempos[i]) * (beats[i+1] - beats[i])
            sample_count = int(duration * self.get_sample_rate())
            x = np.linspace(beats[i], beats[i+1], sample_count, endpoint=False)
            temporal.extend(x)
        temporal = np.array(temporal).reshape((-1, 1))
        
        # Create empty signal
        signal = np.zeros((int(temporal.shape[0] * 1.5), temporal.shape[1]))
        
        # Populate empty signal with key signals
        for i in range(n):
            start = data["beats"][i]
            end = start + data["durations"][i]
            
            samples = temporal[(temporal >= start) & (temporal < end)]
            duration = len(samples) / self.get_sample_rate()
            
            # Store of drum functions
            drums = {
                "kick_drum": self.kick_drum, 
                "snare_drum": self.snare_drum, 
                "closed_hihat": self.closed_hihat, 
                "hihat_pedal": self.hihat_pedal, 
                "open_hihat": self.open_hihat, 
                "ride_cymbal": self.ride_cymbal, 
                "crash_cymbal": self.crash_cymbal
            }
            
            # Get parameters for specific drum type and hit instance
            drum_type = data["types"][i]
            
            if drum_type not in drums.keys():
                raise ValueError(f"[ERROR] Process: Drum type not supported. Got {drum_type}.")
            
            hit_kwargs = {
                "id": "hit",
                "amplitude": data["amplitudes"][i],
                "waveform": False
            }
            
            if drum_type not in ['kick_drum', 'closed_hihat', 'hihat_pedal']:
                hit_kwargs["duration"] = duration

            # Generate the hit signal
            hit = drums[drum_type](**hit_kwargs)
            
            # Insert the hit signal in song signal
            samples = temporal[temporal < start]
            start_idx = len(samples)
            end_idx = start_idx + hit.shape[0]
            signal[start_idx:end_idx] += hit
            
        # Clip excess tail from signal
        signal = signal[:temporal.shape[0]]
        
        if waveform:
            # Create a Waveform object for signal
            wf = Waveform.Waveform(signal_id=id)
            wf.set_signal_data(data=signal, sample_rate=self.get_sample_rate())
            return wf
        else:
            # Return numpy array instead
            return signal
        
        
class Piano(Synthesizer):
    def __init__(self, sample_rate: Optional[int] = None) -> None:
        super().__init__(sample_rate)
        
    def adsr(
        self,
        signal: np.ndarray,
        attack: float,
        decay: float,
        release: float
    ) -> np.ndarray:
        """
        Generate an envelope to simulate a piano key signal with Attack, Decay, Sustain, and Release windows (ADSR).

        Args:
            signal (np.ndarray): The input time series signal
            attack (float): The length of time in seconds.
            decay (float): The length of time in seconds.
            release (float): The length of time in seconds.

        Raises:
            TypeError: If any ADSR parameters are not floats
            ValueError: If any ADSR parameters are not in [0, 1]

        Returns:
            np.ndarray: The ADSR energy envelope signal
        """
        
        # Check and validate
        if not isinstance(attack, float):
            raise TypeError(f"[ERROR] ADSR: Attack must be a float. Got {type(attack).__name__}.")
        if not isinstance(decay, float):
            raise TypeError(f"[ERROR] ADSR: Decay must be a float. Got {type(decay).__name__}.")
        if not isinstance(release, float):
            raise TypeError(f"[ERROR] ADSR: Release must be a float. Got {type(release).__name__}.")
        
        if attack <=0 or attack > 1:
            raise ValueError(f"[ERROR] ADSR: Attack must be between [0, 1]. Got {attack}.")
        if decay <=0 or decay > 1:
            raise ValueError(f"[ERROR] ADSR: Decay must be between [0, 1]. Got {decay}.")
        if release <=0 or release > 1:
            raise ValueError(f"[ERROR] ADSR: Release must be between [0, 1]. Got {release}.")
        
        # Create empty array for envelope weights
        weights = np.ones_like(signal)
        
        # Adjust time parameters as needed
        duration = signal.shape[0] / self.get_sample_rate()
        sustain = duration - attack - decay * 0.8 - release / 1.1
        if sustain <= 0:
            decay = duration - attack - release / 1.1
        
        # Generate attack envelope to start off signal growth
        ts_attack = self.time_series(duration=attack)
        attack_len = ts_attack.shape[0]
        weights[:attack_len] = self.envelope(
            signal=ts_attack,
            decay=60,
            inverse=True,
            scale=True
        )
        
        # Generate decay window to drop off after attack
        ts_decay = self.time_series(duration=decay)
        decay_len = ts_decay.shape[0]
        weights[attack_len:attack_len + decay_len] = self.envelope(
            signal=ts_decay,
            decay=30,
            alpha=1-0.2,
            delta=0.2,
            inverse=False,
            scale=False
        )
        
        # If signal is not short, generate sustaining decay to draw out the signal
        if sustain > 0:
            ts_sustain = self.time_series(duration=sustain)
            sustain_len = ts_sustain.shape[0]
            sustain_start = attack_len + int(decay_len*0.8)
            sustain_end = sustain_start + sustain_len
            weights[sustain_start:sustain_end] = self.envelope(
                signal=ts_sustain,
                decay=0.5,
                alpha=weights[sustain_start][0],
                inverse=False,
                scale=False
            )
        
        # End the signal with a strong decay to simulate natural ending to signal
        ts_release = self.time_series(duration=release)
        release_len = ts_release.shape[0]
        weights[-release_len:] = self.envelope(
            signal=ts_release,
            decay=60,
            alpha=weights[-release_len][0],
            inverse=False,
            scale=False
        )
        
        return weights
        
    
    def key(
        self,
        note: List[int],
        octave: List[int],
        duration: float,
        amplitude: float = 0.5,
        id: str = "piano",
        waveform: bool = True
    ) -> Union[Waveform.Waveform, np.ndarray]:
        """
        Generate a piano key signal to simulate playing a piano sound.

        Args:
            note (List[int]): The note ids. C=0, ..., B=11.
            octave (List[int]): The octave ids. Middle C is in octave 4.
            duration (float): The duration in seconds.
            amplitude (float, optional): The volume amplitude of the signal. Defaults to 0.5.
            id (str, optional): The id name of the signal. Defaults to "piano".
            waveform (bool, optional): Return as waveform or numpy array. Defaults to True.

        Raises:
            TypeError: If any parameters have incorrect typing
            ValueError: If notes and octaves have conflicting array shapes.

        Returns:
            Union[Waveform.Waveform, np.ndarray]: The key signal
        """
        
        # Validate the notes and octaves
        note = Array.validate(array=note)
        octave = Array.validate(array=octave)
        
        if not isinstance(duration, (float, int)):
            raise TypeError(f"[ERROR] Key: Duration must be a float. Got {type(duration).__name__}.")
        if not isinstance(amplitude, (float, int)):
            raise TypeError(f"[ERROR] Key: Amplitude must be a float. Got {type(amplitude).__name__}.")
        if not isinstance(waveform, bool):
            raise TypeError(f"[ERROR] Key: waveform must be a bool. Got {type(waveform).__name__}.")
        
        if octave.shape == (1, 1):
            octave = np.ones_like(note) * octave[0][0]
        if octave.shape != note.shape:
            raise ValueError(f"[ERROR] Key: Notes and octaves must have same shape. Got {note.shape} != {octave.shape}.")
        
        note = note.flatten()
        octave = octave.flatten()
            
        # Generate a noise burst to simulate the hammer strike
        bts = self.time_series(duration=0.1)
        benv = self.envelope(signal=bts, decay=50)
        
        freq_bands = [(20, 400)]
        
        # Generate signal components for the burst
        noise = self.noise(duration=0.1, filter=True, cutoffs=1000)
        resonance = self.resonator(signal=noise, bands=freq_bands)
        burst = noise + resonance
        burst = self.scale(signal=burst*benv, scalar=0.02)
        
        # Add slight amount of time to ensure minimal length of 0.1 seconds 
        duration += 0.1
        
        signal = 0
        for i in range(len(note)):
            # Get frequencies, magnitudes, and energy of a note
            f = empirical_piano[octave[i]][note[i]]["frequency"]
            m = empirical_piano[octave[i]][note[i]]["magnitude"]
            e = empirical_piano[octave[i]][note[i]]["energy"]
            
            # Generate the complex sine wave for a note
            source = 0
            for i in range(len(f)):
                source += Generators.sine(
                    frequency=f[i],
                    duration=duration,
                    sample_rate=self.get_sample_rate(),
                    amplitude=m[i]/max(m)
                )
                
            # Adjust signal energy according to typical key energy
            source, _ = Sound.spl_transform(signal=source, target_spl=e)
        
            # Apply ADSR envelope on signal to simulate energy shape of piano key
            envelope = self.adsr(signal=source, attack=0.02, decay=0.2, release=0.1)
            source *= envelope
            signal += source
            
        # Apply envelopes and scaling
        signal[:bts.shape[0]] += burst
        signal = np.tanh(signal)
        signal = self.smooth(signal=signal, T=100)
        signal = self.scale(signal=signal, scalar=amplitude)
        
        if waveform:
            # Create a Waveform object for signal
            wf = Waveform.Waveform(signal_id=id)
            wf.set_signal_data(data=signal, sample_rate=self.get_sample_rate())
            return wf
        else:
            # Return numpy array instead
            return signal
    
    def process(
        self,
        data: dict,
        id: str = "piano",
        waveform: bool = True
    ) -> Union[Waveform.Waveform, np.ndarray]:
        """
        Process many key signal parameters to generate compositions for more complex signals (i.e., generate a song, rhythm, etc.)

        Args:
            data (dict): The key parameter dataset.
            id (str, optional): An id name. Defaults to "piano".
            waveform (bool, optional): Return as waveform or numpy array. Defaults to True.

        Raises:
            ValueError: If data has mismatched sizes or a bad base unit measure
            TypeError: If base unit is not a number

        Returns:
            Union[Waveform.Waveform, np.ndarray]: The composed signal
        """
        
        # Check and validate song data
        n = len(data["beats"])
        attributes = ["notes", "octaves", "beats", "durations", "amplitudes", "tempos"]
        sorts = ["durations", "beats"]
        
        if not np.all(np.array([len(data[att]) for att in attributes]) == n):
            raise ValueError(f"[ERROR] Process: All song data attributes must be of same length. Got {[len(data[att]) for att in attributes]}.")
        
        if not isinstance(data["base_unit"], float):
            raise TypeError(f"[ERROR] Process: Base unit must be float. Got {type(data['base_unit']).__name__}.")
        
        if data["base_unit"] <= 0:
            raise ValueError(f"[ERROR] Process: Base unit must be a positive number. Got {data['base_unit']}.")
        
        # Sort incoming data
        for att_sort in sorts:
            idx_sorts = sorted(range(n), key=lambda k: data[att_sort][k])
            for att in attributes:
                data[att] = [data[att][i] for i in idx_sorts]
        
        # Create beat tempo mapping
        beat_tempo_map = {}
        max_time = 0
        
        for i in range(n):
            beat = data["beats"][i]
            tempo = data["tempos"][i]
            duration = data["durations"][i]
            
            beats = beat_tempo_map.keys()
            if beat not in beats:
                beat_tempo_map[beat] = tempo
            elif beat in beats and beat_tempo_map[beat] != tempo:
                raise ValueError(f"[ERROR] Process: Only one tempo can be defined at one time. Got {beat_tempo_map[beat]} and {tempo} on beat {beat}.")
            
            if beat + duration > max_time:
                max_time = beat + duration
                
        # Create temporal array
        beats = list(beat_tempo_map.keys())
        beats.append(max_time)
        tempos = list(beat_tempo_map.values())
        
        temporal = []
        for i in range(len(tempos)):
            duration = 4 * data["base_unit"] * (60 / tempos[i]) * (beats[i+1] - beats[i])
            sample_count = int(duration * self.get_sample_rate())
            x = np.linspace(beats[i], beats[i+1], sample_count, endpoint=False)
            temporal.extend(x)
        temporal = np.array(temporal).reshape((-1, 1))
        
        # Create empty signal
        signal = np.zeros((int(temporal.shape[0] * 1.1), temporal.shape[1]))
        
        # Populate empty signal with key signals
        for i in range(n):
            start = data["beats"][i]
            end = start + data["durations"][i]
            
            samples = temporal[(temporal >= start) & (temporal < end)]
            duration = len(samples) / self.get_sample_rate()
            
            key = self.key(
                note=data["notes"][i],
                octave=data["octaves"][i],
                duration=duration,
                amplitude=data["amplitudes"][i],
                waveform=False
            )
            
            samples = temporal[temporal < start]
            start_idx = len(samples)
            end_idx = start_idx + key.shape[0]
            signal[start_idx:end_idx] += key
            
        # Clip excess tail from signal
        signal = signal[:temporal.shape[0]]
        
        if waveform:
            # Create a Waveform object for signal
            wf = Waveform.Waveform(signal_id=id)
            wf.set_signal_data(data=signal, sample_rate=self.get_sample_rate())
            return wf
        else:
            # Return numpy array instead
            return signal
    
    
# Empirically derived parameter values for each piano key on an 88-key piano
# Frequencies and magnitudes were derived using samples contained in the `audio samples/piano` directory.
# Values were extracted using spectral and peak detection tools in AudioSigPy
# Energy values are fitted from a 2nd order polynomial curve on measured SPL values from samples
empirical_piano = {
    0: {
        9: {
            'energy': 77.8, 
            'frequency': [107.66, 135.18, 162.71, 246.08, 274.42, 303.56, 333.51, 362.65, 392.6, 422.55, 484.88, 517.26, 549.64, 582.02, 615.21, 649.21, 683.2, 789.25, 826.48, 900.96, 939.81, 1183.47, 1313.79], 
            'magnitude': [0.5478, 1.0, 0.314, 0.1095, 0.2104, 0.5857, 0.3515, 0.487, 0.5405, 0.3566, 0.1244, 0.1577, 0.2331, 0.4565, 0.2954, 0.2283, 0.1951, 0.0951, 0.1381, 0.1952, 0.2391, 0.1603, 0.1233]
        }, 
        10: {
            'energy': 78.0, 
            'frequency': [114.04, 143.09, 172.15, 201.93, 261.49, 291.27, 321.78, 353.01, 384.24, 415.48, 447.44, 546.22, 580.36, 614.5, 684.96, 721.27, 757.59, 870.18, 908.68, 947.9, 987.85, 1283.48, 1328.51], 
            'magnitude': [0.6209, 0.5401, 0.2827, 0.1777, 0.0993, 0.2867, 0.3552, 1.0, 0.7467, 0.6348, 0.2429, 0.1971, 0.3117, 0.2797, 0.2923, 0.1685, 0.1068, 0.1423, 0.1802, 0.2707, 0.2252, 0.0881, 0.1525]
        }, 
        11: {
            'energy': 78.1, 
            'frequency': [121.81, 152.44, 183.8, 215.17, 278.62, 310.72, 342.81, 375.63, 409.18, 442.74, 476.29, 581.32, 617.06, 652.8, 690.0, 727.2, 765.12, 921.21, 962.06, 1045.21, 1307.78, 1354.47, 1493.05], 
            'magnitude': [1.0, 0.2173, 0.2386, 0.0761, 0.1476, 0.3393, 0.5758, 0.1604, 0.4525, 0.0743, 0.2332, 0.0916, 0.1354, 0.1307, 0.0828, 0.2514, 0.096, 0.1493, 0.0723, 0.0493, 0.0621, 0.0629, 0.0983]
        }
    }, 
    1: {
        0: {
            'energy': 78.2, 
            'frequency': [128.26, 161.41, 193.84, 226.98, 293.28, 327.14, 361.01, 395.6, 430.19, 465.49, 501.52, 611.05, 685.99, 724.18, 763.09, 802.73, 964.86, 1089.52, 1136.35, 1181.03, 1364.78, 1413.06, 1488.72], 
            'magnitude': [1.0, 0.3962, 0.2153, 0.1453, 0.1498, 0.1555, 0.4245, 0.4698, 0.4924, 0.397, 0.3409, 0.0988, 0.2331, 0.3748, 0.2007, 0.14, 0.128, 0.0834, 0.0818, 0.0915, 0.1063, 0.0966, 0.098]
        }, 
        1: {
            'energy': 78.3, 
            'frequency': [102.02, 136.27, 170.52, 205.5, 240.48, 311.17, 346.88, 382.58, 419.02, 456.19, 493.35, 685.74, 725.09, 765.9, 806.7, 848.24, 889.78, 1063.22, 1152.85, 1198.76, 1485.15, 1655.67, 2218.25], 
            'magnitude': [0.0525, 1.0, 0.3139, 0.2953, 0.12, 0.2561, 0.448, 0.4101, 0.4554, 0.7216, 0.3636, 0.0733, 0.2759, 0.1997, 0.1272, 0.0513, 0.0749, 0.0531, 0.1173, 0.0515, 0.0952, 0.0631, 0.0467]
        }, 
        2: {
            'energy': 78.4, 
            'frequency': [108.25, 145.34, 181.67, 218.01, 255.1, 330.04, 368.65, 406.5, 445.1, 484.47, 523.83, 563.19, 604.07, 685.82, 727.46, 769.09, 812.24, 855.39, 898.53, 943.2, 1078.69, 1172.56, 1668.38], 
            'magnitude': [0.3015, 0.6769, 0.669, 0.2124, 0.3388, 0.0869, 0.1656, 1.0, 0.3228, 0.8945, 0.2199, 0.3052, 0.0889, 0.1171, 0.4929, 0.0981, 0.1744, 0.1894, 0.3248, 0.1111, 0.1355, 0.0699, 0.1197]
        }, 
        3: {
            'energy': 78.5, 
            'frequency': [115.12, 153.24, 191.37, 230.23, 269.83, 309.42, 348.28, 388.61, 428.94, 469.27, 510.33, 551.39, 593.18, 634.97, 720.03, 763.29, 807.28, 896.0, 941.46, 1173.9, 1221.56, 1318.34, 1367.47], 
            'magnitude': [0.8016, 0.7481, 0.4576, 0.2771, 0.362, 0.1345, 0.5224, 0.6821, 0.9289, 1.0, 0.5499, 0.6428, 0.2353, 0.1597, 0.0913, 0.1627, 0.118, 0.3019, 0.323, 0.164, 0.1505, 0.0973, 0.1315]
        }, 
        4: {
            'energy': 78.5, 
            'frequency': [81.47, 122.21, 162.95, 204.36, 245.78, 287.19, 371.38, 413.48, 456.25, 499.7, 543.16, 586.61, 631.42, 859.54, 906.39, 953.24, 1001.44, 1174.57, 1247.9, 1298.82, 1350.42, 1781.55, 1839.94], 
            'magnitude': [0.0418, 1.0, 0.5139, 0.4246, 0.1328, 0.1162, 0.0762, 0.4521, 0.5011, 0.3128, 0.1739, 0.192, 0.2354, 0.1041, 0.2856, 0.2907, 0.0889, 0.0708, 0.0619, 0.0436, 0.0646, 0.0607, 0.0528]
        }, 
        5: {
            'energy': 78.6, 
            'frequency': [129.09, 171.88, 216.08, 258.87, 303.07, 347.27, 392.17, 437.07, 482.67, 573.87, 620.17, 667.17, 714.88, 908.51, 957.61, 1008.13, 1109.85, 1241.74, 1372.93, 1427.65, 1482.37, 1765.8, 1967.14], 
            'magnitude': [1.0, 0.232, 0.0603, 0.2769, 0.3418, 0.1085, 0.2974, 0.1028, 0.2661, 0.1924, 0.203, 0.0776, 0.0848, 0.091, 0.1933, 0.1344, 0.1018, 0.0996, 0.1622, 0.0654, 0.1174, 0.0826, 0.0663]
        }, 
        6: {
            'energy': 78.7, 
            'frequency': [91.18, 136.77, 182.36, 227.95, 275.01, 320.6, 414.72, 462.52, 509.58, 558.11, 606.65, 655.91, 705.91, 858.13, 911.07, 963.28, 1016.96, 1071.37, 1126.52, 1312.56, 1416.24, 1475.8, 1927.29], 
            'magnitude': [0.0576, 1.0, 0.4492, 0.115, 0.3696, 0.3361, 0.4132, 0.6351, 0.254, 0.2736, 0.1024, 0.1844, 0.0435, 0.0383, 0.0801, 0.1937, 0.043, 0.0613, 0.0671, 0.1027, 0.0385, 0.0992, 0.0331]
        }, 
        7: {
            'energy': 78.7, 
            'frequency': [97.08, 146.0, 194.92, 243.83, 293.5, 342.42, 392.84, 442.51, 492.94, 544.11, 595.29, 647.21, 699.89, 752.58, 970.07, 1025.76, 1082.2, 1140.15, 1195.84, 1399.79, 1440.43, 1499.13, 1566.11], 
            'magnitude': [0.1775, 0.6501, 0.3757, 0.3509, 1.0, 0.7107, 0.0901, 0.1468, 0.5062, 0.1854, 0.2552, 0.1419, 0.0411, 0.1008, 0.0791, 0.0692, 0.2346, 0.1569, 0.0479, 0.1796, 0.0921, 0.1045, 0.0442]
        }, 
        8: {
            'energy': 78.8, 
            'frequency': [102.54, 153.81, 205.83, 257.85, 309.12, 361.14, 413.16, 466.69, 520.22, 573.75, 627.28, 681.57, 736.61, 791.64, 904.74, 962.04, 1137.71, 1258.34, 1473.97, 1574.99, 1641.34, 1708.44, 1775.54], 
            'magnitude': [0.3161, 0.389, 0.3944, 0.3, 1.0, 0.1768, 0.0608, 0.3751, 0.3151, 0.2709, 0.1918, 0.0516, 0.1585, 0.0477, 0.0643, 0.1587, 0.2287, 0.0438, 0.1433, 0.0639, 0.0663, 0.05, 0.0377]
        }, 
        9: {
            'energy': 78.8, 
            'frequency': [54.55, 109.1, 163.65, 218.2, 273.48, 328.76, 384.04, 495.33, 551.33, 608.06, 664.07, 721.53, 778.26, 836.45, 1013.2, 1133.21, 1195.04, 1255.41, 1317.96, 1557.26, 1638.72, 1705.64, 1771.82], 
            'magnitude': [0.069, 0.8589, 1.0, 0.3196, 0.6592, 0.4647, 0.374, 0.2243, 0.1387, 0.0918, 0.1094, 0.2465, 0.0931, 0.227, 0.0996, 0.1324, 0.0778, 0.1559, 0.1194, 0.1537, 0.0604, 0.126, 0.1898]
        }, 
        10: {
            'energy': 78.9, 
            'frequency': [115.35, 173.41, 231.47, 289.53, 347.59, 406.41, 464.47, 524.06, 582.88, 642.46, 762.4, 822.75, 884.63, 1195.55, 1260.48, 1324.65, 1389.59, 1525.57, 1647.03, 1724.95, 1798.29, 1868.57, 2457.56], 
            'magnitude': [1.0, 0.3789, 0.1451, 0.545, 0.4262, 0.5131, 0.0547, 0.0708, 0.0629, 0.1611, 0.2915, 0.2863, 0.0935, 0.075, 0.0638, 0.122, 0.0583, 0.04, 0.0406, 0.0632, 0.0436, 0.0274, 0.0295]
        }, 
        11: {
            'energy': 78.9, 
            'frequency': [61.69, 122.61, 184.3, 245.99, 307.68, 370.14, 431.83, 494.3, 557.53, 620.76, 684.0, 747.23, 811.23, 876.01, 940.78, 1138.19, 1205.28, 1272.37, 1341.0, 1409.63, 1478.26, 1835.3, 1908.56], 
            'magnitude': [0.0439, 1.0, 0.7081, 0.212, 0.58, 0.1521, 0.2619, 0.0327, 0.07, 0.1555, 0.0629, 0.1021, 0.1517, 0.0289, 0.038, 0.0605, 0.034, 0.061, 0.0814, 0.029, 0.0562, 0.0357, 0.0284]
        }
    }, 
    2: {
        0: {
            'energy': 78.9, 
            'frequency': [64.79, 130.31, 195.82, 261.34, 326.86, 393.1, 459.35, 525.59, 592.57, 658.81, 726.51, 794.21, 852.45, 931.07, 999.5, 1139.27, 1209.88, 1280.5, 1351.84, 1426.09, 1498.89, 1570.95, 1862.87], 
            'magnitude': [0.0618, 1.0, 0.2698, 0.2186, 0.1375, 0.5258, 0.3758, 0.0226, 0.0496, 0.1373, 0.1076, 0.0313, 0.0217, 0.0357, 0.0317, 0.0219, 0.0202, 0.043, 0.035, 0.0164, 0.0362, 0.022, 0.0513]
        }, 
        1: {
            'energy': 78.9, 
            'frequency': [68.3, 137.36, 206.42, 275.48, 344.55, 414.38, 484.21, 624.63, 695.23, 765.83, 837.19, 898.58, 982.22, 1054.35, 1129.56, 1276.89, 1352.09, 1504.8, 1582.3, 1660.57, 1963.68, 2146.31, 2229.95], 
            'magnitude': [0.0912, 1.0, 0.6571, 0.5365, 0.4998, 0.725, 0.2325, 0.1023, 0.0469, 0.1858, 0.0934, 0.0439, 0.1137, 0.0397, 0.0312, 0.0931, 0.0672, 0.0403, 0.0535, 0.0317, 0.0395, 0.0376, 0.0523]
        }, 
        2: {
            'energy': 78.9, 
            'frequency': [72.35, 146.34, 219.52, 292.69, 366.68, 440.68, 514.67, 664.3, 739.12, 814.76, 966.86, 1044.14, 1121.42, 1276.81, 1354.91, 1434.66, 1596.63, 1678.02, 1760.24, 2087.46, 2186.12, 2274.91, 2362.88], 
            'magnitude': [0.0413, 0.8461, 0.1088, 1.0, 0.4174, 0.5856, 0.3928, 0.0753, 0.2871, 0.2647, 0.1366, 0.1356, 0.0444, 0.0832, 0.0462, 0.0447, 0.0674, 0.1451, 0.1841, 0.068, 0.0459, 0.0431, 0.0671]
        }, 
        3: {
            'energy':78.9, 
            'frequency': [76.84, 153.68, 184.27, 231.26, 308.85, 386.43, 464.77, 542.35, 621.43, 699.76, 778.09, 857.92, 937.74, 1018.31, 1099.62, 1181.69, 1346.55, 1597.96, 1683.01, 1768.8, 2199.99, 2302.94, 2395.45], 
            'magnitude': [0.0248, 0.7366, 0.0233, 0.1632, 0.9277, 0.2117, 1.0, 0.1037, 0.0704, 0.0265, 0.0254, 0.0473, 0.0274, 0.1252, 0.1239, 0.2234, 0.0244, 0.0351, 0.1057, 0.1643, 0.1519, 0.035, 0.0637]
        }, 
        4: {
            'energy': 78.9, 
            'frequency': [81.7, 122.15, 137.22, 163.4, 245.89, 328.38, 410.87, 493.37, 576.65, 659.94, 743.22, 827.3, 911.38, 1081.91, 1139.02, 1168.37, 1608.59, 1699.02, 1789.44, 2165.41, 2337.54, 2352.61, 2450.17], 
            'magnitude': [0.0257, 0.0314, 0.0286, 1.0, 0.5334, 0.5023, 0.4148, 0.3138, 0.2826, 0.0327, 0.0746, 0.3169, 0.0664, 0.0644, 0.0215, 0.197, 0.03, 0.0417, 0.0678, 0.0241, 0.0492, 0.0312, 0.0312]
        }, 
        5: {
            'energy': 78.9, 
            'frequency': [86.36, 136.43, 172.71, 261.25, 297.53, 349.06, 436.87, 524.67, 612.48, 790.28, 880.26, 969.52, 1059.51, 1150.94, 1243.83, 1334.54, 1521.77, 1615.38, 1711.18, 1902.03, 2099.42, 2200.29, 2485.49], 
            'magnitude': [0.2306, 0.1039, 0.1433, 1.0, 0.0606, 0.6772, 0.4963, 0.8343, 0.0791, 0.3417, 0.2335, 0.166, 0.0674, 0.3311, 0.3332, 0.1118, 0.0516, 0.0668, 0.0491, 0.0651, 0.0825, 0.0632, 0.189]
        }, 
        6: {
            'energy': 78.9, 
            'frequency': [91.55, 122.07, 165.36, 183.11, 276.08, 344.92, 368.34, 461.31, 554.29, 647.26, 740.94, 836.04, 929.72, 1024.83, 1119.93, 1202.25, 1313.68, 1411.62, 1507.43, 1606.79, 1708.99, 2111.4, 2215.01], 
            'magnitude': [0.0151, 0.0215, 0.0155, 0.7466, 1.0, 0.0069, 0.1334, 0.3051, 0.5096, 0.1342, 0.0108, 0.0662, 0.038, 0.0712, 0.0121, 0.0332, 0.1135, 0.0281, 0.0082, 0.0073, 0.0121, 0.0069, 0.0236]
        }, 
        7: {
            'energy': 78.9, 
            'frequency': [97.52, 120.89, 166.03, 195.04, 294.18, 391.7, 490.03, 589.16, 688.29, 788.23, 887.37, 988.92, 1090.47, 1190.41, 1277.45, 1292.77, 1397.54, 1501.51, 1606.29, 1815.03, 1926.25, 2248.64, 2365.5], 
            'magnitude': [0.0572, 0.0395, 0.013, 0.4044, 1.0, 0.4032, 0.065, 0.2782, 0.1479, 0.0313, 0.0211, 0.0752, 0.0298, 0.0241, 0.0682, 0.043, 0.0337, 0.0161, 0.0149, 0.0199, 0.0143, 0.0489, 0.061]
        }, 
        8: {
            'energy': 78.8, 
            'frequency': [103.16, 164.59, 189.79, 205.54, 279.56, 309.49, 413.44, 517.39, 621.34, 726.08, 830.82, 936.34, 1041.87, 1148.97, 1256.86, 1347.42, 1363.96, 1472.63, 1582.1, 1694.71, 1916.0, 2258.56, 2374.33], 
            'magnitude': [0.3072, 0.014, 0.0346, 0.8082, 0.0282, 1.0, 0.361, 0.1252, 0.4124, 0.0985, 0.0625, 0.0769, 0.0637, 0.0982, 0.0323, 0.0289, 0.0518, 0.0842, 0.058, 0.0253, 0.0174, 0.025, 0.0313]
        }, 
        9: {
            'energy': 78.8, 
            'frequency': [109.24, 182.89, 218.47, 234.19, 298.74, 329.36, 393.08, 439.43, 550.32, 660.38, 771.27, 882.99, 995.53, 1108.08, 1222.28, 1336.48, 1449.86, 1565.71, 1681.57, 1799.08, 2277.4, 2399.88, 2523.18], 
            'magnitude': [1.0, 0.0291, 0.5949, 0.0206, 0.0489, 0.6835, 0.0172, 0.6697, 0.2475, 0.2069, 0.0645, 0.0201, 0.0631, 0.0658, 0.1098, 0.0532, 0.027, 0.0662, 0.0939, 0.016, 0.025, 0.02, 0.0206]
        }, 
        10: {
            'energy': 78.7, 
            'frequency': [115.98, 182.59, 204.53, 231.96, 298.57, 347.94, 391.04, 441.98, 464.7, 581.47, 698.23, 815.78, 934.11, 1052.44, 1173.12, 1291.45, 1411.35, 1514.01, 1535.17, 1655.06, 1778.1, 1901.91, 2279.63], 
            'magnitude': [1.0, 0.0155, 0.0124, 0.1156, 0.0229, 0.575, 0.0095, 0.0091, 0.3909, 0.2755, 0.0746, 0.2078, 0.0164, 0.046, 0.0274, 0.0374, 0.0544, 0.0266, 0.0185, 0.0946, 0.0419, 0.0226, 0.0099]
        }, 
        11: {
            'energy': 78.7, 
            'frequency': [123.17, 184.76, 219.63, 246.34, 276.76, 297.54, 342.06, 370.25, 494.16, 618.08, 742.73, 867.38, 993.52, 1120.4, 1245.8, 1374.16, 1503.27, 1609.37, 1630.89, 1760.74, 1892.81, 2024.89, 2429.27], 
            'magnitude': [1.0, 0.0102, 0.0098, 0.3286, 0.0151, 0.0181, 0.0096, 0.1383, 0.068, 0.0735, 0.069, 0.1127, 0.0094, 0.0381, 0.0657, 0.0433, 0.0088, 0.0181, 0.0095, 0.0444, 0.036, 0.0234, 0.0088]
        }
    }, 
    3: {
        0: {
            'energy': 78.6, 
            'frequency': [90.43, 130.0, 184.9, 260.8, 297.13, 342.35, 391.6, 522.41, 544.21, 654.02, 785.63, 918.05, 1051.27, 1184.5, 1318.53, 1453.37, 1589.02, 1725.47, 1862.74, 2000.81, 2140.49, 2566.01, 2710.54], 
            'magnitude': [0.0048, 1.0, 0.0092, 0.2712, 0.0221, 0.007, 0.2661, 0.0546, 0.0054, 0.0348, 0.0172, 0.0551, 0.0142, 0.0843, 0.0781, 0.0067, 0.0516, 0.0185, 0.0109, 0.051, 0.0111, 0.0051, 0.0087]
        }, 
        1: {
            'energy': 78.5, 
            'frequency': [137.64, 182.43, 276.09, 340.43, 415.36, 441.42, 553.0, 692.26, 831.53, 971.61, 1112.51, 1254.22, 1395.93, 1523.8, 1539.27, 1681.8, 1803.96, 1825.95, 2118.33, 2268.18, 2567.08, 3342.42, 3502.86], 
            'magnitude': [0.6225, 0.0118, 1.0, 0.0065, 0.1413, 0.0059, 0.2728, 0.0424, 0.1022, 0.0446, 0.0252, 0.0267, 0.0799, 0.0069, 0.0054, 0.0252, 0.0223, 0.0157, 0.0237, 0.0121, 0.0082, 0.0118, 0.0054]
        }, 
        2: {
            'energy': 78.4, 
            'frequency': [145.8, 183.11, 291.59, 342.8, 394.0, 439.12, 585.79, 732.45, 879.99, 1027.52, 1176.79, 1326.92, 1477.06, 1628.93, 1779.06, 1907.5, 1932.67, 2088.88, 2242.49, 2402.17, 2882.08, 3045.24, 3535.56], 
            'magnitude': [0.9596, 0.0303, 1.0, 0.0203, 0.0138, 0.499, 0.4701, 0.3623, 0.0323, 0.0987, 0.1141, 0.2196, 0.1128, 0.0165, 0.0426, 0.0302, 0.0355, 0.0156, 0.062, 0.1181, 0.0138, 0.0136, 0.0127]
        }, 
        3: {
            'energy': 78.4, 
            'frequency': [135.97, 155.51, 220.94, 285.53, 311.87, 357.76, 402.8, 466.53, 608.44, 623.74, 638.19, 780.95, 937.31, 1096.22, 1254.28, 1414.04, 1573.8, 1737.81, 1900.96, 2064.12, 2231.53, 2398.94, 2568.04], 
            'magnitude': [0.0675, 1.0, 0.0132, 0.0254, 0.972, 0.0232, 0.0203, 0.398, 0.0227, 0.5676, 0.0199, 0.3234, 0.0455, 0.0794, 0.0604, 0.1558, 0.0911, 0.0398, 0.0226, 0.0145, 0.037, 0.0369, 0.0319]
        }, 
        4: {
            'energy': 78.3, 
            'frequency': [137.4, 164.89, 195.42, 283.21, 330.53, 404.58, 494.66, 661.83, 827.48, 993.13, 1161.07, 1330.53, 1498.47, 1670.99, 1843.51, 2016.79, 2191.6, 2368.7, 2546.56, 2720.61, 2907.63, 3282.44, 3466.41], 
            'magnitude': [0.0297, 1.0, 0.0152, 0.0124, 0.1636, 0.0093, 0.2834, 0.0838, 0.2838, 0.1136, 0.0691, 0.1635, 0.0987, 0.1875, 0.0608, 0.049, 0.0287, 0.0202, 0.0261, 0.0332, 0.011, 0.0092, 0.0092]
        }, 
        5: {
            'energy': 78.2, 
            'frequency': [88.31, 137.17, 174.04, 277.78, 314.64, 349.8, 523.84, 555.56, 700.45, 877.06, 1052.82, 1231.14, 1409.47, 1587.8, 1768.7, 1951.31, 2136.5, 2319.97, 2506.87, 2695.48, 2887.53, 3080.43, 3472.23], 
            'magnitude': [0.0045, 0.0203, 0.5237, 0.0048, 0.0154, 1.0, 0.0199, 0.0057, 0.0807, 0.121, 0.0589, 0.1039, 0.0439, 0.0202, 0.0617, 0.0296, 0.0141, 0.0204, 0.0096, 0.0172, 0.0156, 0.0058, 0.0045]
        }, 
        6: {
            'energy': 78.0, 
            'frequency': [87.77, 130.68, 184.31, 254.53, 306.21, 369.6, 429.09, 554.89, 740.17, 800.63, 885.48, 926.43, 1113.67, 1301.88, 1490.1, 1681.23, 1872.37, 2260.5, 2457.49, 2654.48, 2792.95, 2856.34, 3073.81], 
            'magnitude': [0.0074, 0.0306, 0.6424, 0.0067, 0.0189, 1.0, 0.0071, 0.7227, 0.5359, 0.0066, 0.0072, 0.3973, 0.0823, 0.1749, 0.1734, 0.0278, 0.1284, 0.0124, 0.0677, 0.0156, 0.0066, 0.0171, 0.018]
        }, 
        7: {
            'energy': 77.9, 
            'frequency': [133.77, 163.35, 196.15, 313.83, 345.35, 392.29, 454.67, 555.0, 589.08, 727.35, 786.52, 983.95, 1183.31, 1383.32, 1584.61, 1787.19, 1989.76, 2196.84, 2403.92, 2612.93, 2823.87, 3256.68, 3475.98], 
            'magnitude': [0.0192, 0.017, 0.472, 0.0097, 0.0156, 1.0, 0.0093, 0.0096, 0.1972, 0.007, 0.274, 0.0494, 0.095, 0.0963, 0.138, 0.0297, 0.0918, 0.0305, 0.0562, 0.007, 0.0213, 0.015, 0.0172]
        }, 
        8: {
            'energy': 77.8, 
            'frequency': [88.57, 126.53, 166.18, 207.51, 305.36, 343.32, 415.02, 453.82, 550.83, 622.53, 665.55, 726.29, 832.57, 1041.77, 1251.81, 1463.54, 1676.96, 2106.32, 2324.8, 2546.65, 2769.34, 2994.57, 3458.52], 
            'magnitude': [0.0059, 0.0326, 0.0147, 1.0, 0.0109, 0.0133, 0.679, 0.0218, 0.0066, 0.1283, 0.0055, 0.007, 0.0371, 0.2501, 0.0612, 0.2283, 0.1595, 0.0244, 0.0088, 0.0104, 0.0141, 0.0114, 0.0057]
        }, 
        9: {
            'energy': 77.7, 
            'frequency': [127.9, 164.79, 185.28, 219.72, 277.93, 343.51, 404.18, 440.25, 659.97, 727.2, 882.97, 1105.14, 1328.14, 1551.14, 1778.23, 2006.15, 2238.16, 2468.54, 2700.55, 2936.67, 3177.7, 3399.88, 3419.55], 
            'magnitude': [0.0329, 0.0171, 0.0313, 0.5337, 0.0135, 0.0139, 0.0466, 1.0, 0.2338, 0.0141, 0.1409, 0.2496, 0.1362, 0.1022, 0.2519, 0.0176, 0.039, 0.047, 0.0228, 0.02, 0.0126, 0.0137, 0.0229]
        }, 
        10: {
            'energy': 77.5, 
            'frequency': [129.92, 163.51, 185.6, 232.45, 404.79, 423.36, 446.33, 465.78, 699.99, 727.39, 932.44, 1167.54, 1405.29, 1641.27, 1882.56, 2122.96, 2369.55, 2612.61, 2860.08, 3111.97, 3362.98, 3888.86, 4153.12], 
            'magnitude': [0.0205, 0.0088, 0.0117, 0.583, 0.0215, 0.0109, 0.0324, 1.0, 0.0347, 0.0141, 0.1534, 0.0841, 0.0485, 0.0265, 0.0766, 0.0083, 0.0276, 0.0132, 0.0128, 0.0085, 0.0164, 0.0179, 0.0136]
        }, 
        11: {
            'energy': 77.4, 
            'frequency': [122.74, 165.66, 182.83, 247.2, 405.99, 423.16, 467.79, 494.4, 554.49, 742.46, 991.38, 1239.44, 1490.93, 1745.0, 1998.21, 2254.85, 2514.07, 2775.87, 3039.38, 3308.04, 3574.98, 3742.36, 4415.29], 
            'magnitude': [0.02, 0.0156, 0.0173, 0.2617, 0.0276, 0.0146, 0.0464, 1.0, 0.0154, 0.4016, 0.2104, 0.0589, 0.072, 0.055, 0.1188, 0.0398, 0.0581, 0.0116, 0.0108, 0.0384, 0.0161, 0.0113, 0.0095]
        }
    }, 
    4: {
        0: {
            'energy': 77.2, 
            'frequency': [88.77, 130.86, 165.64, 183.03, 233.36, 261.73, 404.49, 456.65, 485.94, 523.46, 548.17, 764.14, 787.02, 1050.58, 1315.06, 1581.36, 1848.58, 2120.38, 2392.18, 2667.63, 2945.84, 3227.7, 3512.31], 
            'magnitude': [0.0155, 0.0266, 0.0158, 0.0208, 0.0277, 1.0, 0.0346, 0.039, 0.0367, 0.7888, 0.0347, 0.0201, 0.2893, 0.2044, 0.185, 0.2726, 0.0214, 0.1492, 0.0233, 0.0516, 0.0205, 0.0304, 0.0381]
        }, 
        1: {
            'energy': 77.1, 
            'frequency': [121.03, 140.94, 166.42, 183.94, 203.84, 230.12, 277.1, 402.91, 457.05, 518.36, 554.99, 636.21, 832.89, 847.22, 1113.17, 1393.45, 1674.53, 1961.18, 2247.84, 2535.28, 2830.7, 3126.11, 3728.87], 
            'magnitude': [0.0142, 0.0106, 0.0067, 0.0078, 0.0064, 0.0103, 1.0, 0.0179, 0.0134, 0.0126, 0.8375, 0.0098, 0.024, 0.0143, 0.1387, 0.1217, 0.0731, 0.0347, 0.0601, 0.0125, 0.0143, 0.0067, 0.009]
        }, 
        2: {
            'energy': 76.9, 
            'frequency': [93.25, 124.62, 231.44, 293.33, 339.96, 404.39, 455.25, 521.38, 587.51, 619.72, 635.83, 662.11, 852.01, 883.38, 1180.1, 1477.67, 1777.78, 2079.58, 2383.93, 2690.83, 3001.96, 3319.03, 3635.25], 
            'magnitude': [0.0083, 0.0147, 0.008, 1.0, 0.0142, 0.0225, 0.0162, 0.0117, 0.1609, 0.0215, 0.0209, 0.0078, 0.0096, 0.1687, 0.2435, 0.1207, 0.0756, 0.0285, 0.051, 0.0179, 0.0211, 0.0101, 0.011]
        }, 
        3: {
            'energy': 76.7, 
            'frequency': [88.11, 136.24, 165.31, 184.38, 204.37, 281.57, 310.64, 404.19, 455.97, 539.53, 554.06, 622.19, 690.31, 783.86, 919.2, 934.64, 1248.91, 1548.65, 1563.18, 1882.0, 2203.53, 2525.07, 2851.15], 
            'magnitude': [0.0054, 0.0119, 0.0053, 0.0085, 0.0069, 0.0162, 0.2557, 0.02, 0.0117, 0.0104, 0.0139, 1.0, 0.0087, 0.006, 0.0065, 0.1067, 0.0562, 0.0061, 0.0878, 0.0816, 0.0678, 0.0392, 0.0066]
        }, 
        4: {
            'energy': 76.6, 
            'frequency': [88.15, 123.4, 184.3, 283.67, 329.34, 403.07, 455.95, 471.98, 551.31, 635.45, 659.49, 679.52, 746.83, 952.77, 967.2, 991.24, 1323.78, 1657.94, 1994.49, 2335.05, 2677.22, 3024.19, 3374.37], 
            'magnitude': [0.0103, 0.0253, 0.0096, 0.0193, 0.7751, 0.0308, 0.0118, 0.0123, 0.0164, 0.0412, 1.0, 0.0267, 0.0107, 0.0076, 0.008, 0.1637, 0.1186, 0.1154, 0.1174, 0.0678, 0.062, 0.0278, 0.0109]
        }, 
        5: {
            'energy': 76.4, 
            'frequency': [46.54, 117.73, 171.57, 203.51, 231.8, 281.99, 303.9, 334.92, 349.53, 403.37, 447.17, 473.64, 553.95, 619.66, 661.63, 699.05, 1049.49, 1091.47, 1400.84, 1757.67, 2114.49, 2474.97, 2841.84], 
            'magnitude': [0.0051, 0.0137, 0.0046, 0.006, 0.0057, 0.008, 0.0078, 0.0216, 1.0, 0.023, 0.0094, 0.007, 0.0073, 0.0071, 0.0141, 0.1599, 0.0288, 0.0045, 0.0299, 0.0795, 0.0346, 0.0206, 0.0135]
        }, 
        6: {
            'energy': 76.2, 
            'frequency': [88.25, 117.67, 135.7, 185.99, 231.54, 282.78, 329.27, 370.08, 450.74, 474.46, 551.32, 609.2, 665.19, 690.81, 741.1, 800.89, 1113.08, 1487.9, 1863.67, 2244.19, 2627.55, 3015.66, 3408.51], 
            'magnitude': [0.0112, 0.0253, 0.0246, 0.0096, 0.0101, 0.0181, 0.0163, 1.0, 0.0143, 0.0162, 0.0175, 0.0146, 0.029, 0.0182, 0.1734, 0.0113, 0.1586, 0.0646, 0.0805, 0.1001, 0.0788, 0.044, 0.0134]
        }, 
        7: {
            'energy': 76.0, 
            'frequency': [119.83, 187.23, 282.72, 321.1, 354.8, 391.31, 410.03, 451.22, 473.69, 553.26, 607.56, 665.6, 687.13, 762.96, 785.43, 820.06, 1179.55, 1577.41, 1976.21, 2377.81, 2785.97, 3197.88, 3616.34], 
            'magnitude': [0.0141, 0.0102, 0.0108, 0.0153, 0.0289, 1.0, 0.0802, 0.0126, 0.0181, 0.0123, 0.0119, 0.021, 0.0111, 0.025, 0.068, 0.0165, 0.1508, 0.0656, 0.1179, 0.0507, 0.013, 0.0162, 0.01]
        }, 
        8: {
            'energy': 75.8, 
            'frequency': [139.43, 234.41, 282.34, 319.81, 355.54, 415.66, 430.48, 474.05, 554.22, 609.99, 662.27, 763.36, 806.93, 832.2, 849.63, 867.93, 1249.61, 1670.5, 2093.13, 2520.12, 2953.22, 3394.15, 3840.31], 
            'magnitude': [0.0095, 0.0055, 0.0051, 0.0055, 0.0111, 1.0, 0.0339, 0.0116, 0.0064, 0.0059, 0.0125, 0.0058, 0.0158, 0.3195, 0.0047, 0.0052, 0.1039, 0.0619, 0.0302, 0.019, 0.0437, 0.0125, 0.0132]
        }, 
        9: {
            'energy': 75.6, 
            'frequency': [89.07, 121.89, 140.64, 188.46, 282.22, 327.22, 354.41, 379.72, 403.16, 420.04, 438.79, 475.36, 554.12, 608.5, 661.94, 826.02, 880.4, 1321.07, 1767.36, 2217.4, 2671.2, 3128.74, 3590.98], 
            'magnitude': [0.0148, 0.0303, 0.0243, 0.0172, 0.013, 0.0188, 0.0182, 0.0285, 0.0368, 0.0844, 1.0, 0.0366, 0.0149, 0.0156, 0.0259, 0.0132, 0.431, 0.2178, 0.112, 0.05, 0.0442, 0.0467, 0.0202]
        }, 
        10: {
            'energy': 75.3, 
            'frequency': [88.09, 130.53, 185.79, 210.61, 229.83, 282.68, 354.75, 380.38, 420.42, 436.44, 465.27, 553.35, 609.41, 663.06, 684.68, 881.68, 931.33, 1399.8, 1875.48, 2352.75, 2832.43, 3325.73, 3818.22], 
            'magnitude': [0.0191, 0.0291, 0.0192, 0.0181, 0.023, 0.0197, 0.0221, 0.0347, 0.0551, 0.0593, 1.0, 0.017, 0.0244, 0.0385, 0.023, 0.0185, 0.2952, 0.0944, 0.1229, 0.0604, 0.0705, 0.0254, 0.0287]
        }, 
        11: {
            'energy': 75.1, 
            'frequency': [88.35, 121.14, 137.54, 232.27, 284.19, 339.75, 378.91, 421.72, 438.12, 459.07, 494.59, 584.77, 610.27, 630.31, 660.37, 684.05, 990.09, 1027.44, 1488.33, 1991.12, 2500.28, 3014.92, 3538.65], 
            'magnitude': [0.0167, 0.0096, 0.0102, 0.0063, 0.0067, 0.008, 0.012, 0.0144, 0.0121, 0.0204, 1.0, 0.0063, 0.0073, 0.0071, 0.0137, 0.0086, 0.3101, 0.0093, 0.0652, 0.0765, 0.04, 0.0288, 0.0177]
        }
    }, 
    5: {
        0: {
            'energy': 74.9, 
            'frequency': [88.39, 124.65, 176.02, 274.23, 339.96, 378.49, 422.31, 438.17, 458.57, 509.18, 524.29, 549.98, 594.55, 664.05, 682.19, 729.78, 992.68, 1050.85, 1078.8, 1580.43, 2113.04, 2651.68, 3758.44], 
            'magnitude': [0.0113, 0.0148, 0.008, 0.0073, 0.0099, 0.0113, 0.0132, 0.0076, 0.012, 0.061, 1.0, 0.044, 0.0093, 0.0125, 0.0104, 0.0069, 0.0077, 0.1392, 0.0084, 0.0455, 0.0305, 0.019, 0.0113]
        }, 
        1: {
            'energy': 74.6, 
            'frequency': [88.62, 124.38, 213.78, 230.89, 273.64, 340.5, 377.81, 422.12, 468.77, 486.65, 519.3, 555.83, 578.38, 616.47, 645.23, 662.34, 682.55, 718.31, 1114.0, 1672.95, 2242.0, 2814.93, 3396.42], 
            'magnitude': [0.0124, 0.013, 0.0079, 0.0088, 0.0083, 0.01, 0.0114, 0.012, 0.0097, 0.0097, 0.0373, 1.0, 0.0227, 0.0137, 0.0108, 0.0146, 0.0109, 0.0096, 0.1247, 0.0491, 0.0839, 0.0394, 0.014]
        }, 
        2: {
            'energy': 74.4, 
            'frequency': [88.04, 121.63, 136.15, 273.21, 340.37, 380.31, 422.06, 469.26, 587.26, 634.46, 680.75, 728.85, 761.53, 781.5, 1173.61, 1769.94, 2365.37, 2972.6, 3589.81], 
            'magnitude': [0.0582, 0.0394, 0.0332, 0.0309, 0.0438, 0.0431, 0.0442, 0.0336, 1.0, 0.0627, 0.0545, 0.036, 0.0388, 0.0349, 0.269, 0.1754, 0.1868, 0.1066, 0.0392]
        }, 
        3: {
            'energy': 74.1, 
            'frequency': [88.12, 137.73, 213.26, 231.04, 340.63, 378.4, 423.57, 533.16, 554.64, 573.15, 599.81, 622.76, 651.64, 673.12, 706.44, 761.98, 783.45, 825.66, 1246.26, 1875.69, 2514.0, 3161.94, 3820.24], 
            'magnitude': [0.0098, 0.0109, 0.0076, 0.0069, 0.0109, 0.0088, 0.0084, 0.0159, 0.0222, 0.0151, 0.0206, 1.0, 0.0219, 0.023, 0.0117, 0.0138, 0.0132, 0.0076, 0.065, 0.0461, 0.0166, 0.0138, 0.0104]
        }, 
        4: {
            'energy': 73.8, 
            'frequency': [88.15, 122.33, 184.39, 216.78, 321.12, 340.9, 378.68, 420.96, 519.9, 534.29, 554.08, 599.96, 659.32, 706.09, 733.08, 760.06, 785.25, 823.93, 841.92, 1321.34, 1986.96, 2664.27, 3352.37], 
            'magnitude': [0.0131, 0.0154, 0.0077, 0.0069, 0.0066, 0.0081, 0.0099, 0.0095, 0.0139, 0.0141, 0.0183, 0.007, 1.0, 0.0226, 0.0171, 0.0229, 0.0152, 0.0108, 0.0098, 0.1926, 0.0471, 0.0164, 0.0195]
        }, 
        5: {
            'energy': 73.6, 
            'frequency': [88.36, 126.34, 186.22, 209.59, 231.5, 340.31, 424.29, 519.22, 554.28, 599.55, 636.07, 698.14, 781.39, 823.75, 1352.46, 1398.47, 1482.45, 2101.72, 2817.39, 3545.47], 
            'magnitude': [0.0271, 0.0293, 0.0232, 0.0162, 0.0167, 0.0162, 0.0144, 0.0221, 0.0262, 0.0187, 0.0562, 1.0, 0.0405, 0.0309, 0.0136, 0.1633, 0.0159, 0.1498, 0.0377, 0.0182]
        }, 
        6: {
            'energy': 73.3, 
            'frequency': [46.99, 89.03, 118.71, 140.35, 161.37, 184.24, 228.76, 341.9, 407.43, 518.72, 550.25, 599.1, 634.95, 668.96, 707.29, 739.44, 780.25, 1007.15, 1485.07, 1524.63, 2234.4, 2996.1, 3772.63], 
            'magnitude': [0.0086, 0.0157, 0.0175, 0.0156, 0.0103, 0.0141, 0.02, 0.0113, 0.0108, 0.0136, 0.0129, 0.0122, 0.0235, 0.0465, 0.0435, 1.0, 0.0675, 0.0115, 0.1062, 0.0089, 0.1094, 0.0605, 0.0241]
        }, 
        7: {
            'energy': 73.0, 
            'frequency': [88.18, 125.75, 160.26, 184.79, 210.1, 391.82, 407.92, 519.87, 548.24, 654.82, 733.04, 783.64, 821.98, 858.79, 892.52, 907.09, 968.43, 997.57, 1571.12, 2366.26, 3173.67, 3996.42], 
            'magnitude': [0.0135, 0.0111, 0.0054, 0.008, 0.0046, 0.0052, 0.0066, 0.0071, 0.0059, 0.0141, 0.0284, 1.0, 0.04, 0.0205, 0.0109, 0.0097, 0.0069, 0.0048, 0.0729, 0.0173, 0.0058, 0.0062]
        }, 
        8: {
            'energy': 72.7, 
            'frequency': [88.8, 117.86, 184.06, 209.09, 229.27, 326.95, 378.62, 407.68, 518.28, 599.01, 634.53, 653.91, 668.44, 706.38, 731.41, 763.7, 803.26, 832.32, 858.96, 1034.95, 1667.87, 2511.49, 3368.02], 
            'magnitude': [0.0099, 0.0169, 0.0169, 0.0141, 0.0148, 0.0072, 0.0076, 0.0095, 0.0082, 0.0086, 0.013, 0.0195, 0.0199, 0.0063, 0.0167, 0.0413, 0.0437, 1.0, 0.0589, 0.0092, 0.072, 0.0277, 0.0132]
        }, 
        9: {
            'energy': 72.4, 
            'frequency': [88.67, 125.56, 158.9, 185.15, 209.27, 229.13, 292.26, 320.64, 339.79, 380.23, 408.6, 517.85, 654.05, 732.08, 764.71, 781.02, 800.89, 880.34, 939.22, 963.33, 1008.02, 1764.93, 3572.42], 
            'magnitude': [0.0129, 0.0148, 0.0072, 0.0116, 0.0084, 0.0076, 0.0055, 0.0059, 0.0059, 0.0058, 0.0073, 0.0068, 0.0142, 0.0109, 0.018, 0.0123, 0.0158, 1.0, 0.0105, 0.0064, 0.0085, 0.0414, 0.0058]
        }, 
        10: {
            'energy': 72.1, 
            'frequency': [88.79, 108.98, 125.12, 142.88, 162.25, 186.47, 209.07, 229.25, 313.2, 407.65, 599.77, 623.98, 652.24, 729.73, 763.63, 801.57, 858.08, 884.72, 901.67, 931.54, 956.56, 1868.73, 2818.83], 
            'magnitude': [0.0169, 0.0172, 0.0306, 0.0198, 0.0155, 0.0295, 0.0224, 0.0223, 0.0125, 0.0127, 0.0139, 0.0168, 0.0236, 0.0148, 0.0224, 0.0189, 0.0275, 0.0287, 0.0322, 1.0, 0.0366, 0.08, 0.0262]
        }, 
        11: {
            'energy': 71.8, 
            'frequency': [88.17, 124.68, 202.16, 217.3, 408.78, 598.47, 621.63, 653.69, 728.5, 763.23, 781.05, 804.2, 855.85, 905.73, 936.9, 961.83, 987.66, 1016.16, 1033.08, 1054.46, 1253.06, 1983.34, 2989.7], 
            'magnitude': [0.0207, 0.0469, 0.0165, 0.0205, 0.0184, 0.0184, 0.0198, 0.0253, 0.0184, 0.0219, 0.014, 0.0177, 0.0209, 0.0177, 0.0336, 0.061, 1.0, 0.0353, 0.0165, 0.0213, 0.0145, 0.211, 0.0297]
        }
    }, 
    6: {
        0: {
            'energy': 71.4, 
            'frequency': [89.01, 125.5, 200.27, 395.2, 410.33, 598.14, 623.06, 647.99, 805.53, 854.49, 937.27, 960.41, 982.66, 1012.92, 1049.42, 1078.79, 1095.7, 1119.74, 1141.1, 1179.37, 1255.03, 2103.29, 3172.29], 
            'magnitude': [0.0278, 0.0307, 0.0239, 0.0245, 0.027, 0.0266, 0.0273, 0.0372, 0.023, 0.0243, 0.0258, 0.0416, 0.0414, 0.0548, 1.0, 0.0516, 0.0497, 0.0353, 0.0253, 0.0274, 0.0307, 0.1112, 0.0569]
        }, 
        1: {
            'energy': 71.1, 
            'frequency': [131.0, 184.25, 209.81, 230.05, 408.97, 598.55, 623.04, 647.54, 854.16, 960.66, 981.96, 1008.59, 1054.38, 1079.94, 1112.96, 1138.52, 1182.19, 1197.1, 1215.2, 1254.61, 1317.45, 2229.12, 3367.64], 
            'magnitude': [0.0428, 0.0352, 0.03, 0.0295, 0.0252, 0.0279, 0.0292, 0.0348, 0.0247, 0.0334, 0.0258, 0.0247, 0.0346, 0.0508, 1.0, 0.0722, 0.0323, 0.027, 0.0288, 0.0408, 0.0259, 0.2043, 0.0894]
        }, 
        2: {
            'energy': 70.8, 
            'frequency': [88.78, 123.43, 184.07, 228.46, 410.36, 440.67, 507.8, 578.18, 615.0, 648.56, 663.72, 725.44, 854.28, 960.39, 1124.97, 1154.2, 1172.61, 1196.43, 1213.75, 1254.89, 1272.22, 1314.45, 2352.79], 
            'magnitude': [0.0385, 0.0392, 0.0266, 0.0237, 0.025, 0.0232, 0.0208, 0.0217, 0.0285, 0.034, 0.0245, 0.0223, 0.0225, 0.0269, 0.0279, 0.0805, 1.0, 0.0623, 0.0631, 0.0525, 0.0335, 0.0303, 0.1371]
        }, 
        3: {
            'energy': 70.4, 
            'frequency': [100.03, 131.99, 160.87, 184.58, 204.18, 225.83, 291.83, 320.7, 395.98, 410.42, 438.26, 506.32, 554.78, 593.97, 646.56, 664.09, 839.39, 906.42, 960.04, 1019.85, 1055.94, 1245.68, 2498.58], 
            'magnitude': [0.0798, 0.0807, 0.0402, 0.0777, 0.0583, 0.0564, 0.0278, 0.0288, 0.0315, 0.0361, 0.0291, 0.0292, 0.0285, 0.042, 0.0491, 0.0429, 0.0337, 0.0261, 0.0311, 0.0284, 0.0281, 1.0, 0.0639]
        }, 
        4: {
            'energy': 70.1, 
            'frequency': [103.44, 123.72, 162.26, 185.59, 231.22, 411.74, 510.11, 554.73, 610.51, 644.99, 688.6, 724.09, 837.68, 852.89, 956.33, 1055.71, 1148.0, 1222.03, 1318.37, 1398.49, 1490.78, 2652.98, 4012.93], 
            'magnitude': [0.0226, 0.0139, 0.012, 0.0107, 0.0076, 0.008, 0.0083, 0.0106, 0.0134, 0.0144, 0.0078, 0.0075, 0.0088, 0.0087, 0.008, 0.0115, 0.0075, 0.0115, 1.0, 0.0168, 0.011, 0.0143, 0.0109]
        }, 
        5: {
            'energy': 69.7, 
            'frequency': [46.33, 88.32, 125.97, 183.89, 211.4, 230.22, 414.83, 445.24, 509.67, 555.28, 577.0, 609.57, 663.87, 721.79, 839.79, 954.18, 1056.98, 1312.54, 1350.19, 1398.69, 1478.33, 1494.25, 2807.52], 
            'magnitude': [0.0283, 0.0263, 0.0289, 0.0299, 0.0156, 0.0104, 0.0345, 0.0354, 0.0301, 0.0339, 0.0392, 0.0146, 0.0253, 0.0316, 0.0288, 0.0351, 0.034, 0.0658, 0.0638, 1.0, 0.072, 0.0688, 0.0768]
        }, 
        6: {
            'energy': 69.3, 
            'frequency': [46.57, 88.6, 131.2, 161.87, 218.66, 416.31, 445.28, 470.83, 554.32, 576.47, 610.55, 647.47, 663.94, 687.22, 720.73, 959.27, 1037.08, 1056.39, 1423.86, 1452.82, 1480.09, 1503.94, 2982.89], 
            'magnitude': [0.0163, 0.0177, 0.024, 0.0177, 0.0151, 0.0153, 0.0174, 0.0145, 0.0163, 0.021, 0.0233, 0.0248, 0.0275, 0.0149, 0.0169, 0.0153, 0.0179, 0.0213, 0.0171, 0.0329, 1.0, 0.0755, 0.1353]
        }, 
        7: {
            'energy': 69.0, 
            'frequency': [63.11, 99.98, 131.89, 160.25, 184.36, 416.23, 444.59, 509.11, 556.62, 576.48, 606.26, 646.67, 662.27, 718.29, 1036.66, 1095.52, 1116.79, 1440.84, 1572.72, 1596.12, 1618.11, 1665.61, 3166.01], 
            'magnitude': [0.0352, 0.0113, 0.0133, 0.0335, 0.0202, 0.0107, 0.0112, 0.032, 0.0382, 0.0125, 0.0162, 0.0203, 0.0234, 0.0158, 0.012, 0.0317, 0.0234, 0.0341, 1.0, 0.0605, 0.039, 0.0733, 0.1129]
        }, 
        8: {
            'energy': 68.6, 
            'frequency': [99.81, 131.33, 160.23, 183.87, 399.26, 416.33, 442.6, 508.26, 554.23, 576.56, 605.45, 648.79, 663.24, 714.46, 957.43, 1033.6, 1116.34, 1213.53, 1624.6, 1666.63, 1704.72, 1765.13, 3359.53], 
            'magnitude': [0.0373, 0.0297, 0.0109, 0.0189, 0.0376, 0.0119, 0.0149, 0.0327, 0.0405, 0.0467, 0.0511, 0.047, 0.071, 0.0437, 0.0349, 0.0412, 0.0486, 0.0324, 0.0397, 1.0, 0.0486, 0.0559, 0.0355]
        }, 
        9: {
            'energy': 68.2, 
            'frequency': [61.68, 100.32, 131.53, 159.76, 182.8, 329.18, 399.78, 416.12, 442.88, 555.08, 575.89, 606.35, 645.74, 662.83, 686.61, 714.1, 959.32, 1030.65, 1214.19, 1502.51, 1714.29, 1767.04, 1786.36], 
            'magnitude': [0.0281, 0.0202, 0.0243, 0.0276, 0.0274, 0.0295, 0.0294, 0.0288, 0.0855, 0.0756, 0.0857, 0.1034, 0.1008, 0.1416, 0.0718, 0.0871, 0.082, 0.0905, 0.0752, 0.0709, 0.1421, 1.0, 0.186]
        }, 
        10: {
            'energy': 67.8, 
            'frequency': [106.28, 131.44, 183.36, 227.98, 399.18, 416.21, 443.8, 508.7, 554.95, 576.05, 604.44, 662.86, 719.65, 958.18, 1029.58, 1066.09, 1118.01, 1145.6, 1226.73, 1502.59, 1799.53, 1817.38, 1871.74], 
            'magnitude': [0.0251, 0.0258, 0.0265, 0.0227, 0.025, 0.0233, 0.0211, 0.016, 0.0482, 0.0582, 0.07, 0.0928, 0.0494, 0.05, 0.0696, 0.0531, 0.0473, 0.045, 0.0416, 0.0496, 0.0447, 0.0465, 1.0]
        }, 
        11: {
            'energy': 67.4, 
            'frequency': [88.91, 105.2, 131.14, 183.0, 229.67, 396.37, 416.37, 442.3, 554.18, 575.66, 603.07, 664.57, 720.87, 926.84, 960.18, 1020.93, 1043.9, 1226.89, 1502.5, 1909.98, 1939.62, 1988.52, 2014.45], 
            'magnitude': [0.0106, 0.0127, 0.0117, 0.028, 0.0287, 0.0289, 0.0325, 0.0282, 0.0277, 0.0316, 0.0397, 0.0495, 0.0311, 0.0362, 0.035, 0.0464, 0.0467, 0.0315, 0.0356, 0.0273, 0.0296, 1.0, 0.0749]
        }
    }, 
    7: {
        0: {
            'energy': 67.0, 
            'frequency': [2053.07, 2108.74, 2125.87], 
            'magnitude': [0.0501, 1.0, 0.0897]
        }, 
        1: {
            'energy': 66.6, 
            'frequency': [2239.44], 
            'magnitude': [1.0]
        }, 
        2: {
            'energy': 66.2, 
            'frequency': [2326.6, 2359.59, 2383.94], 
            'magnitude': [0.0772, 1.0, 0.1719]
        }, 
        3: {
            'energy': 65.7, 
            'frequency': [2405.12, 2438.08, 2462.36, 2489.25, 2540.42], 
            'magnitude': [0.1222, 0.0771, 0.076, 1.0, 0.0903]
        }, 
        4: {
            'energy': 65.3, 
            'frequency': [2605.76, 2652.25, 2674.72, 2707.26], 
            'magnitude': [0.1378, 0.234, 1.0, 0.1426]
        }, 
        5: {
            'energy': 64.8, 
            'frequency': [2809.5], 
            'magnitude': [1.0]
        }, 
        6: {
            'energy': 64.4, 
            'frequency': [2809.5], 
            'magnitude': [1.0]
        }, 
        7: {
            'energy': 63.9, 
            'frequency': [2991.06, 3013.6, 3030.08], 
            'magnitude': [1.0, 0.1354, 0.0894]
        }, 
        8: {
            'energy': 63.4, 
            'frequency': [3161.81], 
            'magnitude': [1.0]
        }, 
        9: {
            'energy': 63.0, 
            'frequency': [3284.41, 3351.25, 3384.3], 
            'magnitude': [0.1256, 1.0, 0.1925]
        }, 
        10: {
            'energy': 62.5, 
            'frequency': [3563.73, 3605.13, 3629.05], 
            'magnitude': [1.0, 0.2174, 0.2446]
        }, 
        11: {
            'energy': 62.0, 
            'frequency': [3777.26, 3816.12], 
            'magnitude': [1.0, 0.1689]
        }
    }, 
    8: {
        0: {
            'energy': 61.5, 
            'frequency': [4196.9, 4250.84], 
            'magnitude': [0.4817, 1.0]
        }
    }
}