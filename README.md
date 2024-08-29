![AudioSigPy Banner](https://raw.githubusercontent.com/wjonasreger/AudioSigPy/main/imgs/banner.png)

# AudioSigPy

## Welcome to AudioSigPy
**AudioSigPy** is a compact audio and speech signal processing package designed for exploration and experimentation with sound. Enjoy discovering what you can create!

## Why Choose AudioSigPy?
**AudioSigPy** is perfect for those interested in audio processing and sound synthesis. It's a fun tool for exploring the world of sound.

## Key Features
- **Speech Analysis:** Analyze speech patterns.
- **Cochlear Implant Simulation:** Experiment with hearing tech simulations.
- **Sound Synthesis:** Create unique audio profiles and musical sounds.
- **Noise Generation:** Produce waves, filter noise, and more.
- **.WAV Management:** Manage .WAV files for basic creation and editing.

## Quick Setup
Getting started with **AudioSigPy** is straightforward:

1. Clone the repository:
   ```sh
   git clone https://github.com/wjonasreger/AudioSigPy.git
   cd AudioSigPy
   ```

2. Install the package:
   ```sh
   pip install wheel
   python3 setup.py bdist_wheel sdist
   pip install .
   ```

## Usage

Getting started with **AudioSigPy** is simple. Here's a basic example to help you begin:

```py
import numpy as np
import AudioSigPy as asp

# Create a Waveform object
wf = asp.Waveform("your-waveform-name")

# Read an audio file (.wav)
wf.read("path-to-your-wav-file")

# Visualize the waveform data
wf.plot_waveform()
```

## Examples

Explore the diverse capabilities of **AudioSigPy** with these example usage areas:
- **Music**: Delve into sound synthesis, experiment with a talkbox effect, and generate music.
- **Speech**: Conduct speech analysis and predict who's speaking.
- **Hearing**: Simulate cochlear implants and deaf experiences, or create stereo hearing experiences tailored for languages and lyrics.

## Compatibility
**AudioSigPy** is designed for macOS. Please note that no future updates are planned for other operating systems.

## License
**AudioSigPy** is open for use with proper credit. Contributions and forks are welcome. While the project won’t expand beyond 2024, contributions are always encouraged.

## Creator
Created by **W. Jonas Reger** as a fun experiment in audio processing, inspired by a desire to hear ["Für Elise" with a jazz swing](https://github.com/wjonasreger/fur_elise_swing) on piano.