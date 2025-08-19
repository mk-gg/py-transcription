# Real-Time Audio Transcription

A Python application for real-time audio transcription using OpenAI's Whisper model. Captures system audio (like video calls, presentations, or media playback) and provides live transcription.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

<img src="https://github.com/mk-gg/py-transcription/blob/main/preview/preview.gif" alt="Master" style="width:100%; height:auto;">

## Features

- 🎤 **Real-time system audio capture** - Transcribe audio from any application
- 🧠 **Multiple Whisper models** - Choose between speed and accuracy
- 💾 **Session management** - Start, pause, resume, and save transcriptions
- 💿 **Export functionality** - Save transcriptions to text files



## Installation

### Prerequisites

- Python 3.8 or higher
- Git (for cloning the repository)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/mk-gg/py-transcription.git
   cd py-transcription
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

### Platform-Specific Setup

#### Windows
- No additional setup required
- Uses built-in WASAPI for system audio capture
- Recommended: Install Microsoft Visual C++ Redistributable

#### macOS
```bash
# Install PortAudio via Homebrew (recommended)
brew install portaudio

# Alternative: Install via MacPorts
sudo port install portaudio
```

#### Linux (Ubuntu/Debian)
```bash
# Install system dependencies
sudo apt update
sudo apt install python3-dev portaudio19-dev python3-pyaudio

# For newer systems with PipeWire
sudo apt install pipewire-pulse
```

#### Linux (Fedora/RHEL)
```bash
# Install system dependencies
sudo dnf install python3-devel portaudio-devel
```

## Configuration

### Audio Settings
The application uses optimized audio settings by default:

```python
# Default configuration
SAMPLE_RATE = 16000    # Whisper's native sample rate
CHANNELS = 1           # Mono audio for efficiency
CHUNK_SIZE = 1024      # Balance between latency and processing
BUFFER_DURATION = 2.0  # Seconds of audio to buffer
```

### Performance Tuning

#### For Real-Time Performance:
- Use "Tiny" or "Base" models for faster processing
- Ensure your system isn't running CPU-intensive tasks
- Close unnecessary applications to free up resources

#### For Maximum Accuracy:
- Use "Large" or "Turbo" models
- Allow longer processing times
- Consider offline processing for best results

## Troubleshooting

### Common Issues

#### "No suitable audio device found"
- **Windows**: Enable "Stereo Mix" or "What U Hear" in recording devices
- **macOS**: Check System Preferences > Security & Privacy > Microphone
- **Linux**: Verify PulseAudio/PipeWire is running and configured

#### "Failed to start audio capture"
- Check if another application is using the audio device
- Try running as administrator (Windows) or with sudo (Linux)
- Verify the selected device supports recording

#### "Transcription error" or poor accuracy
- Ensure audio levels are adequate (not too quiet or distorted)
- Try a different Whisper model
- Check for background noise or interference
- Verify the audio source is clear and in English

#### High CPU usage
- Switch to a smaller Whisper model (Tiny or Base)
- Close other resource-intensive applications
- Check that hardware acceleration is available

### Performance Optimization

#### Reduce Latency:
```python
# Modify these values in the code for lower latency
BUFFER_DURATION = 1.0    # Shorter buffer (less accuracy)
TRANSCRIPTION_INTERVAL = 1.0  # More frequent transcription
```

#### Improve Accuracy:
```python
# Use longer buffers for better context
BUFFER_DURATION = 3.0    # Longer buffer (higher latency)
```


## Limitations

- **Language Support**: Currently optimized for English (Whisper supports 99 languages)
- **Platform Audio**: System audio capture methods vary by platform
- **Model Size**: Larger models require significant RAM and processing power
- **Real-time Constraints**: Processing speed depends on hardware capabilities
- **Audio Quality**: Transcription accuracy depends on input audio clarity


## Contributing

Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest features.


## Acknowledgments

- **OpenAI** for the Whisper model
- **sounddevice** library for robust cross-platform audio capture
- **DearPyGui** for the GUI framework
- **librosa** for audio processing utilities

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
