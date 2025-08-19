#!/usr/bin/env python3
"""
Real-time Audio Transcription Application
Uses DearPyGui for the interface and Whisper for transcription
"""

import asyncio
import threading
import time
import queue
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum
import json

import numpy as np
import sounddevice as sd
import whisper
import dearpygui.dearpygui as dpg
from collections import deque
import librosa

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SessionState(Enum):
    INACTIVE = "inactive"
    RUNNING_ACTIVE = "running_active" 
    RUNNING_PAUSED = "running_paused"


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    buffer_duration: float = 2.0  # seconds
    device_type: str = "wasapi"  # For Windows loopback


@dataclass
class TranscriptWord:
    text: str
    start: float
    end: float
    confidence: float = 0.0
    is_final: bool = False


class AudioCaptureSystem:
    """Handles real-time audio capture with system audio loopback"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.is_recording = False
        # Increased queue size and made it unbounded for better performance
        self.audio_queue = queue.Queue(maxsize=500)
        self.stream = None
        self.device_id = None
        self.dropped_frames = 0
        
    def get_system_audio_device(self) -> Optional[int]:
        """Find system audio loopback device (Windows WASAPI)"""
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                # Look for WASAPI loopback devices
                if ('wasapi' in device['name'].lower() or 
                    'loopback' in device['name'].lower() or
                    'speakers' in device['name'].lower() or
                    'what u hear' in device['name'].lower()):
                    if device['max_input_channels'] > 0:
                        logger.info(f"Found system audio device: {device['name']}")
                        return i
            
            # Fallback to default input device
            default_device = sd.query_devices(kind='input')
            logger.info(f"Using default input device: {default_device['name']}")
            return sd.default.device[0]
            
        except Exception as e:
            logger.error(f"Error finding audio device: {e}")
            return None
    
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio stream"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        if self.is_recording:
            # Convert to mono if needed
            if indata.shape[1] > 1:
                audio_data = np.mean(indata, axis=1)
            else:
                audio_data = indata[:, 0]
            
            # Put audio data in queue for processing
            try:
                self.audio_queue.put_nowait(audio_data.copy())
            except queue.Full:
                # Drop oldest frames to make room for new ones
                try:
                    # Remove oldest frames (up to 10)
                    for _ in range(min(10, self.audio_queue.qsize())):
                        self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(audio_data.copy())
                    self.dropped_frames += 1
                    if self.dropped_frames % 100 == 0:  # Log every 100 drops
                        logger.warning(f"Dropped {self.dropped_frames} audio frames due to processing lag")
                except queue.Empty:
                    pass
    
    def start_capture(self) -> bool:
        """Start audio capture"""
        try:
            self.device_id = self.get_system_audio_device()
            if self.device_id is None:
                logger.error("No suitable audio device found")
                return False
            
            self.is_recording = True
            self.stream = sd.InputStream(
                device=self.device_id,
                channels=self.config.channels,
                samplerate=self.config.sample_rate,
                callback=self.audio_callback,
                blocksize=self.config.chunk_size,
                dtype=np.float32
            )
            self.stream.start()
            logger.info("Audio capture started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            return False
    
    def stop_capture(self):
        """Stop audio capture"""
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        logger.info("Audio capture stopped")
    
    def get_audio_chunk(self) -> Optional[np.ndarray]:
        """Get audio chunk from queue"""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None


class WhisperTranscriber:
    """Handles Whisper-based transcription"""
    
    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self.model = None
        self.is_loaded = False
        self.available_models = ["tiny", "base", "small", "medium", "large", "turbo", "large-v3"]
        self.model_info = {
            "tiny": {"size": "~39MB", "speed": "Fastest", "accuracy": "Basic"},
            "base": {"size": "~140MB", "speed": "Fast", "accuracy": "Good"},
            "small": {"size": "~460MB", "speed": "Moderate", "accuracy": "Better"},
            "medium": {"size": "~1.5GB", "speed": "Slow", "accuracy": "High"},
            "large": {"size": "~2.9GB", "speed": "Slowest", "accuracy": "Best"},
            "turbo": {"size": "~1.5GB", "speed": "Very Fast", "accuracy": "High"},
            "large-v3": {"size": "~2.9GB", "speed": "Slow", "accuracy": "Excellent"}
        }
        
    def load_model(self) -> bool:
        """Load Whisper model"""
        try:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name)
            self.is_loaded = True
            logger.info("Whisper model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            return False
    
    def change_model(self, new_model_name: str) -> bool:
        """Change to a different Whisper model"""
        if new_model_name == self.model_name and self.is_loaded:
            return True  # Already using this model
        
        if new_model_name not in self.available_models:
            logger.error(f"Invalid model: {new_model_name}")
            return False
        
        # Clear current model
        self.model = None
        self.is_loaded = False
        self.model_name = new_model_name
        
        # Load new model
        return self.load_model()
    
    def get_model_info(self, model_name: str) -> Dict[str, str]:
        """Get information about a model"""
        return self.model_info.get(model_name, {"size": "Unknown", "speed": "Unknown", "accuracy": "Unknown"})
    
    def transcribe_audio(self, audio_data: np.ndarray) -> Optional[Dict[str, Any]]:
        """Transcribe audio data"""
        if not self.is_loaded or self.model is None:
            return None
            
        try:
            # Ensure audio is the right format for Whisper
            if len(audio_data) < 1600:  # Less than 0.1 seconds at 16kHz
                return None
                
            # Normalize audio
            audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Transcribe
            result = self.model.transcribe(
                audio_data,
                language='en',
                task='transcribe',
                fp16=False,
                verbose=False
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None


class TranscriptionSession:
    """Manages transcription session state and processing"""
    
    def __init__(self, audio_config: AudioConfig):
        self.state = SessionState.INACTIVE
        self.audio_config = audio_config
        self.audio_capture = AudioCaptureSystem(audio_config)
        self.transcriber = WhisperTranscriber()
        
        # Audio processing
        self.audio_buffer = deque(maxlen=int(audio_config.sample_rate * audio_config.buffer_duration))
        self.processing_thread = None
        self.stop_event = threading.Event()
        
        # Transcription results
        self.partial_transcription = ""
        self.final_transcriptions: List[TranscriptWord] = []
        self.transcription_callback = None
        
    def set_transcription_callback(self, callback):
        """Set callback for transcription updates"""
        self.transcription_callback = callback
        
    def start_session(self) -> bool:
        """Start transcription session"""
        if self.state != SessionState.INACTIVE:
            return False
            
        # Load Whisper model if not loaded
        if not self.transcriber.is_loaded:
            if not self.transcriber.load_model():
                return False
        
        # Start audio capture
        if not self.audio_capture.start_capture():
            return False
            
        # Start processing thread
        self.stop_event.clear()
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        self.state = SessionState.RUNNING_ACTIVE
        logger.info("Transcription session started")
        return True
    
    def stop_session(self):
        """Stop transcription session"""
        if self.state == SessionState.INACTIVE:
            return
            
        self.state = SessionState.INACTIVE
        self.stop_event.set()
        
        # Stop audio capture
        self.audio_capture.stop_capture()
        
        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
            
        logger.info("Transcription session stopped")
    
    def pause_session(self):
        """Pause transcription session"""
        if self.state == SessionState.RUNNING_ACTIVE:
            self.state = SessionState.RUNNING_PAUSED
            logger.info("Transcription session paused")
    
    def resume_session(self):
        """Resume transcription session"""
        if self.state == SessionState.RUNNING_PAUSED:
            self.state = SessionState.RUNNING_ACTIVE
            logger.info("Transcription session resumed")
    
    def _processing_loop(self):
        """Main processing loop for audio transcription"""
        last_transcription_time = time.time()
        transcription_interval = 2.0  # Transcribe every 2 seconds
        audio_chunks_processed = 0
        
        while not self.stop_event.is_set():
            try:
                # Process multiple audio chunks in one iteration to keep up
                chunks_this_iteration = 0
                max_chunks_per_iteration = 50  # Process up to 50 chunks at once
                
                while chunks_this_iteration < max_chunks_per_iteration:
                    audio_chunk = self.audio_capture.get_audio_chunk()
                    if audio_chunk is None:
                        break
                        
                    if self.state == SessionState.RUNNING_ACTIVE:
                        self.audio_buffer.extend(audio_chunk)
                    
                    chunks_this_iteration += 1
                    audio_chunks_processed += 1
                
                # Log processing stats occasionally
                if audio_chunks_processed % 1000 == 0:
                    logger.debug(f"Processed {audio_chunks_processed} audio chunks, queue size: {self.audio_capture.audio_queue.qsize()}")
                
                # Process transcription periodically
                current_time = time.time()
                if (current_time - last_transcription_time >= transcription_interval and 
                    len(self.audio_buffer) > self.audio_config.sample_rate):  # At least 1 second of audio
                    
                    # Get audio buffer as numpy array
                    audio_data = np.array(list(self.audio_buffer))
                    
                    # Transcribe
                    result = self.transcriber.transcribe_audio(audio_data)
                    if result and result['text'].strip():
                        # Update transcription
                        self.partial_transcription = result['text'].strip()
                        
                        # Create transcript word
                        word = TranscriptWord(
                            text=self.partial_transcription,
                            start=current_time - transcription_interval,
                            end=current_time,
                            is_final=True
                        )
                        self.final_transcriptions.append(word)
                        
                        # Callback to update UI
                        if self.transcription_callback:
                            self.transcription_callback(self.partial_transcription, True)
                        
                        logger.info(f"Transcribed: {self.partial_transcription}")
                    
                    last_transcription_time = current_time
                
                # Shorter sleep if we processed chunks, longer if queue was empty
                if chunks_this_iteration > 0:
                    time.sleep(0.01)  # Very short sleep when actively processing
                else:
                    time.sleep(0.05)  # Slightly longer when idle
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(0.5)


class TranscriptionGUI:
    """DearPyGui-based GUI for the transcription application"""
    
    def __init__(self):
        self.session = TranscriptionSession(AudioConfig())
        self.session.set_transcription_callback(self._on_transcription_update)
        
        # GUI state
        self.transcription_text = ""
        self.is_running = False
        
    def _on_transcription_update(self, text: str, is_final: bool):
        """Callback for transcription updates"""
        if is_final:
            self.transcription_text += f"\n[{time.strftime('%H:%M:%S')}] {text}"
            # Update GUI text (thread-safe)
            dpg.set_value("transcription_display", self.transcription_text)
            
            # Force auto-scroll to bottom - multiple approaches for reliability
            try:
                # Method 1: Set scroll to maximum value
                dpg.set_y_scroll("transcription_window", 1.0)
                
                # Method 2: Use a small delay and try again (for DearPyGui timing)
                def delayed_scroll():
                    time.sleep(0.1)
                    try:
                        # Get the actual scroll max and set it
                        scroll_max = dpg.get_y_scroll_max("transcription_window")
                        if scroll_max > 0:
                            dpg.set_y_scroll("transcription_window", scroll_max)
                        else:
                            # Fallback: set a very large value
                            dpg.set_y_scroll("transcription_window", 999999)
                    except:
                        pass
                
                # Run delayed scroll in a separate thread
                threading.Thread(target=delayed_scroll, daemon=True).start()
                
            except Exception as e:
                logger.debug(f"Scroll update failed: {e}")
    
    def _start_transcription(self):
        """Start transcription callback"""
        if not self.is_running:
            if self.session.start_session():
                self.is_running = True
                dpg.set_value("start_button", "Stop")
                dpg.set_value("status_text", "Status: Recording and Transcribing...")
                logger.info("Started transcription from GUI")
            else:
                dpg.set_value("status_text", "Status: Failed to start")
        else:
            self.session.stop_session()
            self.is_running = False
            dpg.set_value("start_button", "Start")
            dpg.set_value("status_text", "Status: Stopped")
            logger.info("Stopped transcription from GUI")
    
    def _clear_transcription(self):
        """Clear transcription display"""
        self.transcription_text = ""
        dpg.set_value("transcription_display", "")
        logger.info("Cleared transcription display")
    
    def _save_transcription(self):
        """Save transcription to file"""
        if self.transcription_text:
            filename = f"transcription_{int(time.time())}.txt"
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.transcription_text)
                dpg.set_value("status_text", f"Status: Saved to {filename}")
                logger.info(f"Saved transcription to {filename}")
            except Exception as e:
                dpg.set_value("status_text", f"Status: Save failed - {str(e)}")
                logger.error(f"Failed to save transcription: {e}")
        else:
            dpg.set_value("status_text", "Status: No transcription to save")
    
    def _get_audio_devices(self) -> List[str]:
        """Get list of available audio devices"""
        try:
            devices = sd.query_devices()
            device_names = []
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    device_names.append(f"{i}: {device['name']}")
            return device_names
        except Exception as e:
            logger.error(f"Error getting audio devices: {e}")
            return ["Error loading devices"]
    
    def _get_whisper_models(self) -> List[str]:
        """Get list of available Whisper models with info"""
        models = []
        for model in self.session.transcriber.available_models:
            info = self.session.transcriber.get_model_info(model)
            models.append(f"{model.title()} - {info['size']} - {info['accuracy']} accuracy")
        return models
    
    def _on_model_change(self, sender, app_data, user_data):
        """Handle model selection change"""
        if self.is_running:
            dpg.set_value("status_text", "Status: Cannot change model while running")
            return
        
        # Extract model name from selection
        selected = app_data
        model_name = selected.split(' - ')[0].lower()
        
        # Update status
        dpg.set_value("status_text", f"Status: Loading {model_name} model...")
        
        # Load model in background thread to avoid freezing UI
        def load_model():
            success = self.session.transcriber.change_model(model_name)
            if success:
                # Update footer display
                dpg.set_value("model_display", f"Model: Whisper {model_name.title()}")
                dpg.set_value("status_text", f"Status: {model_name.title()} model loaded successfully")
                logger.info(f"Changed to Whisper {model_name} model")
            else:
                dpg.set_value("status_text", f"Status: Failed to load {model_name} model")
                logger.error(f"Failed to change to {model_name} model")
        
        threading.Thread(target=load_model, daemon=True).start()
    
    def create_gui(self):
        """Create the main GUI"""
        # Create Dear PyGui context
        dpg.create_context()
        
        # Main window
        with dpg.window(label="Real-time Audio Transcription", tag="main_window"):
            
            # Header
            dpg.add_text("Real-time Audio Transcription", color=(100, 150, 250))
            dpg.add_separator()
            
            # Controls section
            with dpg.group(horizontal=True):
                dpg.add_button(label="Start", tag="start_button", callback=self._start_transcription, width=100)
                dpg.add_button(label="Clear", callback=self._clear_transcription, width=100)
                dpg.add_button(label="Save", callback=self._save_transcription, width=100)
            
            dpg.add_separator()
            
            # Status
            dpg.add_text("Status: Ready", tag="status_text", color=(150, 150, 150))
            
            # Device and Model selection
            with dpg.group(horizontal=True):
                dpg.add_text("Audio Device:")
                device_list = self._get_audio_devices()
                dpg.add_combo(device_list, default_value=device_list[0] if device_list else "No devices", 
                             width=250, tag="device_combo")
            
            with dpg.group(horizontal=True):
                dpg.add_text("Whisper Model:")
                model_list = self._get_whisper_models()
                # Set default to Base model (index 1)
                default_model = model_list[1] if len(model_list) > 1 else model_list[0]
                dpg.add_combo(model_list, default_value=default_model, 
                             width=250, tag="model_combo", callback=self._on_model_change)
            
            dpg.add_separator()
            
            # Transcription display
            dpg.add_text("Transcription:")
            with dpg.child_window(height=400, tag="transcription_window"):
                dpg.add_input_text(
                    default_value="Transcription will appear here...",
                    tag="transcription_display",
                    multiline=True,
                    readonly=True,
                    width=-1,
                    height=-1
                )
            
            # Footer
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_text("Model: Whisper Base", color=(100, 100, 100), tag="model_display")
                dpg.add_text(" | ", color=(100, 100, 100))
                dpg.add_text("Sample Rate: 16kHz", color=(100, 100, 100))
        
        # Setup viewport
        dpg.create_viewport(title="Real-time Transcription", width=800, height=600)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)
        
    def run(self):
        """Run the GUI application"""
        self.create_gui()
        
        try:
            # Main render loop
            while dpg.is_dearpygui_running():
                dpg.render_dearpygui_frame()
        finally:
            # Cleanup
            if self.is_running:
                self.session.stop_session()
            dpg.destroy_context()


def main():
    """Main application entry point"""
    logger.info("Starting Real-time Audio Transcription Application")
    
    try:
        app = TranscriptionGUI()
        app.run()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        logger.info("Application ended")


if __name__ == "__main__":
    main()
