MAX_TEXT_LENGTH = 1_500_000
import torch
from typing import Optional, Tuple, List, Union
from models import build_model, generate_speech, list_available_voices
from tqdm.auto import tqdm
import soundfile as sf
from pathlib import Path
import numpy as np
import time
import os
import sys

# Define path type for consistent handling
PathLike = Union[str, Path]
# Constants
MAX_GENERATION_TIME = 300  # seconds
MIN_GENERATION_TIME = 60   # seconds
DEFAULT_SAMPLE_RATE = 24000
MIN_SPEED = 0.1
MAX_SPEED = 3.0
DEFAULT_SPEED = 1.0
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Constants with validation
def validate_sample_rate(rate: int) -> int:
    """Validate sample rate is within acceptable range"""
    valid_rates = [16000, 22050, 24000, 44100, 48000]
    if rate not in valid_rates:
        print(f"Warning: Unusual sample rate {rate}. Valid rates are {valid_rates}")
        return 24000  # Default to safe value
    return rate

def validate_language(lang: str) -> str:
    """Validate language code"""
    # Import here to avoid circular imports
    from models import LANGUAGE_CODES
    valid_langs = list(LANGUAGE_CODES.keys())
    if lang not in valid_langs:
        print(f"Warning: Invalid language code '{lang}'. Using 'a' (American English).")
        print(f"Supported language codes: {', '.join(valid_langs)}")
        return 'a'  # Default to American English
    return lang

# Define and validate constants
SAMPLE_RATE = validate_sample_rate(24000)
DEFAULT_MODEL_PATH = Path('kokoro-v1_0.pth').resolve()
DEFAULT_OUTPUT_FILE = Path('output.wav').resolve()
DEFAULT_LANGUAGE = validate_language('a')  # 'a' for American English, 'b' for British English
DEFAULT_TEXT = "Hello, welcome to this text-to-speech test."

# Ensure output directory exists
DEFAULT_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Configure tqdm for better Windows console support
tqdm.monitor_interval = 0

def print_menu():
    """Print the main menu options."""
    print("\n=== Kokoro TTS Menu ===")
    print("1. List available voices")
    print("2. Generate speech from text input")
    print("3. Generate speech from EPUB/PDF file")
    print("4. Exit")
    return input("Select an option (1-4): ").strip()
def extract_text_from_epub(epub_path: PathLike) -> str:
    from ebooklib import epub
    import ebooklib
    from bs4 import BeautifulSoup
    book = epub.read_epub(str(epub_path))
    text = []
    for item in book.get_items():
        # Prefer the numeric constant from the top-level ebooklib module, which
        # is always available (some versions don't expose ITEM_DOCUMENT on the
        # epub submodule).
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text.append(soup.get_text())
    return '\n'.join(text)

def extract_text_from_pdf(pdf_path: PathLike) -> str:
    import PyPDF2
    text = []
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text.append(page.extract_text() or "")
    return '\n'.join(text)

def select_voice(voices: List[str]) -> str:
    """Interactive voice selection."""
    print("\nAvailable voices:")
    for i, voice in enumerate(voices, 1):
        print(f"{i}. {voice}")

    while True:
        try:
            choice = input("\nSelect a voice number (or press Enter for default 'af_bella'): ").strip()
            if not choice:
                return "af_bella"
            choice = int(choice)
            if 1 <= choice <= len(voices):
                return voices[choice - 1]
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def get_text_input() -> str:
    """Get text input from user."""
    print("\nEnter the text you want to convert to speech")
    print("(or press Enter for default text)")
    text = input("> ").strip()
    return text if text else DEFAULT_TEXT

def get_speed() -> float:
    """Get speech speed from user."""
    while True:
        try:
            speed = input(f"\nEnter speech speed ({MIN_SPEED}-{MAX_SPEED}, default {DEFAULT_SPEED}): ").strip()
            if not speed:
                return DEFAULT_SPEED
            speed = float(speed)
            if MIN_SPEED <= speed <= MAX_SPEED:
                return speed
            print(f"Speed must be between {MIN_SPEED} and {MAX_SPEED}")
        except ValueError:
            print("Please enter a valid number.")

def get_stream_choice() -> bool:
    """Ask the user whether to stream playback as segments are generated.

    Returns:
        True if streaming should be enabled, False otherwise.
    """
    while True:
        choice = input("\nStream audio as it's generated? (y/N): ").strip().lower()
        if choice in ("y", "yes"):
            return True
        if choice in ("n", "no", ""):
            return False
        print("Please answer 'y' or 'n'.")


def get_stream_delay() -> float:
    """Get the inter-segment delay in seconds.

    Enter 0 to wait until the segment playback finishes before continuing.
    Default is 0.3 seconds.
    """
    while True:
        val = input("\nInter-segment delay in seconds (enter 0 to wait until playback finishes) [default 0.3]: ").strip()
        if val == "":
            return 0.3
        try:
            f = float(val)
            if f < 0:
                print("Please enter a non-negative number.")
                continue
            return f
        except ValueError:
            print("Please enter a valid number (e.g., 0.2 or 0).")


def play_audio_segment(audio_np: np.ndarray, sample_rate: int, play_handles: list):
    """Attempt to play an audio segment using available playback libraries.

    Tries in order: simpleaudio, sounddevice. If neither is available, writes
    the segment to a temp WAV and notifies the user.

    Returns:
        A tuple (handle_or_info, method) where method is one of:
        'simpleaudio', 'sounddevice', 'file', or 'none'.
        The handle_or_info may be a play handle (for simpleaudio), None
        (for sounddevice), or a temp file path (for 'file').
    """
    # Normalize to float32 in range [-1,1]
    try:
        audio_np = audio_np.astype(np.float32)
    except Exception:
        audio_np = np.array(audio_np, dtype=np.float32)
    # Clip to safe range
    audio_np = np.clip(audio_np, -1.0, 1.0)

    # Try simpleaudio (non-blocking)
    try:
        import simpleaudio as sa
        data = (audio_np * 32767).astype(np.int16).tobytes()
        handle = sa.play_buffer(data, 1, 2, sample_rate)
        play_handles.append(handle)
        return handle, 'simpleaudio'
    except Exception:
        pass

    # Try sounddevice (non-blocking play)
    try:
        import sounddevice as sd
        sd.play(audio_np, sample_rate)
        play_handles.append(None)
        return None, 'sounddevice'
    except Exception:
        pass

    # Fallback: write a temp WAV file so user can play it externally
    try:
        import tempfile
        temp_dir = Path(tempfile.gettempdir())
        temp_file = temp_dir / f"stream_seg_{int(time.time() * 1000)}.wav"
        sf.write(str(temp_file), audio_np, sample_rate)
        print(f"Saved segment to {temp_file} — open it in your media player to hear streamed audio.")
        play_handles.append(str(temp_file))
        return str(temp_file), 'file'
    except Exception as e:
        print(f"Streaming unavailable and failed to save segment: {e}")
        return None, 'none'


def _prefetch_generator_to_queue(gen, out_queue, stop_event):
    """Run the model generator in a background thread and push items to a queue.

    Each item pushed is a tuple (gs, ps, audio_np_or_tensor). When generator
    finishes, a sentinel (None, None, None) will be pushed. If an exception
    occurs, a tuple ('__ERR__', exception, None) will be pushed.
    """
    try:
        for item in gen:
            if stop_event.is_set():
                break
            # Ensure we don't block forever if the consumer dies; use timeout
            pushed = False
            while not stop_event.is_set() and not pushed:
                try:
                    out_queue.put(item, timeout=0.5)
                    pushed = True
                except Exception:
                    # queue.Full or other transient error; check stop_event
                    continue
    except Exception as e:
        try:
            out_queue.put(("__ERR__", e, None), timeout=1)
        except Exception:
            pass
    finally:
        try:
            out_queue.put((None, None, None), timeout=1)
        except Exception:
            pass


def save_audio_with_retry(audio_data: np.ndarray, sample_rate: int, output_path: PathLike, max_retries: int = MAX_RETRIES, retry_delay: float = RETRY_DELAY) -> bool:
    """
    Attempt to save audio data to file with retry logic.

    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate in Hz
        output_path: Path to save the audio file
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds

    Returns:
        True if successful, False otherwise
    """
    # Convert and normalize path to Path object
    output_path = Path(output_path).resolve()

    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try to remove the file if it exists to avoid "file in use" issues
    try:
        if output_path.exists():
            print(f"Removing existing file: {output_path}")
            output_path.unlink()
    except Exception as e:
        print(f"Warning: Could not remove existing file: {e}")
        print("This might indicate the file is in use by another program.")

    for attempt in range(max_retries):
        try:
            # Validate audio data before saving
            if audio_data is None or len(audio_data) == 0:
                raise ValueError("Empty audio data")

            # Check write permissions for the directory
            if not os.access(str(output_path.parent), os.W_OK):
                raise PermissionError(f"No write permission for directory: {output_path.parent}")

            # Try to use a temporary file first, then rename it
            temp_path = output_path.with_name(f"temp_{output_path.name}")

            # Save audio file to temporary location
            print(f"Saving audio to temporary file: {temp_path}")
            sf.write(str(temp_path), audio_data, sample_rate)

            # If successful, rename to final location
            if temp_path.exists():
                # Remove target file if it exists
                if output_path.exists():
                    output_path.unlink()
                # Rename temp file to target file
                temp_path.rename(output_path)
                print(f"Successfully renamed temporary file to: {output_path}")

            return True

        except (IOError, PermissionError) as e:
            if attempt < max_retries - 1:
                print(f"\nFailed to save audio (attempt {attempt + 1}/{max_retries}): {e}")
                print("The output file might be in use by another program (e.g., media player).")
                print(f"Please close any programs that might be using '{output_path}'")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"\nError: Could not save audio after {max_retries} attempts: {e}")
                print(f"Please ensure '{output_path}' is not open in any other program and try again.")
                print(f"You might need to restart your computer if the file remains locked.")
                return False
        except Exception as e:
            print(f"\nUnexpected error saving audio: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                return False
        finally:
            # Clean up temp file if it exists and we failed
            try:
                temp_path = output_path.with_name(f"temp_{output_path.name}")
                if temp_path.exists():
                    temp_path.unlink()
            except Exception as e:
                print(f"Warning: Could not clean up temporary file {temp_path}: {e}")

    return False

def main() -> None:
    import psutil
    import gc
    
    try:
        # Check system memory at startup
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        total_gb = memory.total / (1024**3)
        
        print(f"System memory: {available_gb:.1f}GB available / {total_gb:.1f}GB total")
        
        if available_gb < 2.0:
            print("Warning: Low system memory detected. Consider closing other applications.")
            # Force garbage collection
            gc.collect()

        # Set up device safely
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except (RuntimeError, AttributeError, ImportError) as e:
            print(f"CUDA initialization error: {e}. Using CPU instead.")
            device = 'cpu'  # Fallback if CUDA check fails
        print(f"Using device: {device}")

        # Build model
        print("\nInitializing model...")
        with tqdm(total=1, desc="Building model") as pbar:
            model = build_model(DEFAULT_MODEL_PATH, device)
            pbar.update(1)

        # Cache for voices to avoid redundant calls
        voices_cache = None


        while True:
            choice = print_menu()

            if choice == "1":
                # List voices
                voices_cache = list_available_voices()
                print("\nAvailable voices:")
                for voice in voices_cache:
                    print(f"- {voice}")

            elif choice == "2":
                # Generate speech from text input
                # Use cached voices if available
                if voices_cache is None:
                    voices_cache = list_available_voices()
                if not voices_cache:
                    print("No voices found! Please check the voices directory.")
                    continue
                voice = select_voice(voices_cache)
                text = get_text_input()
                # ...existing code for text length validation and speech generation...
                memory = psutil.virtual_memory()
                available_gb = memory.available / (1024**3)
                dynamic_max_length = MAX_TEXT_LENGTH
                if available_gb < 2.0:
                    dynamic_max_length = min(MAX_TEXT_LENGTH, 3000)
                    print(f"Reduced text limit to {dynamic_max_length} characters due to low memory")
                if len(text) > dynamic_max_length:
                    print(f"Text is too long ({len(text)} chars). Maximum allowed: {dynamic_max_length} characters.")
                    print("Please enter a shorter text.")
                    continue
                speed = get_speed()
                stream = get_stream_choice()
                stream_delay = get_stream_delay() if stream else None
                print(f"\nGenerating speech for: '{text}'")
                print(f"Using voice: {voice}")
                print(f"Speed: {speed}x")
                all_audio = []
                play_handles = []  # Track playback handles for non-blocking playback libraries
                voice_path = Path("voices").resolve() / f"{voice}.pt"
                if not voice_path.exists():
                    print(f"Error: Voice file not found: {voice_path}")
                    continue
                max_gen_time = MAX_GENERATION_TIME
                max_segment_time = MIN_GENERATION_TIME
                start_time = time.time()
                segment_start_time = start_time
                try:
                    import threading
                    generation_complete = False
                    def watchdog_timer():
                        if not generation_complete:
                            print("\nWatchdog: Generation taking too long, process will be cancelled")
                    watchdog = threading.Timer(max_gen_time, watchdog_timer)
                    watchdog.daemon = True
                    watchdog.start()
                    try:
                        generator = model(text, voice=str(voice_path), speed=speed, split_pattern=r'\n+')
                    except (ValueError, TypeError, RuntimeError) as e:
                        print(f"Error initializing speech generator: {e}")
                        watchdog.cancel()
                        continue
                    except Exception as e:
                        print(f"Unexpected error initializing generator: {type(e).__name__}: {e}")
                        watchdog.cancel()
                        continue

                    # When streaming, prefetch the next segment in a background thread so
                    # playback can start immediately without waiting for generation.
                    from queue import Queue
                    q = Queue(maxsize=2)
                    stop_event = threading.Event()
                    producer = threading.Thread(target=_prefetch_generator_to_queue, args=(generator, q, stop_event), daemon=True)
                    producer.start()

                    with tqdm(desc="Generating speech") as pbar:
                        while True:
                            try:
                                gs, ps, audio = q.get(timeout=1)
                            except Exception:
                                # Timeout — check for watchdog/time limits
                                current_time = time.time()
                                if current_time - start_time > max_gen_time:
                                    print("\nWarning: Total generation time exceeded limit, stopping")
                                    stop_event.set()
                                    break
                                continue

                            # Check for sentinel / end / errors
                            if gs is None and ps is None and audio is None:
                                break
                            if gs == "__ERR__":
                                print(f"Error while generating segment: {ps}")
                                stop_event.set()
                                break

                            current_time = time.time()
                            if current_time - start_time > max_gen_time:
                                print("\nWarning: Total generation time exceeded limit, stopping")
                                stop_event.set()
                                break

                            segment_elapsed = current_time - segment_start_time
                            if segment_elapsed > max_segment_time:
                                print(f"\nWarning: Segment took too long ({segment_elapsed:.1f}s), stopping")
                                stop_event.set()
                                break
                            segment_start_time = current_time

                            if audio is not None:
                                audio_tensor = audio if isinstance(audio, torch.Tensor) else torch.from_numpy(audio).float()
                                all_audio.append(audio_tensor)
                                print(f"\nGenerated segment: {gs}")
                                if ps:
                                    print(f"Phonemes: {ps}")
                                # Play the segment immediately if streaming was requested
                                if stream:
                                    try:
                                        audio_np = audio_tensor.numpy()
                                        handle, method = play_audio_segment(audio_np, SAMPLE_RATE, play_handles)
                                        # Delay or wait to avoid cutting mid-sentence; producer keeps
                                        # generating the next segment while we wait here.
                                        if stream_delay is not None:
                                            try:
                                                if stream_delay == 0:
                                                    if method == 'simpleaudio' and handle is not None:
                                                        handle.wait_done()
                                                    elif method == 'sounddevice':
                                                        import sounddevice as sd
                                                        sd.wait()
                                                    else:
                                                        time.sleep(0.5)
                                                else:
                                                    time.sleep(max(0.0, stream_delay))
                                            except Exception:
                                                time.sleep(max(0.0, stream_delay or 0.2))
                                    except Exception as e:
                                        print(f"Error while attempting to play segment: {e}")
                                pbar.update(1)

                    # Ensure producer thread is stopped
                    stop_event.set()
                    try:
                        producer.join(timeout=1)
                    except Exception:
                        pass
                    generation_complete = True
                    watchdog.cancel()
                except ValueError as e:
                    print(f"Value error during speech generation: {e}")
                except RuntimeError as e:
                    print(f"Runtime error during speech generation: {e}")
                    if "CUDA out of memory" in str(e):
                        print("CUDA out of memory error - try using a shorter text or switching to CPU")
                except KeyError as e:
                    print(f"Key error during speech generation: {e}")
                    print("This might be caused by a missing voice configuration")
                except FileNotFoundError as e:
                    print(f"File not found: {e}")
                except Exception as e:
                    print(f"Unexpected error during speech generation: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                # If streaming, wait for any non-blocking players to finish playback
                if stream and play_handles:
                    try:
                        for h in play_handles:
                            if h is not None and hasattr(h, 'wait_done'):
                                h.wait_done()
                    except Exception:
                        pass
                if all_audio:
                    try:
                        if len(all_audio) == 1:
                            final_audio = all_audio[0]
                        else:
                            try:
                                final_audio = torch.cat(all_audio, dim=0)
                            except RuntimeError as e:
                                print(f"Error concatenating audio segments: {e}")
                                continue
                        output_path = DEFAULT_OUTPUT_FILE
                        if save_audio_with_retry(final_audio.numpy(), SAMPLE_RATE, output_path):
                            print(f"\nAudio saved to {output_path}")
                            try:
                                print('\a')
                            except:
                                pass
                        else:
                            print("Failed to save audio file")
                    except Exception as e:
                        print(f"Error processing audio: {type(e).__name__}: {e}")
                else:
                    print("Error: Failed to generate audio")

            elif choice == "3":
                # Generate speech from EPUB/PDF file
                if voices_cache is None:
                    voices_cache = list_available_voices()
                if not voices_cache:
                    print("No voices found! Please check the voices directory.")
                    continue
                voice = select_voice(voices_cache)
                print("\nEnter the path to your EPUB or PDF file:")
                file_path = input("> ").strip()
                if not file_path or not os.path.isfile(file_path):
                    print("Invalid file path. Please try again.")
                    continue
                ext = os.path.splitext(file_path)[1].lower()
                try:
                    if ext == ".epub":
                        text = extract_text_from_epub(file_path)
                    elif ext == ".pdf":
                        text = extract_text_from_pdf(file_path)
                    else:
                        print("Unsupported file type. Please provide an EPUB or PDF file.")
                        continue
                except Exception as e:
                    print(f"Error extracting text: {e}")
                    continue
                if not text.strip():
                    print("No text could be extracted from the file.")
                    continue
                print(f"Extracted {len(text)} characters from file.")
                memory = psutil.virtual_memory()
                available_gb = memory.available / (1024**3)
                dynamic_max_length = MAX_TEXT_LENGTH
                if available_gb < 2.0:
                    dynamic_max_length = min(MAX_TEXT_LENGTH, 3000)
                    print(f"Reduced text limit to {dynamic_max_length} characters due to low memory")
                if len(text) > dynamic_max_length:
                    print(f"Text is too long ({len(text)} chars). Only the first {dynamic_max_length} characters will be used.")
                    text = text[:dynamic_max_length]
                speed = get_speed()
                stream = get_stream_choice()
                stream_delay = get_stream_delay() if stream else None
                print(f"\nGenerating speech from file: {file_path}")
                print(f"Using voice: {voice}")
                print(f"Speed: {speed}x")
                all_audio = []
                play_handles = []
                voice_path = Path("voices").resolve() / f"{voice}.pt"
                if not voice_path.exists():
                    print(f"Error: Voice file not found: {voice_path}")
                    continue
                max_gen_time = MAX_GENERATION_TIME
                max_segment_time = MIN_GENERATION_TIME
                start_time = time.time()
                segment_start_time = start_time
                try:
                    import threading
                    generation_complete = False
                    def watchdog_timer():
                        if not generation_complete:
                            print("\nWatchdog: Generation taking too long, process will be cancelled")
                    watchdog = threading.Timer(max_gen_time, watchdog_timer)
                    watchdog.daemon = True
                    watchdog.start()
                    try:
                        generator = model(text, voice=str(voice_path), speed=speed, split_pattern=r'\n+')
                    except (ValueError, TypeError, RuntimeError) as e:
                        print(f"Error initializing speech generator: {e}")
                        watchdog.cancel()
                        continue
                    except Exception as e:
                        print(f"Unexpected error initializing generator: {type(e).__name__}: {e}")
                        watchdog.cancel()
                        continue

                    # Prefetch producer/consumer queue to generate the next segment while
                    # the current one is being played
                    from queue import Queue
                    q = Queue(maxsize=2)
                    stop_event = threading.Event()
                    producer = threading.Thread(target=_prefetch_generator_to_queue, args=(generator, q, stop_event), daemon=True)
                    producer.start()

                    with tqdm(desc="Generating speech") as pbar:
                        while True:
                            try:
                                gs, ps, audio = q.get(timeout=1)
                            except Exception:
                                current_time = time.time()
                                if current_time - start_time > max_gen_time:
                                    print("\nWarning: Total generation time exceeded limit, stopping")
                                    stop_event.set()
                                    break
                                continue

                            # detect end of generation
                            if gs is None and ps is None and audio is None:
                                break
                            if gs == "__ERR__":
                                print(f"Error while generating segment: {ps}")
                                stop_event.set()
                                break

                            current_time = time.time()
                            if current_time - start_time > max_gen_time:
                                print("\nWarning: Total generation time exceeded limit, stopping")
                                stop_event.set()
                                break

                            segment_elapsed = current_time - segment_start_time
                            if segment_elapsed > max_segment_time:
                                print(f"\nWarning: Segment took too long ({segment_elapsed:.1f}s), stopping")
                                stop_event.set()
                                break
                            segment_start_time = current_time

                            if audio is not None:
                                audio_tensor = audio if isinstance(audio, torch.Tensor) else torch.from_numpy(audio).float()
                                all_audio.append(audio_tensor)
                                print(f"\nGenerated segment: {gs}")
                                if ps:
                                    print(f"Phonemes: {ps}")
                                if stream:
                                    try:
                                        audio_np = audio_tensor.numpy()
                                        handle, method = play_audio_segment(audio_np, SAMPLE_RATE, play_handles)
                                        if stream_delay is not None:
                                            try:
                                                if stream_delay == 0:
                                                    if method == 'simpleaudio' and handle is not None:
                                                        handle.wait_done()
                                                    elif method == 'sounddevice':
                                                        import sounddevice as sd
                                                        sd.wait()
                                                    else:
                                                        time.sleep(0.5)
                                                else:
                                                    time.sleep(max(0.0, stream_delay))
                                            except Exception:
                                                time.sleep(max(0.0, stream_delay or 0.2))
                                    except Exception as e:
                                        print(f"Error while attempting to play segment: {e}")
                                pbar.update(1)

                    stop_event.set()
                    try:
                        producer.join(timeout=1)
                    except Exception:
                        pass
                    generation_complete = True
                    watchdog.cancel()
                except ValueError as e:
                    print(f"Value error during speech generation: {e}")
                except RuntimeError as e:
                    print(f"Runtime error during speech generation: {e}")
                    if "CUDA out of memory" in str(e):
                        print("CUDA out of memory error - try using a shorter text or switching to CPU")
                except KeyError as e:
                    print(f"Key error during speech generation: {e}")
                    print("This might be caused by a missing voice configuration")
                except FileNotFoundError as e:
                    print(f"File not found: {e}")
                except Exception as e:
                    print(f"Unexpected error during speech generation: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                if stream and play_handles:
                    try:
                        for h in play_handles:
                            if h is not None and hasattr(h, 'wait_done'):
                                h.wait_done()
                    except Exception:
                        pass
                if all_audio:
                    try:
                        if len(all_audio) == 1:
                            final_audio = all_audio[0]
                        else:
                            try:
                                final_audio = torch.cat(all_audio, dim=0)
                            except RuntimeError as e:
                                print(f"Error concatenating audio segments: {e}")
                                continue
                        output_path = DEFAULT_OUTPUT_FILE
                        if save_audio_with_retry(final_audio.numpy(), SAMPLE_RATE, output_path):
                            print(f"\nAudio saved to {output_path}")
                            try:
                                print('\a')
                            except:
                                pass
                        else:
                            print("Failed to save audio file")
                    except Exception as e:
                        print(f"Error processing audio: {type(e).__name__}: {e}")
                else:
                    print("Error: Failed to generate audio")

            elif choice == "4":
                print("\nGoodbye!")
                break
            else:
                print("\nInvalid choice. Please try again.")

    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Comprehensive cleanup with error handling
        try:
            print("\nPerforming cleanup...")

            # Ensure model is properly released
            if 'model' in locals() and model is not None:
                print("Cleaning up model resources...")
                # First clear any references to voice models
                if hasattr(model, 'voices'):
                    try:
                        voices_count = len(model.voices)
                        model.voices.clear()
                        print(f"Cleared {voices_count} voice references")
                    except Exception as voice_error:
                        print(f"Error clearing voice references: {voice_error}")

                # Clear any other model attributes that might hold references
                try:
                    for attr in list(model.__dict__.keys()):
                        if hasattr(model, attr) and not attr.startswith('__'):
                            try:
                                delattr(model, attr)
                            except:
                                pass
                except Exception as attr_error:
                    print(f"Error clearing model attributes: {attr_error}")

                # Then delete the model
                try:
                    del model
                    model = None
                    print("Model reference deleted")
                except Exception as del_error:
                    print(f"Error deleting model: {del_error}")

            # Clean up voice cache
            if 'voices_cache' in locals() and voices_cache is not None:
                try:
                    voices_cache.clear()
                    voices_cache = None
                    print("Voice cache cleared")
                except Exception as cache_error:
                    print(f"Error clearing voice cache: {cache_error}")

            # Clean up any CUDA resources
            if torch.cuda.is_available():
                try:
                    print("Cleaning up CUDA resources...")
                    torch.cuda.empty_cache()
                    print("CUDA cache emptied")
                except Exception as cuda_error:
                    print(f"Error clearing CUDA cache: {cuda_error}")


            # Final garbage collection
            try:
                import gc
                gc.collect()
                print("Garbage collection completed")
            except Exception as gc_error:
                print(f"Error during garbage collection: {gc_error}")

            print("Cleanup completed")

        except Exception as e:
            print(f"Error during cleanup: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
