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
import requests
import json

# Define path type for consistent handling
PathLike = Union[str, Path]
# Constants
MAX_GENERATION_TIME = None  # No limit when None
MIN_GENERATION_TIME = None   # No per-segment limit when None
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
    print("5. Interactive Ollama LLM (neural-chat) — stream responses")
    return input("Select an option (1-5): ").strip()
def extract_text_from_epub(epub_path: PathLike, select_chapter: bool = False) -> str:
    """Extract text from an EPUB file.

    If select_chapter is True and multiple document items are present, the
    user will be shown a numbered list of chapters (title + preview) and can
    choose a single chapter, 'a' for all chapters, or press Enter to cancel.
    """
    from ebooklib import epub
    import ebooklib
    from bs4 import BeautifulSoup

    book = epub.read_epub(str(epub_path))

    # Collect document items in reading order
    doc_items = [it for it in book.get_items() if it.get_type() == ebooklib.ITEM_DOCUMENT]

    if not doc_items:
        return ""

    # If the user requested chapter selection and there are multiple items,
    # present a compact preview list and allow choosing one.
    if select_chapter and len(doc_items) > 1:
        chapters = []
        for item in doc_items:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            # Prefer heading or title tags for a readable chapter title
            title_tag = soup.find(['h1', 'h2', 'title'])
            title = title_tag.get_text(strip=True) if title_tag and title_tag.get_text(strip=True) else (item.get_name() or item.file_name or 'Untitled')
            preview = soup.get_text(separator=' ', strip=True)
            preview = ' '.join(preview.split())[:140]
            chapters.append((title, preview, item))

        print("\nFound the following chapters in the EPUB:")
        for i, (title, preview, _) in enumerate(chapters, 1):
            ell = '...' if len(preview) > 100 else ''
            print(f"{i}. {title} — {preview[:100]}{ell}")
        print("\nEnter a chapter number to use, 'a' for all chapters, or press Enter to cancel:")
        while True:
            choice = input('> ').strip().lower()
            if not choice:
                print("No chapter selected; canceling EPUB import.")
                return ""
            if choice in ('a', 'all'):
                break
            try:
                idx = int(choice)
                if 1 <= idx <= len(chapters):
                    sel_item = chapters[idx - 1][2]
                    sel_soup = BeautifulSoup(sel_item.get_content(), 'html.parser')
                    return sel_soup.get_text(separator='\n', strip=True)
                else:
                    print(f"Invalid choice: {choice}. Please enter a number between 1 and {len(chapters)}, 'a', or Enter to cancel.")
            except ValueError:
                print("Please enter a valid number, 'a' for all, or press Enter to cancel.")

    # Default: concatenate all document items in order
    parts = []
    for item in doc_items:
        soup = BeautifulSoup(item.get_content(), 'html.parser')
        parts.append(soup.get_text())
    return '\n'.join(parts)

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


def get_yes_no(prompt: str, default: bool = False) -> bool:
    """Prompt the user for a yes/no answer and return a bool."""
    while True:
        val = input(f"\n{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
        if val == "":
            return default
        if val in ("y", "yes"):
            return True
        if val in ("n", "no"):
            return False
        print("Please answer 'y' or 'n'.")


def ollama_is_available(ollama_url: str = 'http://localhost:11434') -> bool:
    """Quick check whether Ollama appears reachable."""
    try:
        r = requests.get(ollama_url + '/api/health', timeout=1)
        return r.status_code == 200
    except Exception:
        try:
            r = requests.get(ollama_url + '/api/models', timeout=1)
            return r.status_code == 200
        except Exception:
            return False


def _synthesize_and_play_text_chunk(text: str, tts_model, voice_path: Path, speed: float, play_handles: list, sample_rate: int = SAMPLE_RATE):
    """Synthesize `text` with Kokoro model and play segments as they arrive."""
    try:
        gen = tts_model(text, voice=str(voice_path), speed=speed, split_pattern=r'\n+')
        for gs, ps, audio in gen:
            if audio is None:
                continue
            try:
                audio_np = audio.numpy() if isinstance(audio, torch.Tensor) else audio
            except Exception:
                audio_np = np.array(audio, dtype=np.float32)
            try:
                play_audio_segment(audio_np, sample_rate, play_handles)
            except Exception as e:
                print(f"Error playing TTS chunk: {e}")
    except Exception as e:
        print(f"Error during inline TTS playback: {e}")


def ollama_stream_chat_mode(tts_model, device, ollama_url: str = 'http://localhost:11434', model_name: str = 'neural-chat'):
    """Interactive chat that streams responses from an Ollama model and optionally TTSs them."""
    print(f"\nEntering Ollama streaming chat mode (model: {model_name})")
    if not ollama_is_available(ollama_url):
        print(f"Could not reach Ollama at {ollama_url}. Ensure the service is running.")
        return
    print("Type your messages and press Enter. Empty message returns to menu.")
    tts_enabled = get_yes_no("Also speak responses with TTS?", default=False)
    voice = None
    speed = DEFAULT_SPEED
    play_handles = []
    voice_path = None
    if tts_enabled:
        voices = list_available_voices()
        voice = select_voice(voices)
        voice_path = Path("voices").resolve() / f"{voice}.pt"
        if not voice_path.exists():
            print(f"Voice not found: {voice_path}. Disabling TTS.")
            tts_enabled = False
        else:
            speed = get_speed()

    session = requests.Session()
    while True:
        try:
            user_prompt = input("\nYou: ").strip()
            if not user_prompt:
                print("Exiting Ollama chat mode.")
                break
            payload = {"model": model_name, "prompt": user_prompt, "stream": True}
            try:
                resp = session.post(ollama_url + '/api/chat', json=payload, stream=True, timeout=10)
                if resp.status_code != 200:
                    print(f"Error from Ollama: HTTP {resp.status_code} — {resp.text[:200]}")
                    continue
            except Exception as e:
                print(f"Error connecting to Ollama: {e}")
                continue

            print("\nAssistant: ", end='', flush=True)
            buffer = ""
            try:
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    raw = line.strip()
                    # Some servers send "data: ..." lines
                    if raw.startswith("data:"):
                        raw = raw[len("data:"):].strip()
                    try:
                        jobj = json.loads(raw)
                    except Exception:
                        # not JSON: print raw text chunk
                        chunk = raw
                        print(chunk, end='', flush=True)
                        buffer += chunk
                        if tts_enabled:
                            _synthesize_and_play_text_chunk(chunk, tts_model, voice_path, speed, play_handles)
                        continue
                    # extract text from common fields
                    txt = None
                    if isinstance(jobj, dict):
                        txt = jobj.get('response') or jobj.get('content') or jobj.get('text') or jobj.get('output') or jobj.get('message') or jobj.get('delta')
                        # If 'choices' style
                        if txt is None and 'choices' in jobj and isinstance(jobj['choices'], list):
                            try:
                                txt = ''.join([c.get('delta', {}).get('content','') if isinstance(c, dict) else '' for c in jobj['choices']])
                            except Exception:
                                pass
                    if txt:
                        print(txt, end='', flush=True)
                        buffer += txt
                        if tts_enabled:
                            # Simple strategy: speak each received txt chunk
                            _synthesize_and_play_text_chunk(txt, tts_model, voice_path, speed, play_handles)
                print()
            except KeyboardInterrupt:
                print("\n[Interrupted streaming response]")
                continue
            except Exception as e:
                print(f"\nError while streaming response: {e}")
                continue

            # Optionally final TTS (speak remainder or whole buffer)
            if tts_enabled and buffer:
                pass  # we've already spoken chunks inline
        except KeyboardInterrupt:
            print("\nExiting Ollama chat mode.")
            break


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

    The function checks `stop_event` so the producer can be cancelled by the
    consumer or an external interrupt (e.g., user pressing the interrupt key).
    """
    try:
        for item in gen:
            if stop_event.is_set():
                break
            pushed = False
            while not stop_event.is_set() and not pushed:
                try:
                    out_queue.put(item)
                    pushed = True
                except Exception:
                    # queue.Full or other transient error; check stop_event
                    continue
    except Exception as e:
        try:
            out_queue.put(("__ERR__", e, None))
        except Exception:
            pass
    finally:
        try:
            out_queue.put((None, None, None))
        except Exception:
            pass


def _interrupt_listener(interrupt_event, stop_event):
    """Background listener that sets `interrupt_event` when the user requests an interrupt.

    On Windows, uses msvcrt.kbhit/getwch for non-blocking key detection. The
    listener checks for 'i' or 'I' to signal an interrupt and exits when
    `stop_event` is set.
    """
    try:
        import msvcrt
    except Exception:
        # Fallback: no non-blocking console input available
        return

    print("(Streaming) Press 'i' to interrupt and choose another chapter.")
    while not stop_event.is_set() and not interrupt_event.is_set():
        try:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch.lower() == 'i':
                    interrupt_event.set()
                    print("\nInterrupt requested by user.")
                    break
        except Exception:
            # Avoid noisy errors from console
            pass
        time.sleep(0.05)


def _stop_play_handles(play_handles):
    """Attempt to stop currently playing playback handles.

    Handles can be simpleaudio PlayObject, string paths (no-op), or None.
    For sounddevice, we call sd.stop() to halt playback.
    """
    try:
        # stop simpleaudio handles
        for h in play_handles:
            try:
                if hasattr(h, 'stop'):
                    h.stop()
            except Exception:
                pass
        # stop sounddevice global playback if available
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
    except Exception:
        pass


def _control_listener(interrupt_event, pause_event, stop_event):
    """Cross-platform background listener for streaming controls.

    Keys:
      - 'i' : interrupt (stop generation and return to choices)
      - 'p' or space: toggle pause/play

    Uses msvcrt on Windows, otherwise tries to use termios/select on POSIX.
    Exits when stop_event is set.
    """
    try:
        import msvcrt
        use_msvcrt = True
    except Exception:
        use_msvcrt = False

    if use_msvcrt:
        print("(Streaming) Press 'i' to interrupt, 'p' to pause/resume.")
        while not stop_event.is_set() and not interrupt_event.is_set():
            try:
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch.lower() == 'i':
                        interrupt_event.set()
                        print("\nInterrupt requested by user.")
                        break
                    if ch.lower() == 'p' or ch == ' ':
                        if pause_event.is_set():
                            pause_event.clear()
                            print("\nResuming playback.")
                        else:
                            pause_event.set()
                            print("\nPlayback paused. Press 'p' to resume.")
            except Exception:
                pass
            time.sleep(0.05)
        return

    # POSIX fallback
    try:
        import sys, select, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        print("(Streaming) Press 'i' to interrupt, 'p' to pause/resume.")
        while not stop_event.is_set() and not interrupt_event.is_set():
            dr, dw, de = select.select([sys.stdin], [], [], 0.05)
            if dr:
                ch = sys.stdin.read(1)
                if ch.lower() == 'i':
                    interrupt_event.set()
                    print("\nInterrupt requested by user.")
                    break
                if ch.lower() == 'p' or ch == ' ':
                    if pause_event.is_set():
                        pause_event.clear()
                        print("\nResuming playback.")
                    else:
                        pause_event.set()
                        print("\nPlayback paused. Press 'p' to resume.")
    except Exception:
        # No usable stdin controls — silently return
        return
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            pass


def _playback_worker(play_queue, pause_event, stop_event, play_handles, stream_delay=None):
    """Sequential playback worker that respects pause and stop events.

    Playback items are tuples (audio_np, sample_rate). A sentinel (None, None)
    signals end of stream.
    """
    try:
        while not stop_event.is_set():
            try:
                item = play_queue.get(timeout=0.1)
            except Exception:
                continue
            if item is None:
                break
            audio_np, sr = item
            # Wait while paused
            while pause_event.is_set() and not stop_event.is_set():
                time.sleep(0.1)
            if stop_event.is_set():
                break

            # Play synchronously so we can pause/resume and stop reliably
            played_handle = None
            try:
                import simpleaudio as sa
                data = (audio_np * 32767).astype(np.int16).tobytes()
                played_handle = sa.play_buffer(data, 1, 2, sr)
                play_handles.append(played_handle)
                # Wait for playback to finish or stop event
                while not stop_event.is_set() and not (played_handle.is_playing() == False):
                    if pause_event.is_set():
                        try:
                            played_handle.stop()
                        except Exception:
                            pass
                        # Wait until resumed, then re-play from start
                        while pause_event.is_set() and not stop_event.is_set():
                            time.sleep(0.1)
                        if stop_event.is_set():
                            break
                        # Re-play segment fully after resume
                        played_handle = sa.play_buffer(data, 1, 2, sr)
                        play_handles.append(played_handle)
                    time.sleep(0.05)
                # Ensure we wait done if not stopped
                try:
                    played_handle.wait_done()
                except Exception:
                    pass
            except Exception:
                # Fallback to sounddevice
                try:
                    import sounddevice as sd
                    sd.play(audio_np, sr)
                    # Wait for playback, respecting pause and stop
                    while sd.get_stream() is not None and not stop_event.is_set():
                        if pause_event.is_set():
                            try:
                                sd.stop()
                            except Exception:
                                pass
                            while pause_event.is_set() and not stop_event.is_set():
                                time.sleep(0.1)
                            if stop_event.is_set():
                                break
                            sd.play(audio_np, sr)
                        time.sleep(0.05)
                    try:
                        sd.wait()
                    except Exception:
                        pass
                except Exception:
                    # Last resort: write temp file and block until user can play
                    try:
                        import tempfile
                        temp_dir = Path(tempfile.gettempdir())
                        temp_file = temp_dir / f"stream_seg_{int(time.time() * 1000)}.wav"
                        sf.write(str(temp_file), audio_np, sr)
                        print(f"Saved segment to {temp_file} — open it in your media player to hear streamed audio.")
                    except Exception:
                        print("Unable to play segment; skipping.")

            # Inter-segment delay
            if stream_delay and stream_delay > 0 and not stop_event.is_set():
                elapsed = 0.0
                while elapsed < stream_delay and not stop_event.is_set():
                    time.sleep(0.05)
                    elapsed += 0.05
            # If stop_event is set, try to stop any handles
            if stop_event.is_set():
                try:
                    _stop_play_handles(play_handles)
                except Exception:
                    pass
    except Exception as e:
        print(f"Playback worker error: {e}")
        import traceback
        traceback.print_exc()


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
                    # No watchdog timer (unlimited generation allowed)
                    try:
                        generator = model(text, voice=str(voice_path), speed=speed, split_pattern=r'\n+')
                    except (ValueError, TypeError, RuntimeError) as e:
                        print(f"Error initializing speech generator: {e}")
                        # watchdog removed (no time limits)
                        continue
                    except Exception as e:
                        print(f"Unexpected error initializing generator: {type(e).__name__}: {e}")
                        # watchdog removed (no time limits)
                        continue

                    # When streaming, prefetch the next segment in a background thread so
                    # playback can start immediately without waiting for generation.
                    from queue import Queue
                    q = Queue(maxsize=2)
                    stop_event = threading.Event()
                    producer = threading.Thread(target=_prefetch_generator_to_queue, args=(generator, q, stop_event), daemon=True)
                    producer.start()

                    # Playback queue and controls
                    play_queue = Queue()
                    pause_event = threading.Event()  # when set -> paused
                    ctrl_stop_event = threading.Event()
                    playback = None
                    listener = None
                    if stream:
                        playback = threading.Thread(target=_playback_worker, args=(play_queue, pause_event, ctrl_stop_event, play_handles, stream_delay), daemon=True)
                        playback.start()
                        # Start control listener (pause/interrupt)
                        listener = threading.Thread(target=_control_listener, args=(interrupt_event if 'interrupt_event' in locals() else threading.Event(), pause_event, ctrl_stop_event), daemon=True)
                        listener.start()

                    with tqdm(desc="Generating speech") as pbar:
                        while True:
                            gs, ps, audio = q.get()

                            # Check for sentinel / end / errors
                            if gs is None and ps is None and audio is None:
                                # signal playback that generation finished
                                if stream:
                                    try:
                                        play_queue.put(None)
                                    except Exception:
                                        pass
                                break
                            if gs == "__ERR__":
                                print(f"Error while generating segment: {ps}")
                                stop_event.set()
                                if stream:
                                    try:
                                        play_queue.put(None)
                                    except Exception:
                                        pass
                                break

                            if audio is not None:
                                audio_tensor = audio if isinstance(audio, torch.Tensor) else torch.from_numpy(audio).float()
                                all_audio.append(audio_tensor)
                                print(f"\nGenerated segment: {gs}")
                                if ps:
                                    print(f"Phonemes: {ps}")
                                # Enqueue segment for the playback worker if streaming
                                if stream:
                                    try:
                                        audio_np = audio_tensor.numpy()
                                        play_queue.put((audio_np, SAMPLE_RATE))
                                    except Exception as e:
                                        print(f"Error while enqueueing segment for playback: {e}")
                                pbar.update(1)

                    # Ensure producer thread is stopped
                    stop_event.set()
                    try:
                        producer.join(timeout=1)
                    except Exception:
                        pass
                    generation_complete = True
                    # watchdog removed (no time limits)
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
                        text = extract_text_from_epub(file_path, select_chapter=True)
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
                voice_path = Path("voices").resolve() / f"{voice}.pt"
                if not voice_path.exists():
                    print(f"Error: Voice file not found: {voice_path}")
                    continue

                # Allow restarting with a different chapter on interrupt
                current_text = text
                while True:
                    all_audio = []
                    play_handles = []
                    start_time = time.time()
                    segment_start_time = start_time

                    try:
                        import threading
                        generation_complete = False

                        try:
                            generator = model(current_text, voice=str(voice_path), speed=speed, split_pattern=r'\n+')
                        except (ValueError, TypeError, RuntimeError) as e:
                            print(f"Error initializing speech generator: {e}")
                            continue
                        except Exception as e:
                            print(f"Unexpected error initializing generator: {type(e).__name__}: {e}")
                            continue

                        # Prefetch producer/consumer queue
                        from queue import Queue
                        q = Queue(maxsize=2)
                        stop_event = threading.Event()
                        interrupt_event = threading.Event()

                        producer = threading.Thread(target=_prefetch_generator_to_queue, args=(generator, q, stop_event), daemon=True)
                        producer.start()

                        # Start interrupt listener when streaming is active
                        listener = None
                        if stream:
                            listener = threading.Thread(target=_interrupt_listener, args=(interrupt_event, stop_event), daemon=True)
                            listener.start()

                        with tqdm(desc="Generating speech") as pbar:
                            while True:
                                # If the user requested an interrupt, stop producing and break
                                if interrupt_event.is_set():
                                    print("\nInterrupt detected; stopping generation...")
                                    stop_event.set()
                                    break

                                try:
                                    gs, ps, audio = q.get()
                                except Exception:
                                    # Queue get blocking; continue to check interrupt
                                    continue

                                # detect end of generation
                                if gs is None and ps is None and audio is None:
                                    break
                                if gs == "__ERR__":
                                    print(f"Error while generating segment: {ps}")
                                    stop_event.set()
                                    break

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

                        # Clean up producer/listener
                        stop_event.set()
                        try:
                            producer.join(timeout=1)
                        except Exception:
                            pass
                        if listener is not None:
                            try:
                                listener.join(timeout=0.5)
                            except Exception:
                                pass

                        # Stop playback worker if running
                        try:
                            ctrl_stop_event.set()
                            if playback is not None:
                                # Push sentinel to tell playback to exit
                                try:
                                    play_queue.put(None)
                                except Exception:
                                    pass
                                playback.join(timeout=1)
                        except Exception:
                            pass

                        generation_complete = True

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

                    # If the user interrupted, give them choices (only relevant for EPUBs)
                    if 'interrupt_event' in locals() and interrupt_event.is_set():
                        # Stop any active audio playback
                        _stop_play_handles(play_handles)
                        print("\nStreaming interrupted. Options:\n  c - choose another chapter\n  r - restart this chapter\n  Enter - cancel and return to menu")
                        choice2 = input('> ').strip().lower()
                        if choice2 == 'c':
                            # Re-run chapter selection
                            new_text = extract_text_from_epub(file_path, select_chapter=True)
                            if not new_text.strip():
                                print("No chapter selected; returning to menu.")
                                break
                            current_text = new_text
                            print("Starting new chapter...")
                            continue
                        elif choice2 == 'r':
                            print("Restarting current chapter...")
                            continue
                        else:
                            print("Cancelled. Returning to main menu.")
                            break

                    # Otherwise proceed to save the generated audio
                    if all_audio:
                        try:
                            if len(all_audio) == 1:
                                final_audio = all_audio[0]
                            else:
                                try:
                                    final_audio = torch.cat(all_audio, dim=0)
                                except RuntimeError as e:
                                    print(f"Error concatenating audio segments: {e}")
                                    break
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
                    break

            elif choice == "5":
                # Interactive streaming chat with local Ollama neural-chat
                try:
                    ollama_stream_chat_mode(model, device)
                except Exception as e:
                    print(f"Error in Ollama chat mode: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()

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
