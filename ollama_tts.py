"""ollama_tts.py — small interactive tool that connects to a local Ollama model
and optionally speaks responses using the local Kokoro TTS.

Usage:
  python ollama_tts.py               # starts interactive loop
  python ollama_tts.py --send "hi"  # non-interactive one-shot send

Behavior:
- Tries the HTTP API at --url (default: http://localhost:11434) and uses it if reachable.
- Falls back to the `ollama` CLI (invokes `ollama run <model> <message>`) when HTTP is not available.
- If --tts is set, the script will load the Kokoro model and speak received chunks as they arrive.

Designed to be simple and robust for local usage.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests
import numpy as np
import torch

from models import build_model, list_available_voices

# Minimal playback helper (tries simpleaudio, sounddevice, or write a temp WAV)
def play_audio_segment(audio_np: np.ndarray, sample_rate: int):
    try:
        import simpleaudio as sa
        data = (audio_np * 32767).astype(np.int16).tobytes()
        handle = sa.play_buffer(data, 1, 2, sample_rate)
        handle.wait_done()
        return
    except Exception:
        pass
    try:
        import sounddevice as sd
        sd.play(audio_np, sample_rate)
        sd.wait()
        return
    except Exception:
        pass
    # fallback
    import tempfile
    import soundfile as sf
    tf = Path(tempfile.gettempdir()) / f"ollama_tts_{int(time.time()*1000)}.wav"
    sf.write(str(tf), audio_np, 24000)
    print(f"Saved TTS segment to {tf}")


def synthesize_and_play(tts_model, text: str, voice_path: Path, speed: float = 1.0):
    try:
        gen = tts_model(text, voice=str(voice_path), speed=speed, split_pattern=r'\n+')
        for gs, ps, audio in gen:
            if audio is None:
                continue
            audio_np = audio.numpy() if isinstance(audio, torch.Tensor) else np.array(audio, dtype=np.float32)
            # normalize
            audio_np = np.clip(audio_np, -1.0, 1.0).astype(np.float32)
            try:
                play_audio_segment(audio_np, 24000)
            except Exception as e:
                print(f"Error playing segment: {e}")
    except Exception as e:
        print(f"Error in TTS synth: {e}")


def http_available(url: str) -> bool:
    try:
        r = requests.get(url.rstrip('/') + '/api/health', timeout=1)
        return r.status_code == 200
    except Exception:
        try:
            r = requests.get(url.rstrip('/') + '/api/models', timeout=1)
            return r.status_code == 200
        except Exception:
            return False


def send_http_stream(url: str, model_name: str, prompt: str, tts_model=None, voice_path: Optional[Path]=None, speed=1.0):
    sess = requests.Session()
    payload = {"model": model_name, "prompt": prompt, "stream": True}
    try:
        r = sess.post(url.rstrip('/') + '/api/chat', json=payload, stream=True, timeout=10)
        if r.status_code != 200:
            print(f"HTTP error {r.status_code}: {r.text[:400]}")
            return
    except Exception as e:
        print(f"HTTP error: {e}")
        return

    print("Assistant: ", end='', flush=True)
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        raw = line.strip()
        if raw.startswith('data:'):
            raw = raw[len('data:'):].strip()
        # Try JSON parse
        text_chunk = None
        try:
            jobj = json.loads(raw)
            if isinstance(jobj, dict):
                text_chunk = (jobj.get('response') or jobj.get('content') or jobj.get('text') or jobj.get('output') or jobj.get('message') or jobj.get('delta'))
                if text_chunk is None and 'choices' in jobj and isinstance(jobj['choices'], list):
                    try:
                        text_chunk = ''.join([c.get('delta', {}).get('content','') if isinstance(c, dict) else '' for c in jobj['choices']])
                    except Exception:
                        text_chunk = None
        except Exception:
            text_chunk = raw

        if text_chunk:
            print(text_chunk, end='', flush=True)
            if tts_model and voice_path:
                synthesize_and_play(tts_model, text_chunk, voice_path, speed)
    print()  # newline after completion


def send_cli_once(ollama_cmd: str, model_name: str, prompt: str) -> Optional[str]:
    try:
        proc = subprocess.run([ollama_cmd, 'run', model_name, prompt], capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            print(f"CLI error (exit {proc.returncode}): {proc.stderr[:400]}")
            return None
        return proc.stdout
    except FileNotFoundError:
        print(f"`{ollama_cmd}` not found on PATH")
        return None
    except Exception as e:
        print(f"CLI error: {e}")
        return None


def interactive_loop(args):
    tts_model = None
    voice_path = None
    if not args.no_tts:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading TTS model on {device}...")
        tts_model = build_model(Path('kokoro-v1_0.pth'), device)
        voices = list_available_voices()
        if args.voice:
            voice = args.voice
        else:
            print("Available voices:")
            for i, v in enumerate(voices, 1):
                print(f"{i}. {v}")
            sel = input("Select voice number (or press Enter for default 'af_bella'): ").strip()
            if not sel:
                voice = 'af_bella'
            else:
                try:
                    voice = voices[int(sel)-1]
                except Exception:
                    print("Invalid selection; using default 'af_bella'")
                    voice = 'af_bella'
        voice_path = Path('voices').resolve() / f"{voice}.pt"
        if not voice_path.exists():
            print(f"Voice not found: {voice_path}; disabling TTS")
            tts_model = None
            voice_path = None

    use_http = http_available(args.url)
    if use_http:
        print(f"Using Ollama HTTP API at {args.url}")
    else:
        print("HTTP API not available; will use local `ollama` CLI per message if present.")

    while True:
        try:
            msg = input("You: ").strip()
        except EOFError:
            print()
            break
        if not msg:
            print("Exiting.")
            break

        if use_http:
            send_http_stream(args.url, args.model, msg, tts_model=tts_model, voice_path=voice_path, speed=args.speed)
        else:
            resp = send_cli_once(args.cli, args.model, msg)
            if resp:
                print(resp)
                if tts_model and voice_path:
                    synthesize_and_play(tts_model, resp, voice_path, args.speed)


def main():
    parser = argparse.ArgumentParser(description='Interactive Ollama + Kokoro TTS helper')
    parser.add_argument('--url', default='http://localhost:11434', help='Ollama HTTP API base URL')
    parser.add_argument('--model', default='neural-chat', help='Ollama model name')
    parser.add_argument('--cli', default='ollama', help='Path to ollama CLI')
    parser.add_argument('--no-tts', action='store_true', help='Disable TTS playback')
    parser.add_argument('--voice', default=None, help='Voice id to use for TTS (default interactive chooser)')
    parser.add_argument('--speed', type=float, default=1.0, help='TTS speed multiplier')
    parser.add_argument('--send', '-s', help='Send a single message and exit (non-interactive)')
    args = parser.parse_args()

    if args.send:
        # One-shot
        if http_available(args.url):
            send_http_stream(args.url, args.model, args.send)
        else:
            resp = send_cli_once(args.cli, args.model, args.send)
            if resp:
                print(resp)
        return

    interactive_loop(args)


if __name__ == '__main__':
    main()
