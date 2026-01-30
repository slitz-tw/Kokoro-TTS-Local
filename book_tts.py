"""book_tts.py — Batch TTS from EPUB or PDF, saving per-chapter audio files.

Usage example:
  python book_tts.py --input "books/MyBook.epub" --voice af_bella --outdir outputs --speed 1.0

Behavior:
 - EPUB: splits into document items (chapters) using ebooklib/BeautifulSoup and uses available heading tags for chapter titles.
 - PDF: attempts to detect "Chapter" headings by scanning pages; falls back to page-chunks or a single-file output.
 - Output files are named: "<Book Title> - <Chapter Title>.wav" (invalid filename chars are sanitized).

This script uses the same model API as the project's `tts_demo.py` and reuses `save_audio_with_retry` for robust file writes.
"""

from pathlib import Path
import argparse
import re
import sys
from typing import List, Tuple

import torch
from tqdm import tqdm

from models import build_model, list_available_voices
import tts_demo


SAMPLE_RATE = tts_demo.SAMPLE_RATE
DEFAULT_MODEL = tts_demo.DEFAULT_MODEL_PATH


def sanitize_filename(name: str) -> str:
    # Replace invalid/special characters for filenames
    name = re.sub(r'[\\/:*?"<>|\n\r\t]+', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def extract_chapters_from_epub(epub_path: Path) -> Tuple[str, List[Tuple[str, str]]]:
    """Return (book_title, list of (chapter_title, chapter_text))"""
    from ebooklib import epub
    from bs4 import BeautifulSoup
    import ebooklib

    book = epub.read_epub(str(epub_path))

    # Try metadata title
    title_meta = None
    try:
        metas = book.get_metadata('DC', 'title')
        if metas:
            # metas is list of tuples (value, attrs)
            title_meta = metas[0][0]
    except Exception:
        title_meta = None

    book_title = title_meta or epub_path.stem

    doc_items = [it for it in book.get_items() if it.get_type() == ebooklib.ITEM_DOCUMENT]
    chapters = []
    for i, item in enumerate(doc_items, start=1):
        soup = BeautifulSoup(item.get_content(), 'html.parser')
        title_tag = soup.find(['h1', 'h2', 'h3', 'title'])
        chap_title = title_tag.get_text(strip=True) if title_tag and title_tag.get_text(strip=True) else f'Chapter {i}'
        text = soup.get_text(separator='\n', strip=True)
        chapters.append((chap_title, text))
    if not chapters:
        # Fallback: use entire book as single chapter
        txt = tts_demo.extract_text_from_epub(epub_path, select_chapter=False)
        chapters = [(book_title, txt)]
    return book_title, chapters


def extract_chapters_from_pdf(pdf_path: Path) -> Tuple[str, List[Tuple[str, str]]]:
    """Return (document_title, list of (chapter_title, text))

    Heuristic: scan page-by-page and start a new chapter whenever a page contains a line
    that looks like a chapter heading (e.g. starts with "Chapter" or similar).
    """
    import PyPDF2

    reader = PyPDF2.PdfReader(str(pdf_path))
    pages_text = []
    for page in reader.pages:
        pages_text.append(page.extract_text() or "")

    # Heuristic detection
    chapters = []
    current_pages = []
    current_title = None

    heading_re = re.compile(r'^(\s{0,10}(Chapter|CHAPTER|chapter)\b[^\n]*)', re.IGNORECASE|re.MULTILINE)

    for p in pages_text:
        # If page contains a chapter heading, start a new chapter
        m = heading_re.search(p)
        if m:
            # Save previous
            if current_pages:
                chapters.append((current_title or f'Chapter {len(chapters) + 1}', '\n'.join(current_pages)))
            current_pages = [p]
            # Use the matched heading (first match) as chapter title
            t = m.group(1).strip()
            # Normalize title
            current_title = re.sub(r'\s+', ' ', t)
        else:
            current_pages.append(p)

    if current_pages:
        chapters.append((current_title or f'Chapter {len(chapters) + 1}', '\n'.join(current_pages)))

    # If no chapter headings detected, attempt to split by size (~5000 characters) or keep single
    if len(chapters) == 1 and (not heading_re.search(pages_text[0]) and sum(len(p) for p in pages_text) > 6000):
        all_text = '\n'.join(pages_text)
        chunks = []
        chunk_size = 5000
        idx = 0
        n = 1
        while idx < len(all_text):
            chunk = all_text[idx:idx + chunk_size]
            chunks.append((f'Part {n}', chunk))
            idx += chunk_size
            n += 1
        if chunks:
            chapters = chunks
    if not chapters:
        # Fallback to full text
        all_text = '\n'.join(pages_text)
        chapters = [(pdf_path.stem, all_text)]

    doc_title = pdf_path.stem
    return doc_title, chapters


def synthesize_text_to_file(model, voice: str, text: str, out_path: Path, speed: float = 1.0) -> bool:
    """Synthesize `text` with `model` and save to `out_path`. Returns True on success."""
    if not text.strip():
        print(f"Skipping empty chapter for output {out_path}")
        return False

    # check voice model file
    voice_path = Path('voices') / f"{voice}.pt"
    if not voice_path.exists():
        raise FileNotFoundError(f"Voice file not found: {voice_path}")

    # Build generator and collect segments synchronously
    try:
        gen = model(text, voice=str(voice_path), speed=speed, split_pattern=r'\n+')
    except Exception as e:
        print(f"Failed to initialize generator: {e}")
        return False

    audios = []
    for gs, ps, audio in gen:
        if gs is None and ps is None and audio is None:
            break
        if gs == "__ERR__":
            print(f"Generation error: {ps}")
            return False
        if audio is not None:
            import torch
            a = audio if isinstance(audio, torch.Tensor) else torch.from_numpy(audio).float()
            audios.append(a)

    if not audios:
        print(f"No audio generated for {out_path}")
        return False

    try:
        if len(audios) == 1:
            final = audios[0]
        else:
            final = torch.cat(audios, dim=0)
        # Use save helper from tts_demo
        ok = tts_demo.save_audio_with_retry(final.numpy(), SAMPLE_RATE, out_path)
        return ok
    except Exception as e:
        print(f"Error while saving audio to {out_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert EPUB/PDF into per-chapter audio files")
    parser.add_argument('--input', '-i', required=True, help='Path to EPUB or PDF file')
    parser.add_argument('--voice', '-v', default=None, help='Voice name (file in voices/*.pt). If omitted, use first available voice or af_bella')
    parser.add_argument('--outdir', '-o', default='outputs', help='Directory to place generated audio files')
    parser.add_argument('--model', '-m', default=str(DEFAULT_MODEL), help='Path to model .pth (default: project default)')
    parser.add_argument('--speed', type=float, default=1.0, help='Speech speed multiplier')
    parser.add_argument('--max-chars', type=int, default=200000, help='Max characters per chapter (truncate long chapters)')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(2)

    ext = input_path.suffix.lower()
    if ext not in ('.epub', '.pdf'):
        print("Unsupported input file type. Use .epub or .pdf")
        sys.exit(2)

    # Pick voice
    voices = list_available_voices()
    if args.voice:
        if args.voice not in voices:
            print(f"Voice '{args.voice}' not found. Available voices: {', '.join(voices)}")
            sys.exit(2)
        voice = args.voice
    else:
        voice = voices[0] if voices else 'af_bella'

    # Build model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model {args.model} on {device}...")
    model = build_model(Path(args.model), device)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract chapters
    if ext == '.epub':
        book_title, chapters = extract_chapters_from_epub(input_path)
    else:
        book_title, chapters = extract_chapters_from_pdf(input_path)

    book_title = sanitize_filename(book_title)
    print(f"Found {len(chapters)} chapters in '{book_title}'")

    # Synthesize per-chapter
    for idx, (chap_title, chap_text) in enumerate(tqdm(chapters, desc='Chapters'), start=1):
        # guard length
        if len(chap_text) > args.max_chars:
            print(f"Chapter '{chap_title}' is very long ({len(chap_text)} chars); truncating to {args.max_chars} chars.")
            chap_text = chap_text[:args.max_chars]

        # Prepare filename: prefer readable chapter title or fallback to Chapter N
        safe_title = sanitize_filename(chap_title) or f'Chapter {idx}'
        filename = f"{book_title} - {safe_title}.wav"
        out_path = out_dir / filename

        print(f"Generating audio for: {safe_title} -> {out_path}")
        try:
            ok = synthesize_text_to_file(model, voice, chap_text, out_path, speed=args.speed)
            if ok:
                print(f"Saved: {out_path}")
            else:
                print(f"Failed to generate '{safe_title}'")
        except Exception as e:
            print(f"Error for chapter '{safe_title}': {e}")

    print("Done.")


if __name__ == '__main__':
    main()
