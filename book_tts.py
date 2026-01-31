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


# --- Grouping helpers -------------------------------------------------------
def group_by_size(chapters, min_chars: int):
    """Merge adjacent chapters until each group has at least min_chars characters."""
    if not min_chars or min_chars <= 0:
        return chapters
    grouped = []
    curr_title = None
    curr_text = ''
    for title, text in chapters:
        if not curr_text:
            curr_title = title
            curr_text = text
        else:
            curr_text = curr_text + '\n' + text
        if len(curr_text) >= min_chars:
            grouped.append((curr_title or f'Part {len(grouped)+1}', curr_text))
            curr_text = ''
            curr_title = None
    if curr_text:
        grouped.append((curr_title or f'Part {len(grouped)+1}', curr_text))
    return grouped


def group_by_fixed(chapters, group_size: int):
    """Combine every group_size consecutive chapters into one group."""
    if not group_size or group_size <= 1:
        return chapters
    groups = []
    for i in range(0, len(chapters), group_size):
        chunk = chapters[i:i+group_size]
        title = chunk[0][0]
        text = '\n'.join(t for _, t in chunk)
        groups.append((title, text))
    return groups


def group_epub_by_heading(epub_path: Path, group_level: int = 1, min_chars: int = 8000):
    """Split EPUB by heading tags at `h{group_level}` then merge small sections by min_chars."""
    from ebooklib import epub
    from bs4 import BeautifulSoup
    import ebooklib

    book = epub.read_epub(str(epub_path))
    title_meta = None
    try:
        metas = book.get_metadata('DC', 'title')
        if metas:
            title_meta = metas[0][0]
    except Exception:
        title_meta = None
    book_title = title_meta or epub_path.stem

    doc_items = [it for it in book.get_items() if it.get_type() == ebooklib.ITEM_DOCUMENT]
    sections = []
    heading_tag = f'h{max(1, group_level)}'
    for item in doc_items:
        soup = BeautifulSoup(item.get_content(), 'html.parser')
        headings = soup.find_all(heading_tag)
        if headings:
            for h in headings:
                # Collect content until next same-level heading
                title = h.get_text(strip=True) or 'Untitled'
                parts = []
                for sib in h.next_siblings:
                    if getattr(sib, 'name', None) == heading_tag:
                        break
                    # convert Tag or NavigableString to text
                    try:
                        parts.append(sib.get_text(separator='\n', strip=True))
                    except Exception:
                        parts.append(str(sib))
                text = h.get_text(separator='\n', strip=True) + '\n' + '\n'.join([p for p in parts if p])
                sections.append((title, text))
        else:
            title_tag = soup.find(['h1', 'h2', 'h3', 'title'])
            title = title_tag.get_text(strip=True) if title_tag and title_tag.get_text(strip=True) else (item.get_name() or 'Untitled')
            text = soup.get_text(separator='\n', strip=True)
            sections.append((title, text))

    # If no sections found, fallback to whole book
    if not sections:
        txt = tts_demo.extract_text_from_epub(epub_path, select_chapter=False)
        sections = [(book_title, txt)]

    # Merge small sections
    grouped = group_by_size(sections, min_chars)
    return book_title, grouped


def group_pdf_by_regex(pdf_path: Path, chapter_regex: str = r'^(PART|CHAPTER|Chapter)\\b', min_chars: int = 8000):
    """Detect main headings by regex on PDF pages and group pages between matches."""
    import PyPDF2
    import re

    reader = PyPDF2.PdfReader(str(pdf_path))
    pages_text = [page.extract_text() or '' for page in reader.pages]
    rgx = re.compile(chapter_regex, re.IGNORECASE|re.MULTILINE)

    groups = []
    curr_title = None
    curr_pages = []

    for p in pages_text:
        m = rgx.search(p)
        if m:
            # start new group
            if curr_pages:
                groups.append((curr_title or f'Chapter {len(groups)+1}', '\n'.join(curr_pages)))
            curr_pages = [p]
            curr_title = m.group(0).strip()
        else:
            curr_pages.append(p)

    if curr_pages:
        groups.append((curr_title or f'Chapter {len(groups)+1}', '\n'.join(curr_pages)))

    # If only one group and it's large, try splitting by size
    grouped = group_by_size(groups, min_chars)
    return pdf_path.stem, grouped


def preview_and_select_chapters(chapters):
    """Show a numbered preview of chapters and allow deselecting any before processing.

    Input: chapters is a list of (title, text).
    Returns a filtered list of chapters (in same order).
    """
    if not chapters:
        return chapters

    def print_list(chaps):
        print("\nDetected chapters:\n")
        for i, (title, text) in enumerate(chaps, start=1):
            preview = ' '.join(title.split())
            preview_text = ' '.join(text[:200].split())
            print(f"{i:3d}. {preview} — {len(text)} chars — {preview_text[:140]}{'...' if len(preview_text)>140 else ''}")
        print("")

    max_idx = len(chapters)
    while True:
        print_list(chapters)
        print("Options: enter indices to EXCLUDE (e.g. 1,3-5), 'a' to accept all, 'r' to reprint, 'q' to cancel and exit.")
        choice = input('Exclude indices> ').strip().lower()
        if choice == '' or choice == 'a':
            return chapters
        if choice == 'r':
            continue
        if choice == 'q':
            return []
        # parse indices/ranges
        exclude = set()
        parts = [p.strip() for p in choice.split(',') if p.strip()]
        invalid = False
        for p in parts:
            if '-' in p:
                try:
                    a, b = p.split('-', 1)
                    a = int(a); b = int(b)
                    if a < 1 or b > max_idx or a > b:
                        invalid = True; break
                    exclude.update(range(a, b+1))
                except Exception:
                    invalid = True; break
            else:
                try:
                    v = int(p)
                    if v < 1 or v > max_idx:
                        invalid = True; break
                    exclude.add(v)
                except Exception:
                    invalid = True; break
        if invalid:
            print("Invalid selection. Try again.")
            continue
        # Build filtered list
        new_chapters = [chap for i, chap in enumerate(chapters, start=1) if i not in exclude]
        if not new_chapters:
            print("All chapters excluded — nothing to do. Returning to selection.")
            continue
        # Confirm
        print(f"Preparing to process {len(new_chapters)} chapters (excluded {len(exclude)}). Continue? [Y/n]")
        ok = input('> ').strip().lower()
        if ok in ('','y','yes'):
            return new_chapters
        else:
            print("Selection cancelled — returning to list.")
            continue


def interactive_menu(args):
    """Interactive menu to review and modify CLI options before processing.

    Shows current values and allows the user to change them or proceed.
    """
    def prompt_val(prompt, current, cast=str, allowed=None):
        while True:
            val = input(f"{prompt} [{current}]: ").strip()
            if val == "":
                return current
            try:
                v = cast(val)
            except Exception:
                print("Invalid value; try again.")
                continue
            if allowed and v not in allowed:
                print(f"Value must be one of: {allowed}")
                continue
            return v

    print("\n=== Configuration Menu ===")
    print("Press Enter to keep the current value.")

    # Positional input (file)
    while True:
        args.input = input(f"Input file (EPUB/PDF) [{args.input}]: ").strip() or args.input
        if args.input:
            from pathlib import Path
            p = Path(args.input)
            if p.exists():
                break
            print(f"File not found: {args.input}")
        else:
            print("Input is required.")

    args.voice = input(f"Voice [{args.voice}]: ").strip() or args.voice
    args.outdir = input(f"Output directory [{args.outdir}]: ").strip() or args.outdir
    args.model = input(f"Model path [{args.model}]: ").strip() or args.model

    # Numeric and choice fields
    args.speed = prompt_val("Speed", args.speed, float)
    args.max_chars = prompt_val("Max chars per chapter", args.max_chars, int)
    args.max_chapters = prompt_val("Max chapters (0=all)", args.max_chapters, int)
    args.parallel = prompt_val("Parallel workers (1=sequential)", args.parallel, int)

    # Grouping options
    group_by_choices = ['heading','size','fixed','regex','none']
    args.group_by = prompt_val("Group by (heading|size|fixed|regex|none)", args.group_by, str, allowed=group_by_choices)
    if args.group_by == 'heading':
        args.group_level = prompt_val("Heading level to use (1=H1)", args.group_level, int)
        args.group_min_chars = prompt_val("Minimum chars per grouped output", args.group_min_chars, int)
    elif args.group_by == 'size':
        args.group_min_chars = prompt_val("Minimum chars per grouped output", args.group_min_chars, int)
    elif args.group_by == 'fixed':
        args.group_size = prompt_val("Group size (number of items to combine)", args.group_size, int)
    elif args.group_by == 'regex':
        args.chapter_regex = input(f"Chapter regex [{args.chapter_regex}]: ").strip() or args.chapter_regex
        args.group_min_chars = prompt_val("Minimum chars per grouped output", args.group_min_chars, int)

    print("\nConfiguration complete. Proceeding with these values:")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    ok = input("\nProceed? (Y/n): ").strip().lower()
    if ok in ('n','no'):
        print("Restarting configuration...")
        return interactive_menu(args)
    return args


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


# --- Worker helpers for parallel generation ---------------------------------
_WORKER_MODEL = None
_WORKER_MODEL_PATH = None
_WORKER_DEVICE = None

def _init_worker(model_path, device):
    """Initializer for worker processes. Builds a model once per process."""
    global _WORKER_MODEL, _WORKER_MODEL_PATH, _WORKER_DEVICE
    _WORKER_MODEL_PATH = model_path
    _WORKER_DEVICE = device
    import os
    try:
        # If worker should use CPU, hide GPU devices to avoid accidental CUDA init and fork-related reinitialization issues
        if device == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        _WORKER_MODEL = build_model(Path(model_path), device)
        print(f"Worker initialized model on {device}")
    except Exception as e:
        print(f"Worker failed to initialize model: {e}")
        _WORKER_MODEL = None


def _worker_synthesize(task):
    """Worker entrypoint. `task` is a tuple:
    (idx, chap_title, chap_text, out_path_str, voice, speed, max_chars)

    Returns (idx, chap_title, ok, error_message_or_None)
    """
    import torch
    from pathlib import Path

    idx, chap_title, chap_text, out_path_str, voice, speed, max_chars = task

    try:
        if _WORKER_MODEL is None:
            return (idx, chap_title, False, "worker model not initialized")
        # Truncate if needed
        if max_chars and len(chap_text) > max_chars:
            chap_text = chap_text[:max_chars]

        voice_path = Path('voices') / f"{voice}.pt"
        if not voice_path.exists():
            return (idx, chap_title, False, f"voice not found: {voice_path}")

        gen = _WORKER_MODEL(chap_text, voice=str(voice_path), speed=speed, split_pattern=r'\n+')

        audios = []
        for gs, ps, audio in gen:
            if gs is None and ps is None and audio is None:
                break
            if gs == "__ERR__":
                return (idx, chap_title, False, f"generation error: {ps}")
            if audio is not None:
                a = audio if isinstance(audio, torch.Tensor) else torch.from_numpy(audio).float()
                audios.append(a)

        if not audios:
            return (idx, chap_title, False, "no audio generated")

        final = audios[0] if len(audios) == 1 else torch.cat(audios, dim=0)
        out_path = Path(out_path_str)
        ok = tts_demo.save_audio_with_retry(final.numpy(), SAMPLE_RATE, out_path)
        return (idx, chap_title, ok, None if ok else "save failed")

    except Exception as e:
        return (idx, chap_title, False, str(e))


# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert EPUB/PDF into per-chapter audio files")
    parser.add_argument('input', help='Path to EPUB or PDF file (EPUB or PDF)')
    parser.add_argument('--voice', '-v', default=None, help='Voice name (file in voices/*.pt). If omitted, use first available voice or af_bella')
    parser.add_argument('--outdir', '-o', default='.', help='Directory to place generated audio files (default: workspace root)')
    parser.add_argument('--model', '-m', default=str(DEFAULT_MODEL), help='Path to model .pth (default: project default)')
    parser.add_argument('--speed', type=float, default=1.0, help='Speech speed multiplier')
    parser.add_argument('--max-chars', type=int, default=200000, help='Max characters per chapter (truncate long chapters)')
    parser.add_argument('--max-chapters', '-n', type=int, default=3, help='Max number of chapters to process (default 3, 0 for all)')
    parser.add_argument('--parallel', '-p', type=int, default=3, help='Number of worker processes for parallel generation (default 3). Use 1 for sequential.')

    # Grouping options
    parser.add_argument('--group-by', choices=['heading','size','fixed','regex','none'], default='none', help='How to group subchapters into main chapters')
    parser.add_argument('--group-level', type=int, default=1, help='Heading level to treat as main (for EPUB heading grouping)')
    parser.add_argument('--group-min-chars', type=int, default=8000, help='Minimum characters per grouped output; smaller pieces will be merged (size fallback)')
    parser.add_argument('--group-size', type=int, default=3, help='When using fixed grouping, how many consecutive items to combine')
    parser.add_argument('--chapter-regex', type=str, default=r'^(PART|CHAPTER|Chapter)\\b', help='Regex for detecting main headings in PDFs (used with --group-by regex)')

    args = parser.parse_args()
    # Present interactive menu so all CLI options can be reviewed/changed at runtime
    args = interactive_menu(args)

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

    # Extract raw chapters
    if ext == '.epub':
        book_title, raw_chapters = extract_chapters_from_epub(input_path)
    else:
        book_title, raw_chapters = extract_chapters_from_pdf(input_path)

    # Apply grouping if requested
    if args.group_by == 'heading' and ext == '.epub':
        book_title, chapters = group_epub_by_heading(input_path, group_level=args.group_level, min_chars=args.group_min_chars)
        print(f"Grouped into {len(chapters)} main chapters using heading level {args.group_level}")
    elif args.group_by == 'regex' and ext == '.pdf':
        book_title, chapters = group_pdf_by_regex(input_path, chapter_regex=args.chapter_regex, min_chars=args.group_min_chars)
        print(f"Grouped into {len(chapters)} main chapters using regex '{args.chapter_regex}'")
    elif args.group_by == 'fixed':
        chapters = group_by_fixed(raw_chapters, args.group_size)
        print(f"Grouped into {len(chapters)} main chapters by fixed size {args.group_size}")
    elif args.group_by == 'size':
        chapters = group_by_size(raw_chapters, args.group_min_chars)
        print(f"Grouped into {len(chapters)} main chapters by size min {args.group_min_chars}")
    else:
        chapters = raw_chapters

    book_title = sanitize_filename(book_title)
    print(f"Found {len(chapters)} chapters in '{book_title}'")

    # Show preview and let user deselect chapters before proceeding
    chapters = preview_and_select_chapters(chapters)
    if not chapters:
        print("No chapters selected; exiting.")
        sys.exit(0)

    # Limit chapters if requested
    if args.max_chapters and args.max_chapters > 0:
        chapters = chapters[:args.max_chapters]
        print(f"Processing first {len(chapters)} chapters (max_chapters={args.max_chapters})")

    # Parallel processing when requested
    if args.parallel and args.parallel > 1 and len(chapters) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing, os
        print(f"Processing up to {args.parallel} chapters in parallel (workers will initialize models on CPU)...")
        # If CUDA is available in the main process, prefer 'spawn' start method to avoid CUDA reinitialization errors in forked children
        try:
            if torch.cuda.is_available():
                current = multiprocessing.get_start_method(allow_none=True)
                if current != 'spawn':
                    multiprocessing.set_start_method('spawn')
                    print("Set multiprocessing start method to 'spawn' to avoid CUDA fork issues.")
        except RuntimeError as e:
            print(f"Warning: Could not set multiprocessing start method to 'spawn': {e}. If you encounter CUDA errors, run with --parallel 1 (sequential).")
        except Exception as e:
            print(f"Warning: Error while configuring multiprocessing start method: {e}")

        tasks = []
        for idx, (chap_title, chap_text) in enumerate(chapters, start=1):
            # Guard length early
            if len(chap_text) > args.max_chars:
                print(f"Chapter '{chap_title}' is very long ({len(chap_text)} chars); truncating to {args.max_chars} chars.")
                chap_text = chap_text[:args.max_chars]
            safe_title = sanitize_filename(chap_title) or f'Chapter {idx}'
            filename = f"{book_title} - {safe_title}.wav"
            out_path = (out_dir / filename).resolve()
            tasks.append((idx, chap_title, chap_text, str(out_path), voice, args.speed, args.max_chars))

        with ProcessPoolExecutor(max_workers=args.parallel, initializer=_init_worker, initargs=(str(args.model), 'cpu')) as ex:
            futures = {ex.submit(_worker_synthesize, task): task for task in tasks}
            for fut in as_completed(futures):
                idx, chap_title, ok, err = fut.result()
                safe_title = sanitize_filename(chap_title) or f'Chapter {idx}'
                out_path = out_dir / f"{book_title} - {safe_title}.wav"
                if ok:
                    print(f"Saved: {out_path}")
                else:
                    print(f"Failed: {safe_title} — {err}")

    else:
        # Sequential processing using the main model
        for idx, (chap_title, chap_text) in enumerate(tqdm(chapters, desc='Chapters'), start=1):
            if len(chap_text) > args.max_chars:
                print(f"Chapter '{chap_title}' is very long ({len(chap_text)} chars); truncating to {args.max_chars} chars.")
                chap_text = chap_text[:args.max_chars]
            safe_title = sanitize_filename(chap_title) or f'Chapter {idx}'
            filename = f"{book_title} - {safe_title}.wav"
            out_path = (out_dir / filename).resolve()
            print(f"Generating audio for: {safe_title} -> {out_path}")
            try:
                if model is None:
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    model_local = build_model(Path(args.model), device)
                else:
                    model_local = model
                ok = synthesize_text_to_file(model_local, voice, chap_text, out_path, speed=args.speed)
                if ok:
                    print(f"Saved: {out_path}")
                else:
                    print(f"Failed to generate '{safe_title}'")
            except Exception as e:
                print(f"Error for chapter '{safe_title}': {e}")

    print("Done.")


if __name__ == '__main__':
    main()
