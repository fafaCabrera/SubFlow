##############UnifiedSubFlow.py#####################
import os
import sys
import subprocess
import argparse
from pathlib import Path
import srt
from datetime import datetime, timedelta
from transformers import MarianMTModel, MarianTokenizer
import torch
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import whisper
from whisper.utils import WriteSRT

# ============================
# Configurable Section
# ============================
IGNORE_FOLDERS = ["Los Simpsons", "The Expanse Complete Series", "The Leftovers"]
OPENSUBTITLES_USER = "username"  # Replace with your OpenSubtitles username
OPENSUBTITLES_PASSWORD = "password"  # Replace with your OpenSubtitles password
TARGET_LANGUAGE = "es"  # Default target language (e.g., "es" for Spanish, "fr" for French)
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv'}

# Keywords to ignore (sample, trailer, etc.)
IGNORE_KEYWORDS = ['sample', 'trailer', 'clip', 'demo']


def is_ignored_file(filename):
    """Check if the file should be ignored based on keywords"""
    name_lower = filename.lower()
    return any(keyword in name_lower for keyword in IGNORE_KEYWORDS)


def install_dependencies():
    """Install required packages with user feedback"""
    required_packages = {
        'subliminal': 'subliminal',
        'transformers': 'transformers',
        'torch': 'torch',
        'srt': 'srt',
        'sentencepiece': 'sentencepiece',
        'watchdog': 'watchdog',
        'whisper': 'openai-whisper',
        'ffmpeg-python': 'ffmpeg-python'
    }
    print("Checking dependencies...")
    installed = 0
    for package, install_name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {package} already installed")
        except ImportError:
            print(f"● Installing {package}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", install_name],
                stdout=subprocess.DEVNULL
            )
            print(f"✓ {package} installed successfully")
            installed += 1
    if installed == 0:
        print("All dependencies are already installed!")
    else:
        print(f"Installed {installed} new packages")


def should_skip_folder(path):
    """Check if the folder should be skipped based on IGNORE_FOLDERS"""
    for folder in IGNORE_FOLDERS:
        if folder.lower() in path.lower():
            return True
    return False


def get_model_name(src_lang, tgt_lang):
    """Get model name with user feedback"""
    model_map = {
        ('en', 'es'): 'Helsinki-NLP/opus-mt-en-es',
        ('en', 'fr'): 'Helsinki-NLP/opus-mt-en-fr',
    }
    return model_map.get((src_lang, tgt_lang), f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}')


def translate_text(text, model, tokenizer, device):
    """Translate individual text segment with GPU support"""
    try:
        batch = tokenizer([text], return_tensors="pt", truncation=True).to(device)
        with torch.inference_mode():
            gen = model.generate(**batch)
        return tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            torch.cuda.empty_cache()
            return translate_text(text, model, tokenizer, device)
        raise


def add_credits_to_subtitle(subtitle_path, mode):
    """Add credits to the subtitle file based on the mode."""
    credit_lines = {
        "opensubtitles_en": (
            "Descargado con github.com/fafaCabrera/SubFlow\n"
            "Modo OpenSubtitles"
        ),
        "opensubtitles_es": (
            "Descargado con github.com/fafaCabrera/SubFlow\n"
            "Modo OpenSubtitles"
        ),
        "translated_es": (
            "Traducido con github.com/fafaCabrera/SubFlow\n"
            "Modo traducción"
        ),
        "whisper_en": (
            "Generado con Whisper\n"
            "github.com/fafaCabrera/SubFlow"
        ),
        "whisper_es": (
            "Generado con Whisper\n"
            "github.com/fafaCabrera/SubFlow"
        ),
    }
    try:
        with open(subtitle_path, 'r', encoding='utf-8') as f:
            subs = list(srt.parse(f.read()))
    except Exception as e:
        print(f"Error reading subtitle file {subtitle_path}: {e}")
        return

    credits_entry = srt.Subtitle(
        index=0,
        start=timedelta(seconds=3),
        end=timedelta(seconds=13),
        content=credit_lines[mode]
    )
    subs.insert(0, credits_entry)

    with open(subtitle_path, 'w', encoding='utf-8') as f:
        f.write(srt.compose(subs))


def generate_subtitles_with_whisper(video_path, lang):
    """Generate subtitles using Whisper"""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.dirname(video_path)
    output_path = os.path.join(video_dir, f"{base_name}.{lang}.1.srt")

    # Skip if already exists
    if os.path.exists(output_path):
        print(f"Whisper subtitle already exists, skipping generation: {output_path}")
        return output_path

    print(f"Generating subtitles with Whisper for: {video_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = whisper.load_model("small", device=device)
    result = model.transcribe(video_path, verbose=False)

    with open(output_path, 'w', encoding='utf-8') as srt_file:
        writer = WriteSRT(video_dir)
        writer.write_result(result, srt_file, options=None)

    print(f"Whisper subtitles generated: {output_path}")
    add_credits_to_subtitle(output_path, f"whisper_{lang}")
    return output_path


def download_subtitles(video_path, lang, log_files):
    """Download subtitles using subliminal."""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.dirname(video_path)
    subtitle_path = os.path.join(video_dir, f"{base_name}.{lang}.srt")

    # Skip if already exists
    if os.path.exists(subtitle_path):
        print(f"{lang.upper()} subtitle already exists: {video_path}")
        return True

    print(f"Downloading {lang.upper()} subtitles for: {video_path}")
    result1 = subprocess.run(["subliminal", "download", "-l", lang, video_path])
    result2 = subprocess.run([
        "subliminal", "--opensubtitles", OPENSUBTITLES_USER, OPENSUBTITLES_PASSWORD,
        "download", "-p", "opensubtitles", "-l", lang, video_path
    ])

    if os.path.exists(subtitle_path):
        print(f"{lang.upper()} subtitle downloaded for: {video_path}")
        log_files[f'{lang}_down_log'].write(f"{lang.upper()} subtitle downloaded for: {video_path}\n")
        add_credits_to_subtitle(subtitle_path, f"opensubtitles_{lang}")
        return True
    return False


def translate_srt(input_file, src_lang, tgt_lang):
    """Translate SRT file with proper naming and duplicate prevention."""
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    dir_name = os.path.dirname(input_file)

    # Determine output name based on input suffix
    if base_name.endswith(f'.{src_lang}.1'):
        base_main = base_name[:-len(f'.{src_lang}.1')]
        output_file = os.path.join(dir_name, f"{base_main}.{tgt_lang}.1.srt")
    elif base_name.endswith(f'.{src_lang}'):
        base_main = base_name[:-len(f'.{src_lang}')]
        output_file = os.path.join(dir_name, f"{base_main}.{tgt_lang}.srt")
    else:
        output_file = os.path.join(dir_name, f"{base_name}.{tgt_lang}.srt")

    # Skip if translation already exists
    if os.path.exists(output_file):
        print(f"Translated subtitle already exists, skipping: {output_file}")
        return output_file

    start_time = datetime.now()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nStarting translation: {src_lang} → {tgt_lang}")
    print(f"Using: {'GPU acceleration' if device.type == 'cuda' else 'CPU'} mode")
    print(f"Input file: {os.path.abspath(input_file)}")

    model_name = get_model_name(src_lang, tgt_lang)
    print(f"Loading model '{model_name}'...")
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(device)
        print(f"Model loaded on {device.type.upper()}")
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        sys.exit(1)

    with open(input_file, 'r', encoding='utf-8') as f:
        subs = list(srt.parse(f.read()))
    total_subs = len(subs)
    print(f"Found {total_subs} subtitles to translate")

    translated_subs = []
    print("\nTranslation progress:")
    for idx, sub in enumerate(subs, 1):
        original_lines = sub.content.split('\n')
        translated_lines = []
        for line in original_lines:
            if line.strip():
                translated_line = translate_text(line, model, tokenizer, device)
                translated_lines.append(translated_line)
            else:
                translated_lines.append(line)
        translated_content = '\n'.join(translated_lines)
        translated_subs.append(srt.Subtitle(
            index=sub.index,
            start=sub.start,
            end=sub.end,
            content=translated_content
        ))
        if idx % max(1, total_subs // 10) == 0 and device.type == 'cuda':
            mem_usage = torch.cuda.memory_allocated() / 1024**3
            print(f"  GPU VRAM: {mem_usage:.2f} GB used | {idx}/{total_subs} ({idx / total_subs:.0%})")
        else:
            progress = idx / total_subs
            print(f"  █ {idx}/{total_subs} ({progress:.0%})", end='\r')

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(srt.compose(translated_subs))

    add_credits_to_subtitle(output_file, "translated_es")

    duration = datetime.now() - start_time
    print(f"\nTranslation completed in {duration.total_seconds():.1f} seconds")
    print(f"Output file created: {os.path.abspath(output_file)}")
    return output_file


def process_video_file(video_path, log_files, tgt_lang, is_daemon=False):
    """Process a single video file"""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.dirname(video_path)

    # Skip if file is a sample/clip/etc.
    if is_ignored_file(base_name):
        print(f"Ignoring sample/clip file: {video_path}")
        log_files['log'].write(f"Ignoring sample file: {video_path}\n")
        return

    # Check if .es.srt already exists (from any source)
    es_srt = os.path.join(video_dir, f"{base_name}.{tgt_lang}.srt")
    if os.path.exists(es_srt):
        print(f"{tgt_lang.upper()} subtitle found for: {video_path}")
        log_files['log'].write(f"{tgt_lang.upper()} subtitle found for: {video_path}\n")

    # --- Paso 1: Descargar .es.srt y .en.srt ---
    downloaded_es = download_subtitles(video_path, tgt_lang, log_files)
    downloaded_en = download_subtitles(video_path, "en", log_files)

    en_srt = os.path.join(video_dir, f"{base_name}.en.srt")

    # Si hay .en.srt pero no .es.srt, traducir
    if os.path.exists(en_srt) and not os.path.exists(es_srt):
        print("EN Found, Translating....")
        log_files['en_only_log'].write(f"Needs Translate: {video_path}\n")
        translate_srt(en_srt, "en", tgt_lang)

    # --- Paso 2: Modo daemon → generar con Whisper y traducir ---
    if is_daemon:
        print("Daemon mode: Generating Whisper subtitles...")
        whisper_en_srt = generate_subtitles_with_whisper(video_path, "en")
        if os.path.exists(whisper_en_srt):
            translate_srt(whisper_en_srt, "en", tgt_lang)
    else:
        # Modo escaneo: registrar si faltan subtítulos
        if not downloaded_es and not downloaded_en:
            print(f"No subtitles found or downloaded for: {video_path}")
            log_files['failed_log'].write(f"Faltan subtítulos en español para: {video_path}\n")
            log_files['log'].write(f"Faltan subtítulos en español para: {video_path}\n")


def process_folder(folder_path, log_files, tgt_lang, is_daemon=False):
    """Process all video files in a folder and its subfolders."""
    for root, dirs, files in os.walk(folder_path):
        if should_skip_folder(root):
            log_files['log'].write(f"Hardcoded Skipping \"{root}\"\n")
            continue
        for file in files:
            if file.endswith(tuple(VIDEO_EXTENSIONS)):
                video_path = os.path.join(root, file)
                process_video_file(video_path, log_files, tgt_lang, is_daemon)


def monitor_folders(folder_paths, log_files, tgt_lang):
    """Monitor multiple folders for new video files."""
    class VideoFileHandler(FileSystemEventHandler):
        def on_created(self, event):
            if not event.is_directory:
                file_path = event.src_path
                _, ext = os.path.splitext(file_path)
                if ext.lower() in VIDEO_EXTENSIONS:
                    if is_ignored_file(os.path.basename(file_path)):
                        print(f"Ignoring sample file on creation: {file_path}")
                        return
                    print(f"New video file detected: {file_path}")
                    process_video_file(file_path, log_files, tgt_lang, is_daemon=True)

        def on_moved(self, event):
            if not event.is_directory:
                dest_path = event.dest_path
                _, ext = os.path.splitext(dest_path)
                if ext.lower() in VIDEO_EXTENSIONS:
                    if is_ignored_file(os.path.basename(dest_path)):
                        print(f"Ignoring sample file on move: {dest_path}")
                        return
                    print(f"Renamed video file detected: {dest_path}")
                    process_video_file(dest_path, log_files, tgt_lang, is_daemon=True)

    observers = []
    for folder_path in folder_paths:
        observer = Observer()
        event_handler = VideoFileHandler()
        observer.schedule(event_handler, path=folder_path, recursive=True)
        observer.start()
        observers.append(observer)

    print("Monitoring folders for new video files... (Press Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping monitor...")
        for observer in observers:
            observer.stop()
        for observer in observers:
            observer.join()


def main(input_paths, log_name, skip_scan=False):
    """Main function to process video files or folders"""
    if any(os.path.isdir(path) for path in input_paths):
        install_dependencies()

    script_dir = Path(__file__).parent.absolute()
    logs_dir = script_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    log_files = {
        'log': open(logs_dir / f"log_subs_{log_name}.txt", 'w'),
        'failed_log': open(logs_dir / f"log_subsfailed_{log_name}.txt", 'w'),
        f'{TARGET_LANGUAGE}_down_log': open(logs_dir / f"log_subs{TARGET_LANGUAGE.upper()}download_{log_name}.txt", 'w'),
        'en_down_log': open(logs_dir / f"log_subsENGdownload_{log_name}.txt", 'w'),
        'en_only_log': open(logs_dir / f"log_subsENGonly_{log_name}.txt", 'w')
    }
    log_files['log'].write("LOG START:\n")
    log_files['log'].write(f"Ignoring folders: {', '.join(IGNORE_FOLDERS)}\n")
    log_files['log'].write(f"Ignoring keywords: {', '.join(IGNORE_KEYWORDS)}\n")
    log_files['log'].write(f"Target language: {TARGET_LANGUAGE.upper()}\n")

    folder_paths = [path for path in input_paths if os.path.isdir(path)]
    file_paths = [path for path in input_paths if os.path.isfile(path)]

    for input_path in file_paths:
        if input_path.endswith(tuple(VIDEO_EXTENSIONS)):
            if is_ignored_file(os.path.basename(input_path)):
                print(f"Ignoring sample file: {input_path}")
                log_files['log'].write(f"Ignoring sample file: {input_path}\n")
                continue
            process_video_file(input_path, log_files, TARGET_LANGUAGE)
        elif input_path.endswith(".srt"):
            print(f"Translating subtitle: {input_path}")
            translate_srt(input_path, "en", TARGET_LANGUAGE)
        else:
            print(f"Unsupported file type: {input_path}")
            log_files['log'].write(f"Unsupported file type: {input_path}\n")

    if skip_scan:
        print("Skipping initial scan. Starting in daemon mode only...")
        if folder_paths:
            monitor_folders(folder_paths, log_files, TARGET_LANGUAGE)
    else:
        for folder_path in folder_paths:
            process_folder(folder_path, log_files, TARGET_LANGUAGE)
        if folder_paths:
            monitor_folders(folder_paths, log_files, TARGET_LANGUAGE)

    for log in log_files.values():
        log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video files and download subtitles.")
    parser.add_argument('input_paths', nargs='+', help='Paths to video files, subtitle files, or folders containing video files.')
    parser.add_argument('log_name', help='Name to use for log files.')
    parser.add_argument('--daemon', '-d', action='store_true', help='Skip initial scan and go directly to daemon mode (monitor only).')
    args = parser.parse_args()
    main(args.input_paths, args.log_name, skip_scan=args.daemon)