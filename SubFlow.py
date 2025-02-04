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
from queue import Queue
from threading import Thread
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time

# ============================
# Configurable Section
# ============================
IGNORE_FOLDERS = ["Los Simpsons", "The Expanse Complete Series", "The Leftovers"]
OPENSUBTITLES_USER = "user"  # Replace with your OpenSubtitles username
OPENSUBTITLES_PASSWORD = "password"  # Replace with your OpenSubtitles password
TARGET_LANGUAGE = "es"  # Default target language (e.g., "es" for Spanish, "fr" for French)
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv'}

def install_dependencies():
    """Install required packages with user feedback"""
    required_packages = {
        'subliminal': 'subliminal',
        'transformers': 'transformers',
        'torch': 'torch',
        'srt': 'srt',
        'sentencepiece': 'sentencepiece',
        'watchdog': 'watchdog'
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
        # Add other model pairs as needed
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
        "sublime_en": (
            "Downloaded with github.com/fafaCabrera/SubFlow\n"
            "Sublime mode"
        ),
        "sublime_es": (
            "Downloaded with github.com/fafaCabrera/SubFlow\n"
            "Sublime mode"
        ),
        "opensubtitles_es": (
            "Downloaded with github.com/fafaCabrera/SubFlow\n"
            "Subliminal mode (opensubtitles)"
        ),
        "translated_es": (
            "Downloaded with github.com/fafaCabrera/SubFlow\n"
            "Subliminal mode and translated"
        ),
    }

    with open(subtitle_path, 'r', encoding='utf-8') as f:
        subs = list(srt.parse(f.read()))

    # Create a new subtitle entry for the credits
    credits_entry = srt.Subtitle(
        index=0,
        start=timedelta(seconds=3),
        end=timedelta(seconds=13),
        content=credit_lines[mode]
    )

    # Insert the credits at the beginning
    subs.insert(0, credits_entry)

    # Write the updated subtitles back to the file
    with open(subtitle_path, 'w', encoding='utf-8') as f:
        f.write(srt.compose(subs))


def translate_srt(input_file, src_lang, tgt_lang):
    """Main translation function with enhanced GPU monitoring"""
    start_time = datetime.now()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nStarting translation: {src_lang} → {tgt_lang}")
    print(f"Using: {'GPU acceleration' if device.type == 'cuda' else 'CPU'} mode")
    print(f"Input file: {os.path.abspath(input_file)}")

    # Model setup
    model_name = get_model_name(src_lang, tgt_lang)
    print(f"Loading model '{model_name}'...")
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(device)
        print(f"Model loaded on {device.type.upper()}")
        print(f"Model device: {next(model.parameters()).device}")
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        sys.exit(1)

    # Read and parse SRT
    print("\nProcessing subtitle file...")
    with open(input_file, 'r', encoding='utf-8') as f:
        subs = list(srt.parse(f.read()))
    total_subs = len(subs)
    print(f"Found {total_subs} subtitles to translate")

    # Translation process
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

        # Print GPU memory usage every 10%
        if idx % max(1, total_subs // 10) == 0 and device.type == 'cuda':
            mem_usage = torch.cuda.memory_allocated() / 1024**3
            print(f"  GPU VRAM: {mem_usage:.2f} GB used | {idx}/{total_subs} ({idx / total_subs:.0%})")
        else:
            progress = idx / total_subs
            print(f"  █ {idx}/{total_subs} ({progress:.0%})", end='\r')

    # Cleanup
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Generate output filename
    base_path = os.path.splitext(input_file)[0].rsplit('.', 1)[0]
    output_file = f"{base_path}.{tgt_lang}.srt"

    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(srt.compose(translated_subs))

    # Add credits
    add_credits_to_subtitle(output_file, "translated_es")

    # Final report
    duration = datetime.now() - start_time
    print(f"\n\nTranslation completed in {duration.total_seconds():.1f} seconds")
    print(f"Output file created: {os.path.abspath(output_file)}")


def download_subtitles(video_path, lang, log_files):
    """Download subtitles using subliminal."""
    print(f"Downloading {lang.upper()} subtitles for: {video_path}")
    subprocess.run(["subliminal", "download", "-l", lang, video_path])
    subprocess.run([
        "subliminal", "--opensubtitles", OPENSUBTITLES_USER, OPENSUBTITLES_PASSWORD,
        "download", "-p", "opensubtitles", "-l", lang, video_path
    ])

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    subtitle_path = os.path.join(os.path.dirname(video_path), f"{base_name}.{lang}.srt")

    if os.path.exists(subtitle_path):
        print(f"{lang.upper()} subtitle downloaded for: {video_path}")
        log_files[f'{lang}_down_log'].write(f"{lang.upper()} subtitle downloaded for: {video_path}\n")

        # Add credits based on the mode
        if lang == "en":
            add_credits_to_subtitle(subtitle_path, "sublime_en")
        elif lang == TARGET_LANGUAGE:
            add_credits_to_subtitle(subtitle_path, "sublime_es")

        return True
    return False


def process_video_file(video_path, log_files, tgt_lang):
    """Process a single video file"""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.dirname(video_path)

    # Check if target language subtitle exists
    tgt_srt = os.path.join(video_dir, f"{base_name}.{tgt_lang}.srt")
    if os.path.exists(tgt_srt):
        print(f"{tgt_lang.upper()} subtitle found for: {video_path}")
        log_files['log'].write(f"{tgt_lang.upper()} subtitle found for: {video_path}\n")
        return

    # Try downloading target language subtitle
    if not download_subtitles(video_path, tgt_lang, log_files):
        # Check if English subtitle exists
        en_srt = os.path.join(video_dir, f"{base_name}.en.srt")
        if not os.path.exists(en_srt):
            download_subtitles(video_path, "en", log_files)

        if os.path.exists(en_srt):
            print(f"EN Found, Translating....")
            log_files['en_only_log'].write(f"Needs Translate: {video_path}\n")
            translate_srt(en_srt, "en", tgt_lang)
        else:
            print(f"Failed to get subtitles for: {video_path}")
            log_files['failed_log'].write(f"Failed to get subtitles for: {video_path}\n")
            log_files['log'].write(f"Failed to get subtitles for: {video_path}\n")


def process_folder(folder_path, log_files, tgt_lang):
    """Process all video files in a folder and its subfolders."""
    for root, dirs, files in os.walk(folder_path):
        if should_skip_folder(root):
            log_files['log'].write(f"Hardcoded Skipping \"{root}\"\n")
            continue

        for file in files:
            if file.endswith(tuple(VIDEO_EXTENSIONS)):
                video_path = os.path.join(root, file)
                process_video_file(video_path, log_files, tgt_lang)


def monitor_folders(folder_paths, log_files, tgt_lang):
    """Monitor multiple folders for new video files."""
    class VideoFileHandler(FileSystemEventHandler):
        def on_created(self, event):
            if not event.is_directory:
                file_path = event.src_path
                _, ext = os.path.splitext(file_path)
                if ext.lower() in VIDEO_EXTENSIONS:
                    print(f"New video file detected: {file_path}")
                    process_video_file(file_path, log_files, tgt_lang)

        def on_moved(self, event):
            if not event.is_directory:
                dest_path = event.dest_path
                _, ext = os.path.splitext(dest_path)
                if ext.lower() in VIDEO_EXTENSIONS:
                    print(f"Renamed video file detected: {dest_path}")
                    process_video_file(dest_path, log_files, tgt_lang)

    observers = []
    for folder_path in folder_paths:
        observer = Observer()
        event_handler = VideoFileHandler()
        observer.schedule(event_handler, path=folder_path, recursive=True)
        observer.start()
        observers.append(observer)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        for observer in observers:
            observer.stop()
        for observer in observers:
            observer.join()


def main(input_paths, log_name):
    """Main function to process video files or folders"""
    # Install dependencies only if processing folders
    if any(os.path.isdir(path) for path in input_paths):
        install_dependencies()

    # Get the script directory
    script_dir = Path(__file__).parent.absolute()
    logs_dir = script_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Initialize log file paths
    log_files = {
        'log': open(logs_dir / f"log_subs_{log_name}.txt", 'w'),
        'failed_log': open(logs_dir / f"log_subsfailed_{log_name}.txt", 'w'),
        f'{TARGET_LANGUAGE}_down_log': open(logs_dir / f"log_subs{TARGET_LANGUAGE.upper()}download_{log_name}.txt", 'w'),
        'en_down_log': open(logs_dir / f"log_subsENGdownload_{log_name}.txt", 'w'),
        'en_only_log': open(logs_dir / f"log_subsENGonly_{log_name}.txt", 'w')
    }

    log_files['log'].write("LOG START:\n")
    log_files['log'].write(f"Ignoring folders: {', '.join(IGNORE_FOLDERS)}\n")
    log_files['log'].write(f"Target language: {TARGET_LANGUAGE.upper()}\n")

    # Process input
    for input_path in input_paths:
        if os.path.isfile(input_path):
            if input_path.endswith(tuple(VIDEO_EXTENSIONS)):
                process_video_file(input_path, log_files, TARGET_LANGUAGE)
            elif input_path.endswith(".srt"):
                print(f"Translating subtitle: {input_path}")
                translate_srt(input_path, "en", TARGET_LANGUAGE)
            else:
                print(f"Unsupported file type: {input_path}")
                log_files['log'].write(f"Unsupported file type: {input_path}\n")
        elif os.path.isdir(input_path):
            process_folder(input_path, log_files, TARGET_LANGUAGE)
        else:
            print(f"The specified path does not exist: {input_path}")
            sys.exit(1)

    # If folders were provided, start monitoring them
    if any(os.path.isdir(path) for path in input_paths):
        folder_paths = [path for path in input_paths if os.path.isdir(path)]
        monitor_folders(folder_paths, log_files, TARGET_LANGUAGE)

    # Close log files
    for log in log_files.values():
        log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video files and download subtitles.")
    parser.add_argument('input_paths', nargs='+', help='Paths to video files, subtitle files, or folders containing video files.')
    parser.add_argument('log_name', help='Name to use for log files.')
    args = parser.parse_args()

    main(args.input_paths, args.log_name)