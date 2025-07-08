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

# ============================
# Configurable Section
# ============================
IGNORE_FOLDERS = ["Los Simpsons", "The Expanse Complete Series", "The Leftovers"]
OPENSUBTITLES_USER = "username"  # Replace with your OpenSubtitles username
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
        'sentencepiece': 'sentencepiece'
    }
    print("Checking dependencies...")
    installed = 0

    # Check and install Python packages
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
    output_file = f"{base_path}.{tgt_lang}{suffix}.srt"
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(srt.compose(translated_subs))

    # Final report
    duration = datetime.now() - start_time
    print(f"\nTranslation completed in {duration.total_seconds():.1f} seconds")
    print(f"Output file created: {os.path.abspath(output_file)}")

def translate_subtitle(subtitle_path, src_lang, tgt_lang, log_files):
    """Translate an SRT file using integrated translation logic"""
    print(f"Translating subtitle: {subtitle_path}")
    try:
        translate_srt(subtitle_path, src_lang, tgt_lang)
        log_files['en_only_log'].write(f"Translated: {subtitle_path}\n")
    except Exception as e:
        print(f"Error translating subtitle: {subtitle_path} - {str(e)}")
        log_files['failed_log'].write(f"Failed to translate: {subtitle_path}\n")

def process_video_file(video_path, log_files, script_dir, tgt_lang):
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
    print(f"{tgt_lang.upper()} subtitle NOT found for: {video_path}, executing {tgt_lang.upper()} subliminal...")
    subprocess.run(["subliminal", "download", "-l", tgt_lang, video_path])
    subprocess.run([
        "subliminal", "--opensubtitles", OPENSUBTITLES_USER, OPENSUBTITLES_PASSWORD,
        "download", "-p", "opensubtitles", "-l", tgt_lang, video_path
    ])

    if os.path.exists(tgt_srt):
        print(f"{tgt_lang.upper()} subtitle downloaded for: {video_path}")
        log_files[f'{tgt_lang}_down_log'].write(f"{tgt_lang.upper()} subtitle downloaded for: {video_path}\n")
        return

    # Check if English subtitle exists
    en_srt = os.path.join(video_dir, f"{base_name}.en.srt")
    if os.path.exists(en_srt):
        print(f"EN Found, Translating....")
        log_files['en_only_log'].write(f"Needs Translate: {video_path}\n")
        translate_subtitle(en_srt, "en", tgt_lang, log_files)
        return

    # Try downloading English subtitle
    print(f"English subtitle NOT found for: {video_path}, executing ENG subliminal...")
    subprocess.run(["subliminal", "download", "-l", "en", video_path])
    subprocess.run([
        "subliminal", "--opensubtitles", OPENSUBTITLES_USER, OPENSUBTITLES_PASSWORD,
        "download", "-p", "opensubtitles", "-l", "en", video_path
    ])

    if os.path.exists(en_srt):
        print(f"EN Found, Translating....")
        log_files['en_only_log'].write(f"Needs Translate: {video_path}\n")
        translate_subtitle(en_srt, "en", tgt_lang, log_files)
    else:
        print(f"Failed to get subtitles for: {video_path}")
        log_files['failed_log'].write(f"Failed to get subtitles for: {video_path}\n")
        log_files['log'].write(f"Failed to get subtitles for: {video_path}\n")

def process_subtitle_file(subtitle_path, log_files, tgt_lang):
    """Process a single subtitle file"""
    if subtitle_path.endswith(".en.srt"):
        print(f"English subtitle found: {subtitle_path}")
        log_files['en_only_log'].write(f"Needs Translate: {subtitle_path}\n")
        translate_subtitle(subtitle_path, "en", tgt_lang, log_files)
    else:
        print(f"Unsupported subtitle format: {subtitle_path}")
        log_files['log'].write(f"Unsupported subtitle format: {subtitle_path}\n")

def main(input_path, log_name):
    """Main function to process video files or folders"""
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
    if os.path.isfile(input_path):
        if input_path.endswith(('.mkv', '.mp4', '.avi')):
            process_video_file(input_path, log_files, script_dir, TARGET_LANGUAGE)
        elif input_path.endswith(".srt"):
            process_subtitle_file(input_path, log_files, TARGET_LANGUAGE)
        else:
            print(f"Unsupported file type: {input_path}")
            log_files['log'].write(f"Unsupported file type: {input_path}\n")
    elif os.path.isdir(input_path):
        for root, dirs, files in os.walk(input_path):
            if should_skip_folder(root):
                log_files['log'].write(f"Hardcoded Skipping \"{root}\"\n")
                continue
            for file in files:
                if file.endswith(('.mkv', '.mp4', '.avi')):
                    video_path = os.path.join(root, file)
                    process_video_file(video_path, log_files, script_dir, TARGET_LANGUAGE)
    else:
        print(f"The specified path does not exist: {input_path}")
        sys.exit(1)

    # Close log files
    for log in log_files.values():
        log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video files and download subtitles.")
    parser.add_argument('input_paths', nargs='+', help='Paths to video files, subtitle files, or folders containing video files.')
    parser.add_argument('log_name', help='Name to use for log files.')
    args = parser.parse_args()

    main(args.input_path, args.log_name)