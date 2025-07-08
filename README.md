# SubFlow.py
[![Python Version](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/) [![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)

## Overview

`SubFlow.py` is a Python script designed to automate the process of downloading and translating subtitles for video files. It supports both individual file processing and bulk folder scanning, with optional daemon mode for real-time monitoring of folders. The script integrates subtitle downloading via `subliminal`, translation using Hugging Face's `transformers`, and logging for tracking operations.

---

## Features

- **Subtitle Downloading**:
  - Automatically downloads subtitles in the target language (default: Spanish) using `subliminal`.
  - Falls back to English subtitles if the target language subtitles are unavailable.
  
- **Subtitle Translation**:
  - Translates English subtitles to the target language using machine learning models from Hugging Face (`MarianMT`).

- **AI-Powered Translation**:
  - Translates English subtitles into your target language (e.g., Spanish) using state-of-the-art machine translation models from Hugging Face (`Helsinki-NLP`).

- **AI-Powered Transcribing**:
  - Transcribes subtitles from video user open-ai Whisper model.

- **Customizable**:
  - Configure OpenSubtitles credentials directly in the script.
  - Set your target language (default is Spanish, but you can change it to French, German, etc.).

- **Logging**:
  - Generates detailed logs for all operations, including successes, failures, and translations.
---

## Installation

### **Step 1: Install Python**
Make sure you have [![Python Version](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/) installed. You can download it from [python.org](https://www.python.org/downloads/).

### **Step 2: Clone the Repository**
Clone this repository to your local machine:
```bash
git clone https://github.com/yourusername/SubFlow.git
cd SubFlow
```

### **Step 3: Install Dependencies**
You can either let the script handle dependency installation automatically, or you can manually install them using the provided requirements.txt:

```bash
pip install -r requirements.txt
```
---

## Usage

### Command-Line Arguments

The script accepts two arguments:

1. **Input Path**:
   - A single video file, subtitle file, or folder containing video files.
   
2. **Log Name**:
   - A name to identify the log files generated during execution.

#### Example Commands

- Process a single video file:
  ```bash
  python SubFlow.py /path/to/video.mkv my_log
  ```

- Monitor a folder of video files:
  ```bash
  python SubFlow.py /path/to/folder my_log
  ```

- Monitor more than on folder in daemon mode:
  ```bash
  python SubFlow.py /path/to/folder1 /path/to/folder2 my_log
  ```

---

## Configuration

### Configurable Variables

- **IGNORE_FOLDERS**:
  - List of folder names to skip during folder scanning.
  - Example:
    ```python
    IGNORE_FOLDERS = ["Los Simpsons", "The Expanse Complete Series"]
    ```
2. **Target Language :**
Change the default target language (Spanish) to another language (e.g., French):
    ```bash
    TARGET_LANGUAGE = "fr"  # Default is "es" for Spanish
    ```
3. **Ignored Folders :**
Add folders to skip during processing:
    ```bash
    IGNORE_FOLDERS = ["Folder1", "Folder2"]
    ```
---
## **Dependencies**
- **whisper** : For transcribing audio.
- **subliminal** : For downloading subtitles.
- **transformers** : For AI-powered translation.
- **torch** : PyTorch backend for GPU acceleration.
- **srt** : For parsing and writing .srt files.
- **sentencepiece** : Required by the translation model.
---
## **Logs**
All logs are stored in the logs directory relative to the script's location. The following log files are generated:

- **TARGET_LANGUAGE**:
  - Target language for subtitles (default: `"es"` for Spanish).
  - Example:
    ```python
    TARGET_LANGUAGE = "fr"  # For French
    ```

- **OPENSUBTITLES_USER** and **OPENSUBTITLES_PASSWORD**:
  - Your OpenSubtitles credentials for subtitle downloading.

---

## Functionality Details

### 1. Individual File Processing

- If a video file is provided:
  - Attempts to download subtitles in the target language.
  - If unavailable, falls back to English subtitles and translates them.

- If a subtitle file is provided:
  - Translates the subtitle file to the target language.

### 2. Folder Processing

- Scans all subfolders for video files.
- Processes each video file as described above.

### 3. Daemon Mode

- Monitors specified folders for new or modified video files.
- Processes detected files in real-time without reinstalling dependencies.

---

## Logs

All logs are stored in the `logs` directory within the script's folder. Log files include:

- `log_subs_<log_name>.txt`: General log.
- `log_subsfailed_<log_name>.txt`: Failed operations.
- `log_subs<lang>download_<log_name>.txt`: Successful subtitle downloads.
- `log_subsENGonly_<log_name>.txt`: Files requiring translation.

---

## Requirements

The following dependencies are required to run the script. They can be installed using the `requirements.txt` file:

### `requirements.txt`
```text
subliminal
transformers
torch
srt
sentencepiece
watchdog
```

Install the dependencies with:
```bash
pip install -r requirements.txt
```

---

## Troubleshooting

### Common Issues

1. **Subliminal Provider Errors**:
   - Ensure `subliminal` is updated (`pip install --upgrade subliminal`).
   - Verify OpenSubtitles credentials.

2. **Translation Failures**:
   - Check GPU availability and memory usage.
   - Ensure `transformers` and `torch` are properly installed.

3. **Daemon Mode Not Working**:
   - Ensure the `watchdog` library is installed.
   - Verify folder permissions.

---

## License

This script is open-source and available under the MIT License. Feel free to modify and distribute it as needed.

For support or contributions, visit the [GitHub repository](https://github.com/fafaCabrera/SubFlow).

---
