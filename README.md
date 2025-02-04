# SubFlow.py - README

## Overview

`SubFlow.py` is a Python script designed to automate the process of downloading and translating subtitles for video files. It supports both individual file processing and bulk folder scanning, with optional daemon mode for real-time monitoring of folders. The script integrates subtitle downloading via `subliminal`, translation using Hugging Face's `transformers`, and logging for tracking operations.

---

## Features

- **Subtitle Downloading**:
  - Automatically downloads subtitles in the target language (default: Spanish) using `subliminal`.
  - Falls back to English subtitles if the target language subtitles are unavailable.
  
- **Subtitle Translation**:
  - Translates English subtitles to the target language using machine learning models from Hugging Face (`MarianMT`).

- **Daemon Mode**:
  - Monitors specified folders for new or modified video files and processes them in real-time.

- **Logging**:
  - Generates detailed logs for all operations, including successes, failures, and translations.
---

## Installation

### Prerequisites

1. **Python 3.7+**: Ensure Python is installed on your system.
2. **Dependencies**:
   - Install the required dependencies using the provided `requirements.txt` file:
     ```bash
     pip install -r requirements.txt
     ```

3. **OpenSubtitles Credentials**:
   - Update the `OPENSUBTITLES_USER` and `OPENSUBTITLES_PASSWORD` variables in the script with your OpenSubtitles account details.

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

- Process a folder of video files:
  ```bash
  python SubFlow.py /path/to/folder my_log
  ```

- Monitor a folder in daemon mode:
  ```bash
  python SubFlow.py /path/to/folder my_log
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
