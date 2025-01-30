# **SubFlow**

[![Python Version](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/) [![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)

**SubFlow** is a powerful Python script designed to automate the process of downloading and translating subtitles for video files. It integrates seamlessly with OpenSubtitles and uses AI-powered translation models to translate subtitles into your desired language.

---

## **Features**

- **Automated Subtitle Downloading**:
  - Automatically downloads subtitles in your preferred language (e.g., Spanish) using OpenSubtitles.
  - Falls back to English subtitles if the preferred language is unavailable.

- **AI-Powered Translation**:
  - Translates English subtitles into your target language (e.g., Spanish) using state-of-the-art machine translation models from Hugging Face (`Helsinki-NLP`).

- **Customizable**:
  - Configure OpenSubtitles credentials directly in the script.
  - Set your target language (default is Spanish, but you can change it to French, German, etc.).

- **Logging**:
  - Detailed logs are generated for successful downloads, failed attempts, and translations.

- **Dependency Management**:
  - Automatically installs all required dependencies if they’re not already installed.

---

## **Installation**

### **Step 1: Install Python**
Make sure you have Python 3.7 or higher installed. You can download it from [python.org](https://www.python.org/downloads/).

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

## **Usage**
### **Command-Line Arguments**
**The script accepts two arguments:**
1. Input Path : Path to a video file, subtitle file, or folder containing video files.
2. Log Name : Name to use for log files.

**Examples:**
1. Process a Single Video File :
```bash
python subs_full.py "C:\path\to\video.mp4" LOG_NAME
```
2. Process a Folder of Videos :
```bash
python subs_full.py "C:\path\to\folder" LOG_NAME
```
3. Process a Single Subtitle File :
```bash
python subs_full.py "C:\path\to\subtitle.en.srt" LOG_NAME
```
---
## **Configuration**
You can customize the following settings directly in the script:

1. **OpenSubtitles Credentials :**
    ```bash
    OPENSUBTITLES_USER = "your_username"
    OPENSUBTITLES_PASSWORD = "your_password"
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
- **subliminal** : For downloading subtitles.
- **transformers** : For AI-powered translation.
- **torch** : PyTorch backend for GPU acceleration.
- **srt** : For parsing and writing .srt files.
- **sentencepiece** : Required by the translation model.
---
## **Logs**
All logs are stored in the logs directory relative to the script's location. The following log files are generated:

- **log_subs_{LOG_NAME}.txt:** General log.
- **log_subsfailed_{LOG_NAME}.txt:** Failed subtitle downloads.
- **log_subsESPdownload_{LOG_NAME}.txt:** Successfully downloaded Spanish subtitles.
- **log_subsENGdownload_{LOG_NAME}.txt:** Successfully downloaded English subtitles.
- **log_subsENGonly_{LOG_NAME}.txt:** Subtitles that required translation.
---
## **Contributing**
###### Contributions are welcome! If you find a bug or want to add a feature, feel free to open an issue or submit a pull request.
---
## **License**
###### This project is licensed under the MIT License. See the LICENSE file for details.
---
## **Acknowledgments**
OpenSubtitles : For providing subtitle downloads.
Hugging Face : For their amazing translation models.
PyTorch : For enabling GPU-accelerated translation.
