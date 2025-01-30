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
