# 🏛️ HeritageAI (செம்மொழி சுடர்)

> **AI-Powered Ancient Tamil Palm Leaf Manuscript (ஓலைச்சுவடி) & Epigraphy Restoration, OCR, Multilingual Translation, and Audio Narration System.**

Built for the **Thoothukudi Hackathon (TTH)**.

---

## 🌟 Key Features

1. **🔬 Computer Vision Image Enhancement**:
   - Converts aged, degraded, low-contrast manuscript scans into high-contrast images.
   - **CLAHE (Contrast Limited Adaptive Histogram Equalization)** with interactive user slider (clip limit 1.0 - 8.0).
   - **Gaussian Adaptive Thresholding** to highlight engraved Tamil characters.

2. **📜 Multimodal AI OCR & Ancient Script Transcription**:
   - Powered by Google **Gemini 2.5 Flash**.
   - Deciphers archaic Tamil scripts (வட்டெழுத்து / பழந்தமிழ் / கல்வெட்டு).

3. **🌐 Scholarly Multilingual Translation**:
   - Translates ancient meanings into contemporary understanding with context, poetic nuances, and cultural background.
   - Supported languages:
     - English
     - Tamil (எளிய தமிழ் உரை)
     - Hindi (हिन्दी)
     - French (Français)
     - German (Deutsch)
     - Spanish (Español)
     - Japanese (日本語)
     - Arabic (عربي)

4. **🔊 Audio Narration**:
   - Uses **Google Text-to-Speech (gTTS)** to synthesize spoken audio of the translated meaning in the target language.
   - Integrated audio player in the Streamlit interface.

5. **✨ Premium Dark Gold UI**:
   - Custom styling with Google Fonts (`Cinzel`, `Inter`, `Mukta Malar`).
   - Side-by-side vision comparison (Original Artifact vs. AI Restored View).

---

## 📁 Project Directory Structure

```
C:\Heritage\
├── app.py                      # Core Streamlit application (recovered)
├── requirements.txt            # Python dependencies
├── run.bat                     # 1-Click double-click launcher
├── run.ps1                     # PowerShell runner
├── README.md                   # Complete documentation
├── sample_manuscripts/         # Sample palm leaf & inscription images for testing
│   ├── image2.png              # Wide palm leaf manuscript scan
│   ├── image4.png              # Cropped manuscript leaf
│   ├── image7.jpeg             # Ancient palm leaf fragment
│   └── ...
├── reports/                    # Historical analysis outputs & transcripts
│   ├── HeritageAI_Report.txt
│   └── HeritageAI_Report_Thirukkural.txt
└── presentation/               # Original Hackathon PPTX presentations
    ├── Thoothukudi_Hackathon_2025_final.pptx
    └── TTH03_PitchDeck.pptx
```

---

## 🚀 How to Run

### Option 1: Double-Click
Simply double-click `run.bat` in this folder.

### Option 2: Terminal
```bash
cd C:\Heritage
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🔑 Configuration
Enter your **Gemini API Key** in the sidebar input field before clicking **⚡ Extract & Translate**.

---

## 🎥 Original Demo Recordings
- Prototype Video: `C:\Users\Ramyaiyyappan\OneDrive\Desktop\prototype .mp4`
- Prototype Video with English Audio: `C:\Users\Ramyaiyyappan\OneDrive\Desktop\PROTOYPE ENG AUDIO.mp4`
