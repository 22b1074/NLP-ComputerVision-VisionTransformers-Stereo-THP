## Q1_NLP

# 🔊 Multilingual Text-to-Speech (NLP)

- Python-based **Text-to-Speech system** for reading project updates
- Runs via **command-line (CLI)** using user inputs
- Extracts text from **PDF & DOCX** files
- **Automatic language detection**
- **Multilingual translation** support
- Speech synthesis using **Google Text-to-Speech (gTTS)**
- End-to-end **NLP → Translation → Speech** pipeline
- Modular and easy to extend (offline TTS, GUI, APIs)

**Tech Stack:** Python, gTTS, googletrans, langdetect, PyPDF2, python-docx  
**Author:** Preethi Chappidi | IIT Bombay

--

## Q1_Object_Detection

# 🧪 Laboratory Object Detection (YOLOv8)

- Built an **object detection model** using **YOLOv8 (PyTorch)** for laboratory objects
- Classes trained: **Laptop, Book**
- Dataset sourced from **Open Images V7** using **FiftyOne**
- Used **200 images** → split into **Train (160)** and **Validation (40)**
- Exported and trained in **YOLO format**
- Fine-tuned **YOLOv8n** for **10 and 50 epochs**
- Evaluated using **Precision, Recall, F1-score, mAP**

## 📊 Performance Summary

| Class   | Precision | Recall | mAP@0.5 | Images | Instances |
|--------|-----------|--------|---------|---------|---------|
| Laptop | 0.664     | 0.71  | 0.52   | 2-   | 28   |
| Book   | 0.72     | 0.21  | 0.31   | 20   | 94   |
| **Overall** | **0.693** | **~0.464** | **0.516** | **40** | **122** |

- Strong detection performance for **Laptop**
- Lower recall for **Book** due to visual variability and limited data

**Tech Stack:** PyTorch, YOLOv8, FiftyOne, Open Images V7  

--
