#  AI Text Summarizer using Fine-Tuned T5 Transformer

<p align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-Web%20Framework-009688?style=for-the-badge&logo=fastapi)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</p>

An AI-powered **Text Summarization Web Application** built using a **Fine-Tuned T5-Small Transformer**, **FastAPI**, and **Hugging Face Transformers**. The application generates concise and meaningful summaries from lengthy dialogues or text through an intuitive and user-friendly web interface.

---

#  Project Overview

Text Summarization is one of the most important tasks in **Natural Language Processing (NLP)**. It aims to automatically generate a shorter version of a document while preserving its essential information and context.

In this project, the **T5-Small Transformer** model was fine-tuned on the **SAMSum Dialogue Summarization Dataset** to generate abstractive summaries of conversational text. The trained model is deployed using **FastAPI**, providing an interactive web application where users can instantly summarize long dialogues.

---

#  Features

-  Fine-Tuned **T5-Small Transformer**
-  Dialogue & Text Summarization
-  FastAPI Backend
-  Interactive Web Interface
-  Automatic Text Preprocessing
-  REST API Support
-  Cross-Platform Compatibility (CPU / CUDA / Apple MPS)
-  Fast and Lightweight Deployment

---

#  Model Architecture

```
                    Input Dialogue/Text
                            │
                            ▼
                    Text Preprocessing
                            │
                            ▼
                     T5 Tokenizer
                            │
                            ▼
               Fine-Tuned T5-Small Encoder
                            │
                  Multi-Head Self Attention
                            │
                  Transformer Decoder
                            │
                    Beam Search Decoding
                            │
                            ▼
                  Generated Text Summary
                            │
                            ▼
                      FastAPI Backend
                            │
                            ▼
               HTML • CSS • JavaScript UI
```

> **Architecture Diagram:** `assets/architecture.png`

---

#  Project Structure

```text
Text-Summarizer-T5/
│
├── app.py
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── script.js
│
├── saved_summary_model/
│
├── notebook/
|   └── Text_Summarizer.ipynb
```

---

#  Dataset

The model was fine-tuned using the **SAMSum Dialogue Summarization Dataset**, a benchmark dataset designed for abstractive dialogue summarization.

### Dataset Link

https://tinyurl.com/5r6ha25c

### Dataset Highlights

- Human-written summaries
- Multi-speaker conversations
- High-quality annotations
- Suitable for abstractive summarization
- Widely used NLP benchmark

---

#  Data Preprocessing

Before training, the following preprocessing steps were applied:

- Removal of HTML tags
- Removal of unnecessary whitespaces
- Removal of line breaks
- Conversion to lowercase
- Tokenization using the T5 Tokenizer
- Input padding and truncation

---

#  Model Training

| Parameter | Value |
|-----------|-------|
| Model | T5-Small |
| Framework | Hugging Face Transformers |
| Optimizer | AdamW |
| Epochs | 6 |
| Batch Size | 8 |
| Maximum Input Length | 512 Tokens |
| Maximum Output Length | 150 Tokens |
| Beam Search | 4 |

---

#  Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| PyTorch | Deep Learning Framework |
| Hugging Face Transformers | NLP Model |
| FastAPI | Backend Framework |
| HTML5 | Frontend Structure |
| CSS3 | Styling |
| JavaScript | Client-side Interaction |
| Jinja2 | HTML Templating |

---

#  Installation

### Install the Required Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python -m uvicorn app:app --reload
```

After the server starts, open your browser and visit:

```
http://127.0.0.1:8000
```

### API Documentation

FastAPI automatically provides interactive API documentation at:

```
http://127.0.0.1:8000/docs
```

---

#  Future Improvements

- Fine-tune larger T5 variants (T5-Base / T5-Large)
- Add ROUGE evaluation metrics
- Support PDF and DOCX summarization
- Multi-language summarization
- Upload text files for summarization
- Docker containerization
- Cloud deployment (AWS, Azure, Render)
- User authentication
- Dark mode interface
- Real-time summarization API

---

#  Learning Outcomes

This project provided practical experience in:

- Natural Language Processing (NLP)
- Transformer-based Language Models
- Fine-Tuning Pre-trained Models
- Hugging Face Transformers
- FastAPI Development
- REST API Design
- Web Application Deployment
- Frontend Integration
- Model Inference using PyTorch
- End-to-End AI Application Development

---

#  Author

**Soumya Kanti Upadhyay**

**B.Tech in Computer Science & Engineering**  
Jalpaiguri Government Engineering College

---


---
