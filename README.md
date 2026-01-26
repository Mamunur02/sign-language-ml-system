# Sign Language ML System

End-to-end machine learning system for British Sign Language (BSL) recognition, built with a focus on real-world ML engineering rather than isolated model training.

## Project Intent
The goal of this project is to build a production-style ML system for British Sign Language recognition. 
The project starts with a simple image-based baseline to validate the ML pipeline and progressively evolves toward hand landmark and motion-based models for word-level recognition.

## Data Format
Raw data is expected in the following structure:

data/raw/
  <label_name>/
    image_1.jpg
    image_2.png

## Setup
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

## Roadmap
- Image-based baseline model (pipeline validation)
- Hand landmark extraction using MediaPipe
- Temporal modelling for word-level BSL recognition
- Real-time inference and API serving
- Monitoring and retraining lifecycle