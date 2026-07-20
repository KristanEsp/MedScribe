<h1 align="center">MedScribe</h1>

<img width="940" height="564" alt="image" src="https://github.com/user-attachments/assets/23a83fd9-96ae-47dd-9899-d49afa270fd2" />

# Introduction
- Medscribe is an application that transcribes a doctor-patient consultation and produces a SOAP medical summary.
- The main goal is to provide an aiding tool for medical professionals to reduce time spent on performing Electronic Health Record (EHR) tasks

# How to run
<h3> HuggingFace Space </h3>
Demo link: https://huggingface.co/spaces/KroyEsp/MedScribe

- Takes approximately 2-3 minutes to restart the application

# Method
## Speech Processing Pipeline
<img width="563" height="563" alt="image" src="https://github.com/user-attachments/assets/101feeba-fc75-4959-a41a-ef2ecd6620b5" />

<h2> Stage 1: Audio Pre-processing & Transcription </h2>

- Convertion to WAV format, Normalization and audio boosting to improve transcription and diarization accuracy
- Faster Whisper (medium-en) model for lightweight, fast and accurate transcription

<h2> Stage 2: Speaker Diarization </h2>

- Voice embedding using Resemblyzer to extract high-level voice features
- Agglomerative Clustering to segment the extracted embeddings into speaker-specific groups or cluster


<h2> Stage 3: Speaker Identification </h2>

- Classifies each speaker as either a doctor or the patient using a Bi-Directional LSTM model
- Model was trained on a text transcript dataset with a sample size of 544 labelled transcripts

## Abstractive Summarization (SOAP Summary)

- Achieved by using Meta's Llama-3-8B Instruct LLM model which is optimized for dialogue-based conversations and can perform summarization tasks effectively
- However, a domain-specific model + RAG system are recommended to reduce textual hallucinations (future work)

# Limitations

1.) LSTM model trained on a small, single dataset due to difficulties obtaining available transcription data. Hence, model is highly likely to poorly generalize in real-world scenarios

2.) Reduced speaker diarization accuracy with higher number of speakers. Application works best with one-on-one consultations

3.) Overlapping speeches and excessive background noise can reduce transcription accuracy

*Created June 2026*
