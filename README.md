# Transformer-Based Brain-to-Text Decoding

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red?style=for-the-badge)
![Transformer](https://img.shields.io/badge/Architecture-Transformer-green?style=for-the-badge)
![Mamba](https://img.shields.io/badge/Decoder-Mamba--GRU-orange?style=for-the-badge)
![BCI](https://img.shields.io/badge/Domain-Brain--Computer--Interface-purple?style=for-the-badge)
![Research](https://img.shields.io/badge/Type-Research_Project-success?style=for-the-badge)

</p>

<p align="center">
  <b>Decoding Human Neural Activity into Natural Language using Transformer and Mamba-GRU Architectures</b>
</p>

---

# Abstract

Brain-Computer Interfaces (BCIs) are transforming the way humans interact with machines by enabling direct communication through neural signals. This project presents a **Transformer-Based Brain-to-Text Decoding Framework** capable of translating intracortical neural recordings into meaningful text.

The framework combines the powerful contextual learning capabilities of **Transformer Networks** with the efficiency of **Mamba State Space Models** and the temporal memory retention of **GRU Networks**. The resulting hybrid architecture is designed to improve neural sequence decoding accuracy while maintaining computational efficiency for real-time applications.

Built upon the **NEJM Brain-to-Text Dataset**, this work explores advanced sequence modeling techniques for neural speech restoration and assistive communication systems.

---

# Architecture Overview

<p align="center">
  <img src="b2txt_methods_overview.png" width="900">
</p>

<p align="center">
Transformer-Based Brain-to-Text Decoding Pipeline
</p>

---

# Key Contributions

✅ Transformer-based neural sequence encoding

✅ Hybrid Mamba-GRU decoding architecture

✅ End-to-end Brain-to-Text pipeline

✅ Neural feature extraction and preprocessing

✅ Word Error Rate (WER) evaluation framework

✅ Research-oriented modular implementation

✅ Scalable architecture for future real-time deployment

---

# System Architecture

```text
┌─────────────────────────┐
│ Intracortical Recordings│
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Neural Signal Processing│
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Feature Representation  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Transformer Encoder     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Hybrid Mamba-GRU Decoder│
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Text Generation Module  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ WER Evaluation          │
└─────────────────────────┘
```

---

# Repository Structure

```text
Transformer-Based-Brain-to-Text-Decoding
│
├── src/
│   ├── main.py
│   ├── dataset.py
│   ├── model.py
│   ├── model_transformer.py
│   ├── train.py
│   ├── train_transformer.py
│   ├── evaluate.py
│   ├── convert_t12.py
│   └── run_nejm_decoder.py
│
├── notebooks/
│   └── Untitled.ipynb
│
├── docs/
│   ├── README_128GB_STEPS.md
│   └── wer_results.json
│
├── analyses/
├── language_model/
├── model_training/
├── nejm_b2txt_utils/
├── setup.py
└── README.md
```

---

# Methodology

## 1. Neural Signal Acquisition

The system utilizes intracortical neural recordings collected during speech production tasks. These neural signals contain rich temporal information associated with language generation.

## 2. Signal Preprocessing

Raw neural activity undergoes preprocessing and feature extraction to generate structured representations suitable for deep learning models.

## 3. Transformer Encoder

A Transformer encoder captures long-range dependencies and contextual neural information using self-attention mechanisms.

### Benefits

- Long-range temporal modeling
- Context-aware neural representation
- Improved sequence understanding

## 4. Hybrid Mamba-GRU Decoder

The decoder combines:

### Mamba State Space Model

- Efficient sequence processing
- Reduced computational complexity
- Long-context information retention

### GRU Network

- Temporal memory retention
- Stable gradient propagation
- Sequential decoding capability

The combination provides a balance between efficiency and decoding accuracy.

## 5. Text Generation

Decoded neural representations are translated into natural language sequences through autoregressive text generation.

---

# Experimental Pipeline

```text
Dataset
   │
   ▼
Preprocessing
   │
   ▼
Feature Extraction
   │
   ▼
Transformer Encoding
   │
   ▼
Mamba-GRU Decoding
   │
   ▼
Text Prediction
   │
   ▼
Performance Evaluation
```

---

# Dataset

This project is based on the NEJM Brain-to-Text research framework and utilizes neural speech recordings for sequence-to-sequence learning.

### Dataset Characteristics

- Intracortical neural recordings
- Speech-related neural activity
- Time-series neural signals
- Text transcription labels
- High-dimensional neural feature space

---

# Technologies Used

| Category | Technologies |
|-----------|-------------|
| Programming | Python |
| Deep Learning | PyTorch |
| Sequence Modeling | Transformer |
| State Space Models | Mamba |
| Recurrent Networks | GRU |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib |
| Experimentation | Jupyter Notebook |
| Domain | Brain-Computer Interface |

---

# Results and Evaluation

## Evaluation Metrics

The model performance is evaluated using:

| Metric | Description |
|----------|------------|
| WER | Word Error Rate |
| CER | Character Error Rate |
| Accuracy | Sequence Prediction Accuracy |
| Loss | Training Objective |

---

## Model Performance Expectations

The proposed architecture is designed to achieve:

| Capability | Expected Improvement |
|-------------|--------------------|
| Long-Term Dependency Modeling | High |
| Temporal Sequence Learning | High |
| Computational Efficiency | Improved |
| Neural Signal Representation | Enhanced |
| Text Decoding Quality | Improved |
| Real-Time Inference Potential | Strong |

---

## Comparative Analysis

| Architecture | Strengths |
|-------------|-----------|
| GRU | Efficient temporal memory |
| Transformer | Long-range context modeling |
| Mamba | Efficient sequence processing |
| Transformer + Mamba-GRU | Context + Efficiency + Memory |

---

## Evaluation Outputs

Detailed evaluation outputs and WER metrics are stored in:

```text
docs/wer_results.json
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/Manikandan-official/Transformer-Based-Brain-to-Text-Decoding.git

cd Transformer-Based-Brain-to-Text-Decoding
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

or

```bash
pip install torch numpy pandas scipy matplotlib tqdm
```

---

# Training

## Transformer Training

```bash
python src/train_transformer.py
```

## Mamba-GRU Training

```bash
python src/train.py
```

---

# Evaluation

```bash
python src/evaluate.py
```

---

# Inference

```bash
python src/run_nejm_decoder.py
```

---

# Applications

### Assistive Communication

Enabling communication for individuals with severe speech impairments.

### Neural Prosthetics

Developing intelligent speech restoration systems.

### Healthcare AI

Improving patient quality of life through AI-powered neurotechnology.

### Brain-Computer Interfaces

Advancing direct neural communication systems.

### Computational Neuroscience

Understanding neural language representations.

---

# Research Significance

This project contributes to the growing field of Brain-Computer Interfaces by exploring modern deep learning architectures for neural speech decoding.

The combination of Transformer attention mechanisms and Mamba state-space modeling provides a promising direction for efficient and accurate neural language generation systems.

---

# Future Work

- Real-time Brain-to-Text deployment
- Large Language Model integration
- Multi-subject generalization
- Adaptive neural signal learning
- Transformer-Mamba fusion improvements
- Clinical communication system deployment
- Streaming neural speech decoding

---

# Keywords

Brain-Computer Interface (BCI) • Brain-to-Text • Neural Decoding • Deep Learning • Transformer Networks • Mamba State Space Models • GRU Networks • Neural Speech Restoration • Intracortical Recording • Sequence Modeling • Artificial Intelligence • Computational Neuroscience

---

# Repository Statistics

![GitHub Repo stars](https://img.shields.io/github/stars/Manikandan-official/Transformer-Based-Brain-to-Text-Decoding?style=for-the-badge)

![GitHub Forks](https://img.shields.io/github/forks/Manikandan-official/Transformer-Based-Brain-to-Text-Decoding?style=for-the-badge)

![GitHub Last Commit](https://img.shields.io/github/last-commit/Manikandan-official/Transformer-Based-Brain-to-Text-Decoding?style=for-the-badge)

---

# Author

### Manikandan

Researcher | AI Engineer | Deep Learning Enthusiast

GitHub:
https://github.com/Manikandan-official

---

# Citation

```bibtex
@misc{manikandan2025braintotext,
  title={Transformer-Based Brain-to-Text Decoding using Mamba-GRU Architectures},
  author={Manikandan},
  year={2025},
  publisher={GitHub},
  url={https://github.com/Manikandan-official/Transformer-Based-Brain-to-Text-Decoding}
}
```

---

# License

This project is released for research and educational purposes.

---

# Acknowledgements

- NEJM Brain-to-Text Research Team
- Brain-Computer Interface Research Community
- PyTorch Development Team
- Open-Source AI Community
- Computational Neuroscience Researchers