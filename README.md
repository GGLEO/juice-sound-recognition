# 🍹 Juice Sound Recognition

This project aims to recognize different types of juice based on their pouring sound using deep learning techniques.

## 📌 Project Description

We propose an audio-based juice classification system that analyzes the sound of pouring liquids to identify the juice type. This system extracts acoustic features and uses deep learning models such as CNN, RNN, CRNN, Transformer, and AST to perform classification.

## 🧪 Experimental Setup

- **Recording Device:** OPPO Reno5 built-in microphone  
- **Environment:** Residential indoor setting  
- **Juice Types:** Water, Apple Juice, Cold Tea  
- **Audio Format:** `.mp3` converted to `.wav`

### Data Preparation
- Converted `.mp3` to `.wav`
- Applied noise reduction and volume normalization
- Extracted features using:
  - MFCC (13-dim with Δ & ΔΔ)
  - Mel Spectrogram (128-dim)

### Models Used
- CNN
- RNN
- CRNN
- Transformer
- AST (Audio Spectrogram Transformer)

### Training Strategy
- 60/40 train-test split
- 5-fold cross-validation
- EarlyStopping and ReduceLROnPlateau used

## 📊 Results

| Train Size | Feature | CNN | RNN | CRNN | Transformer | AST |
|------------|---------|-----|-----|------|-------------|-----|
| 20         | MFCC    | 0.5333 | 0.6167 | 0.6500 | 0.4500 | - |
|            | Mel     | 0.5000 | 0.3333 | 0.5000 | 0.5667 | 0.8833 |
| 40         | MFCC    | 0.7000 | 0.8000 | 0.9833 | 0.4167 | - |
|            | Mel     | 0.5500 | 0.3333 | 0.5833 | 0.6000 | 0.9667 |
| 60         | MFCC    | 1.0000 | 1.0000 | 1.0000 | 1.0000 | - |
|            | Mel     | 1.0000 | 0.3333 | 0.4667 | 1.0000 | 1.0000 |

## 💻 Environment

- **OS:** macOS (Apple M2)
- **Language:** Python 3.9
- **Frameworks:** TensorFlow, NumPy, librosa, matplotlib
- **Notebook:** Jupyter via Anaconda

## 🔮 Future Work

- Expand dataset with more juice types
- Explore deep-learning-based audio feature extraction
- Add noise-robust mechanisms
- Apply the approach to other food types like coffee or tea

## 📁 Project Structure

