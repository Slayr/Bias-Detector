# Media Bias Detector

A comprehensive tool for detecting media bias using a hybrid approach. This project leverages both lightweight machine learning and deep learning to identify biased language in news sentences. By combining multiple datasets and providing explainable AI features, it aims to offer a robust and transparent solution for media analysis.

## Overview

The Media Bias Detector is designed to identify various forms of bias in news text, including explicit partisan attacks, loaded language, subjective statements framed as objective facts, and one-sided framing. It utilizes a multi-backend architecture to balance performance and accuracy.

## Features

- **Multi-tier Backend Support**:
  - `rf`: Random Forest classifier using `nomic-embed-text` embeddings via Ollama. This backend is fast and does not require a GPU.
  - `local`: Fine-tuned DeBERTa model loaded directly via the Transformers library. This offers the highest accuracy and includes multi-label support.
  - `llm`: Integration with fine-tuned models hosted on Ollama for remote or containerized inference.
- **Explainable AI (XAI)**: Integrated `transformers-interpret` for the `local` backend to highlight specific words contributing to the bias verdict through attribution scores.
- **Multi-label Classification**: The system does not only detect "Bias" as a binary label but also predicts "Partisan" and "Subjective Opinion" traits when using the `local` backend.
- **Batch Processing**: Tools to classify entire CSV files of text, outputting detailed results including probabilities and uncertainty flags.
- **Interactive CLI**: A command-line interface powered by the Rich library for real-time analysis and visualization of model confidence.

## Installation

### Prerequisites

- Python 3.9 or higher
- [Ollama](https://ollama.com) (required for the `rf` and `llm` backends)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Slayr/bias-detector.git
   cd bias-detector
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Ollama**:
   Ensure Ollama is running and pull the required embedding model:
   ```bash
   ollama pull nomic-embed-text
   ```

## Data Preparation

The detector is trained and evaluated on a combination of three major datasets:
- **MBIC**: A media bias annotation dataset by Spinde et al. (2021).
- **BABE**: Media bias annotations from the Media Bias Group.
- **MBIB**: A benchmark for media bias identification.

### Download Datasets

Run the provided scripts to download and format the BABE and MBIB datasets:
```bash
python download_babe.py
python download_mbib.py
```

*Note: For the MBIC dataset, you must download `mbic.xlsx` from [Kaggle](https://www.kaggle.com/datasets/timospinde/mbic-a-media-bias-annotation-dataset) and place it in the `data/` folder.*

## Training and Fine-tuning

### Fine-tune DeBERTa (Recommended for Accuracy)

To train the deep learning model (requires a GPU for reasonable speed):
```bash
python finetune.py --epochs 3 --batch 16
```
This script uses soft labels to account for annotator agreement and saves the resulting model to `models/finetuned/`.

### Train Random Forest (Lightweight)

To train the Random Forest model using embeddings from Ollama:
```bash
python main.py train
```
This will embed the MBIC dataset and save a local Random Forest classifier to `models/bias_detector.joblib`.

## Usage

### Interactive Mode

Test individual sentences through the CLI:
```bash
# Using Random Forest (Default)
python main.py detect

# Using the Fine-tuned DeBERTa model
python main.py detect --backend local
```

### Batch Processing

Classify a CSV file of sentences. The input file must have a column named `text` or `sentence`.
```bash
python main.py batch -i input.csv -o output.csv --backend local
```

### Evaluation

Run a multi-tier evaluation suite that tests the model across:
1. In-distribution data (MBIC test split)
2. Cross-dataset generalization (BABE and MBIB)
3. Hand-curated hard cases (Neutral reporting vs. loaded language)

```bash
python evaluate.py
```

## Technical Configuration

The application can be configured via environment variables or by modifying the configuration section in `main.py`:

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBED_MODEL` | Ollama embedding model for RF | `nomic-embed-text` |
| `CONFIDENCE_THRESHOLD` | Threshold to flag uncertainty | `0.65` |
| `MBIC_CSV` | Path to the MBIC excel file | `mbic.xlsx` |
| `FINETUNE_BASE` | Base transformer model for fine-tuning | `microsoft/deberta-v3-base` |

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for the full text.

## Acknowledgments

This research and implementation rely on datasets provided by Spinde et al. and the Media Bias Group. Their contributions to the field of media bias detection are gratefully acknowledged.
