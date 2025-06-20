# Indonesian Emotion Tweet Web Application

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![PyTorch](https://img.shields.io/badge/pytorch-v1.13+-orange.svg)
![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)

A comprehensive web application for analyzing emotions in Indonesian text using IndoBert or Word2Vec models. This application supports both single text analysis and batch processing with an intuitive Streamlit interface.

## 🌟 Features

- **Dual Model Support**: Choose between IndoBERT and Word2Vec + Logistic Regression models
- **Real-time Analysis**: Instant emotion prediction with confidence scores
- **Interactive Visualizations**: Dynamic probability distribution charts
- **Batch Processing**: Upload CSV files for bulk emotion analysis
- **Preprocessing Details**: View step-by-step text preprocessing
- **Modern UI**: Clean, responsive Streamlit interface with custom styling
- **Download Results**: Export batch analysis results as CSV

## 🎯 Supported Emotions

The application can detect 5 different emotions in Indonesian text:
- **Anger** (Marah)
- **Fear** (Takut)
- **Happy** (Senang)
- **Love** (Cinta)
- **Sadness** (Sedih)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for faster IndoBERT inference)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/raddhuha/indo-emotion-tweet.git
   cd indo-emotion-tweet
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download required models and data**
   
   Make sure you have the following files in your project directory:
   ```
   saved_models/
   ├── preprocessing_components.pkl
   ├── label_encoder.pkl
   ├── word2vec_logistic_model.pkl
   ├── indobert_emotion_model.pth
   └── tokenizer/
       ├── config.json
       ├── pytorch_model.bin
       ├── tokenizer_config.json
       ├── tokenizer.json
       └── vocab.txt
   Word2Vec_400dim.txt
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Open your browser**
   
   Navigate to `http://localhost:8501` to access the application.

### 🐳 Docker Setup (Optional)

```bash
# Build Docker image
docker build -t indo-emotion-tweet .

# Run container
docker run -p 8501:8501 indo-emotion-tweet
```

## 📋 Requirements
    
### System Requirements

- **RAM**: Minimum 8GB (16GB recommended for IndoBERT)
- **Storage**: 2GB free space for models
- **Python**: 3.8, 3.9, 3.10, or 3.11
- **OS**: Windows 10+, macOS 10.15+, Ubuntu 18.04+

## 🔧 Usage

### Single Text Analysis

1. Navigate to the **"📝 Analisis Teks"** tab
2. Enter your Indonesian text in the input field
3. Select your preferred model (IndoBERT or Word2Vec)
4. Click **"🔍 Analisis Emosi"** to get results
5. View the predicted emotion, confidence score, and probability distribution

### Batch Processing

1. Go to the **"📁 Analisis Batch"** tab
2. Upload a CSV file with a 'text' column
3. Click **"🔍 Analisis Batch"** to process all texts
4. Download the results as a CSV file

### Preprocessing Details

1. After analyzing text, visit the **"🔧 Detail Preprocessing"** tab
2. View each preprocessing step:
   - Original text
   - Cleaned text (removed URLs, mentions, special characters)
   - Substituted text (slang words replaced)
   - Final processed text (lemmatized)

## 🤖 Models

### IndoBERT Model
- **Architecture**: BERT-base transformer fine-tuned for Indonesian
- **Performance**: 
  - Training Accuracy: ~99.9%
  - Validation Accuracy: ~71.0%
  - Test Accuracy: ~70.5%
- **Best for**: High accuracy emotion detection

### Word2Vec + Logistic Regression
- **Architecture**: 400-dimensional Word2Vec embeddings with Logistic Regression
- **Performance**:
  - Training Accuracy: ~73.6%
  - Validation Accuracy: ~61.2%
  - Test Accuracy: ~58.7%
- **Best for**: Faster inference, interpretable results

## 📊 Model Performance Comparison

| Model | Train Acc | Val Acc | Test Acc | Speed | Memory |
|-------|-----------|---------|----------|-------|---------|
| IndoBERT | 99.9% | 71.0% | 70.5% | Slower | Higher |
| Word2Vec + LR | 73.6% | 61.2% | 58.7% | Faster | Lower |

## 🔄 Preprocessing Pipeline

1. **Text Cleaning**
   - Remove URLs, mentions (@username), hashtags
   - Remove special characters and numbers
   - Convert to lowercase

2. **Slang Substitution**
   - Replace Indonesian slang words with formal equivalents
   - Uses predefined dictionary mapping

3. **Lemmatization**
   - Reduce words to their root forms
   - POS tagging for accurate lemmatization

4. **Tokenization**
   - Split text into tokens for model input

### 🚀 How to Contribute

1. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **Make your changes**
   - Add new features
   - Fix bugs
   - Improve documentation
   - Add tests

3. **Test your changes**
   ```bash
   python -m pytest tests/
   ```

4. **Commit with descriptive message**
   ```bash
   git commit -m "Add: Amazing new feature for emotion detection"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```

6. **Create Pull Request**
   - Go to GitHub and create a PR
   - Describe your changes
   - Link any related issues

### 📝 Contribution Guidelines

- **Code Style**: Follow PEP 8
- **Documentation**: Update README and docstrings
- **Testing**: Add tests for new features
- **Commits**: Use conventional commit messages
- **Issues**: Check existing issues before creating new ones

## 📖 Acknowledgments

- **IndoBERT**: [IndoNLU Benchmark](https://github.com/indobenchmark/indonlu)
- **Transformers**: [Hugging Face Transformers](https://github.com/huggingface/transformers)
- **Streamlit**: [Streamlit Framework](https://streamlit.io/)

### 🌟 Show Your Support

If this project helped you, please consider:

- ⭐ **Star** this repository
- 🍴 **Fork** and contribute
- 📢 **Share** with others
- 💬 **Provide feedback**

[![GitHub stars](https://img.shields.io/github/stars/raddhuha/indo-emotion-tweet.svg?style=social&label=Star)](https://github.com/raddhuha/indo-emotion-tweet)
[![GitHub forks](https://img.shields.io/github/forks/raddhuha/indo-emotion-tweet.svg?style=social&label=Fork)](https://github.com/raddhuha/indo-emotion-tweet/fork)

---

<div align="center">

[![Visitors](https://visitor-badge.laobi.icu/badge?page_id=raddhuha.indo-emotion-tweet)](https://github.com/raddhuha/indo-emotion-tweet)
[![Last Commit](https://img.shields.io/github/last-commit/raddhuha/indo-emotion-tweet)](https://github.com/raddhuha/indo-emotion-tweet/commits/main)
[![Issues](https://img.shields.io/github/issues/raddhuha/indo-emotion-tweet)](https://github.com/raddhuha/indo-emotion-tweet/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/raddhuha/indo-emotion-tweet)](https://github.com/raddhuha/indo-emotion-tweet/pulls)

</div>