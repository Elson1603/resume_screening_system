# 🎯 AI-Powered Resume Screening System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![Transformers](https://img.shields.io/badge/🤗_Transformers-4.30+-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)
![Stars](https://img.shields.io/github/stars/Elson1603/resume_screening_system?style=social)

**Automate your hiring process with AI-powered resume screening and intelligent candidate ranking! **

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [API](#-api) • [Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Configuration](#-configuration)
- [API Documentation](#-api-documentation)
- [Results & Performance](#-results--performance)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🌟 Overview

The **AI-Powered Resume Screening System** is an intelligent solution designed to revolutionize the recruitment process. Using advanced Natural Language Processing (NLP) and Machine Learning techniques, this system automatically analyzes, ranks, and shortlists candidates based on job requirements.

### 🎥 Demo

![Demo GIF](https://via.placeholder.com/800x400/4A90E2/FFFFFF?text=Resume+Screening+System+Demo)

> *Upload resumes, define job requirements, and get ranked candidates in seconds!*

---

## ✨ Features

<table>
  <tr>
    <td width="33%" align="center">
      <h3>📄 Multi-Format Support</h3>
      <p>Upload resumes in PDF, DOCX, or TXT formats</p>
    </td>
    <td width="33%" align="center">
      <h3>🤖 AI-Powered Ranking</h3>
      <p>Uses transformer models for semantic understanding</p>
    </td>
    <td width="33%" align="center">
      <h3>⚡ Real-time Processing</h3>
      <p>Get instant candidate rankings and insights</p>
    </td>
  </tr>
  <tr>
    <td width="33%" align="center">
      <h3>🎯 Smart Matching</h3>
      <p>Matches skills, experience, and education</p>
    </td>
    <td width="33%" align="center">
      <h3>📊 Detailed Analytics</h3>
      <p>Comprehensive scoring and visualization</p>
    </td>
    <td width="33%" align="center">
      <h3>🌐 Web Interface</h3>
      <p>User-friendly Flask-based web application</p>
    </td>
  </tr>
</table>

### 🔑 Key Capabilities

- ✅ **Semantic Text Analysis** - Goes beyond keyword matching with deep learning
- ✅ **Experience Extraction** - Automatically identifies years of experience
- ✅ **Skills Detection** - Extracts technical and soft skills from resumes
- ✅ **Education Matching** - Validates educational qualifications
- ✅ **Batch Processing** - Handle multiple resumes simultaneously
- ✅ **Customizable Criteria** - Define your own job requirements
- ✅ **Export Results** - Download ranked candidates as CSV

---

## 🛠️ Tech Stack

<div align="center">

| Category | Technologies |
|----------|-------------|
| **Backend** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white) |
| **AI/ML** | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![HuggingFace](https://img.shields.io/badge/🤗_Transformers-FFD21E?style=flat) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E? style=flat&logo=scikit-learn&logoColor=white) |
| **NLP** | ![NLTK](https://img.shields.io/badge/NLTK-154f3c?style=flat) ![spaCy](https://img.shields.io/badge/spaCy-09A3D5?style=flat&logo=spacy&logoColor=white) ![Sentence Transformers](https://img.shields.io/badge/Sentence_Transformers-orange?style=flat) |
| **Data** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243? style=flat&logo=numpy&logoColor=white) |
| **Visualization** | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) ![Seaborn](https://img.shields.io/badge/Seaborn-9cf?style=flat) |

</div>

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step-by-Step Installation

1. **Clone the repository**

```bash
git clone https://github.com/Elson1603/resume_screening_system.git
cd resume_screening_system
```

2. **Create a virtual environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download required NLP models**

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
python -m spacy download en_core_web_sm
```

5. **Create necessary directories**

```bash
mkdir -p uploads data models results
```

---

## 🚀 Quick Start

### Running the Web Application

```bash
python app.py
```

Then open your browser and navigate to: 
```
http://localhost:5000
```

### Running CLI Version

```bash
python main.py
```

---

## 💻 Usage

### Web Interface

1. **Start the Application**
   ```bash
   python app.py
   ```

2. **Upload Resumes**
   - Navigate to the upload page
   - Select multiple resume files (PDF, DOCX, TXT)
   - Enter job description
   - Set required experience (years)
   - Specify required education level

3. **View Results**
   - See ranked candidates with scores
   - View detailed analytics
   - Export results as CSV

### Command Line Interface

```python
from src.inference. resume_ranking import ResumeRanker
import pandas as pd

# Initialize ranker
ranker = ResumeRanker()

# Define job requirements
job_description = """
We are looking for a Python Developer with experience in Machine Learning. 
Strong skills in Python, TensorFlow, and scikit-learn required. 
Minimum 2 years of experience. 
"""

# Load and process resumes
df_resumes = pd.read_csv('data/resumes.csv')

# Rank candidates
scores_df = ranker.rank_candidates(
    df_resumes,
    job_description,
    required_experience=2,
    required_education="Bachelor's degree"
)

# Display top candidates
ranker.display_top_candidates(scores_df, top_n=10)
```

---

## 📁 Project Structure

```
resume_screening_system/
│
├── 📄 app.py                          # Flask web application
├── 📄 main.py                         # CLI entry point
├── 📄 requirements.txt                # Project dependencies
├── 📄 generate_results_documentation.py  # Results generator
│
├── 📂 src/                            # Source code
│   ├── 📄 config.py                   # Configuration settings
│   ├── 📂 training/                   # Training modules
│   │   ├── 📄 data_preprocessing.py   # Data preprocessing
│   │   └── 📄 model_trainer.py        # Model training
│   └── 📂 inference/                  # Inference modules
│       ├── 📄 resume_ranking.py       # Main ranking logic
│       └── 📄 utils.py                # Utility functions
│
├── 📂 templates/                      # HTML templates
│   └── 📄 index.html                  # Web interface
│
├── 📂 static/                         # Static files (CSS, JS, images)
│
├── 📂 data/                           # Data directory
│   ├── 📂 raw/                        # Raw resume data
│   └── 📂 processed/                  # Processed data
│
├── 📂 models/                         # Trained models
│
├── 📂 results/                        # Output results
│
└── 📂 uploads/                        # Uploaded resumes
```

---

## 🧠 How It Works

```mermaid
graph LR
    A[Upload Resumes] --> B[Text Extraction]
    B --> C[Text Cleaning]
    C --> D[Feature Extraction]
    D --> E[Embedding Generation]
    E --> F[Similarity Calculation]
    F --> G[Score Aggregation]
    G --> H[Candidate Ranking]
    H --> I[Results Display]
```

### Processing Pipeline

1. **📥 Document Ingestion**
   - Accepts PDF, DOCX, and TXT files
   - Extracts raw text from documents

2. **🧹 Text Preprocessing**
   - Removes special characters and noise
   - Tokenization and lemmatization
   - Stopword removal

3. **🔍 Feature Extraction**
   - Skills extraction using keyword matching
   - Experience years detection with regex
   - Education level identification

4. **🤖 Semantic Analysis**
   - Generates embeddings using Sentence Transformers
   - Calculates semantic similarity with job description
   - Uses cosine similarity for matching

5. **📊 Scoring & Ranking**
   - Combines multiple signals: 
     - Semantic similarity (40%)
     - Skills match (30%)
     - Experience alignment (20%)
     - Education match (10%)
   - Generates final composite score

6. **✅ Candidate Shortlisting**
   - Ranks candidates by final score
   - Flags top candidates for shortlisting
   - Provides detailed breakdown

---

## ⚙️ Configuration

Edit `src/config.py` to customize settings:

```python
class Config:
    # Directories
    DATA_DIR = 'data'
    MODELS_DIR = 'models'
    RESULTS_DIR = 'results'
    
    # Model settings
    EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
    
    # Scoring weights
    SIMILARITY_WEIGHT = 0.4
    SKILLS_WEIGHT = 0.3
    EXPERIENCE_WEIGHT = 0.2
    EDUCATION_WEIGHT = 0.1
    
    # Thresholds
    SHORTLIST_THRESHOLD = 0.65
```

---

## 📖 API Documentation

### REST API Endpoints

#### Upload and Rank Resumes

```http
POST /api/rank
Content-Type: multipart/form-data

Parameters:
- resumes:  File[] (resume files)
- job_description: string
- required_experience: integer
- required_education: string

Response:
{
  "status": "success",
  "total_candidates": 25,
  "shortlisted":  8,
  "results": [
    {
      "filename": "john_doe.pdf",
      "final_score": 0.8542,
      "skills_match": 0.85,
      "experience_match": 0.90,
      "shortlisted": true
    },
    ...
  ]
}
```

#### Get Candidate Details

```http
GET /api/candidate/{id}

Response:
{
  "id": "123",
  "name": "John Doe",
  "score": 0.8542,
  "skills": ["Python", "Machine Learning", "TensorFlow"],
  "experience_years": 4,
  "education": "Master's Degree"
}
```

---

## 📊 Results & Performance

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 87.3% |
| **Precision** | 84.6% |
| **Recall** | 89.1% |
| **F1-Score** | 86.8% |
| **Processing Speed** | ~2 sec/resume |

### Sample Results

Check out [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) for detailed analysis and benchmarks.

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation
- Ensure all tests pass

---

## 🐛 Troubleshooting

<details>
<summary><b>Issue:  NLTK data not found</b></summary>

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```
</details>

<details>
<summary><b>Issue: spaCy model not found</b></summary>

```bash
python -m spacy download en_core_web_sm
```
</details>

<details>
<summary><b>Issue: Memory error with large files</b></summary>

Reduce batch size in `config.py` or process resumes in smaller batches.
</details>

---

## 🎯 Roadmap

- [x] Basic resume screening
- [x] Multi-format support
- [x] Web interface
- [ ] Real-time notifications
- [ ] Email integration
- [ ] Advanced analytics dashboard
- [ ] Multi-language support
- [ ] ATS integration
- [ ] Mobile application
- [ ] API authentication

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Elson1603**

- GitHub: [@Elson1603](https://github.com/Elson1603)
- Repository: [resume_screening_system](https://github.com/Elson1603/resume_screening_system)

---

## 🌟 Show Your Support

If you find this project useful, please consider: 

- ⭐ Starring the repository
- 🐛 Reporting bugs or issues
- 💡 Suggesting new features
- 📢 Sharing with others

---

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for transformer models
- [Sentence Transformers](https://www.sbert.net/) for semantic embeddings
- [Flask](https://flask.palletsprojects.com/) for web framework
- The open-source community

---

<div align="center">

**Made with ❤️ by Elson1603**

[⬆ Back to Top](#-ai-powered-resume-screening-system)

</div>
