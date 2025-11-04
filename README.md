# 🔍 Unsupervised News Topic Modeling: Fake vs Real News Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLP](https://img.shields.io/badge/NLP-NLTK%20%7C%20Gensim-green)](https://www.nltk.org/)
[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=Jupyter&logoColor=white)](https://jupyter.org/)

> 📰 An advanced unsupervised machine learning project leveraging natural language processing to analyze and distinguish patterns in fake versus real news articles through topic modeling and clustering techniques.

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
  - [LDA Topic Modeling](#lda-topic-modeling)
  - [Clustering Algorithms](#clustering-algorithms)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Overview

This project explores **unsupervised machine learning techniques** to analyze news articles without relying on labeled data. By applying advanced NLP methods, we uncover hidden patterns, topics, and structures that distinguish fake news from authentic journalism.

**Key Objectives:**
- 🎭 Identify latent topics in news articles using **Latent Dirichlet Allocation (LDA)**
- 🔬 Discover natural groupings through **clustering algorithms**
- 📊 Analyze linguistic patterns and text characteristics
- 🧩 Extract meaningful insights without supervised labels

## ✨ Key Features

- **📝 Advanced Text Preprocessing**: Tokenization, stopword removal, lemmatization, and normalization
- **🎨 Topic Modeling with LDA**: Discover hidden thematic structures in news corpus
- **🔍 Multiple Clustering Approaches**: K-Means, Hierarchical, and DBSCAN clustering
- **📈 Comprehensive Visualizations**: Interactive plots and word clouds for insights
- **🎯 Feature Engineering**: TF-IDF vectorization and custom text features
- **📊 Exploratory Data Analysis**: Statistical analysis and pattern recognition
- **🔬 Model Evaluation**: Silhouette scores, coherence metrics, and interpretability analysis

## 📚 Dataset

The project utilizes a comprehensive news dataset containing both fake and real news articles:

- **Source**: Publicly available fake news datasets (e.g., Kaggle Fake News Dataset)
- **Size**: Thousands of news articles across multiple categories
- **Features**: Article text, titles, authors, and publication metadata
- **Categories**: Politics, world news, entertainment, technology, etc.

### Dataset Characteristics:
- 📰 Real news from credible sources
- 🚫 Fake news from known misinformation sources
- 🌐 Diverse topics and writing styles
- 📅 Time-stamped articles for temporal analysis

## 🔬 Methodology

### LDA Topic Modeling

**Latent Dirichlet Allocation (LDA)** is a generative probabilistic model used to discover abstract topics within a collection of documents.

**How it works:**
1. 📄 Each document is represented as a mixture of topics
2. 🎯 Each topic is represented as a distribution of words
3. 🔄 Algorithm iteratively assigns words to topics based on probability
4. 📊 Reveals underlying thematic structures in the corpus

**Applications in this project:**
- Identify dominant themes in fake vs. real news
- Understand topic distributions across article types
- Discover emerging patterns and narrative structures

### Clustering Algorithms

Multiple unsupervised clustering techniques are employed to group similar articles:

#### 🎯 K-Means Clustering
- Partitions articles into K distinct clusters
- Fast and scalable for large datasets
- Identifies spherical groupings in feature space

#### 🌳 Hierarchical Clustering
- Creates a tree-like structure of nested clusters
- Provides multi-level granularity
- Useful for understanding article relationships

#### 🔍 DBSCAN (Density-Based Spatial Clustering)
- Discovers clusters of arbitrary shapes
- Automatically identifies outliers
- Robust to noise and irregularities

## 🛠️ Tech Stack

### Core Technologies

| Technology | Purpose | Badge |
|-----------|---------|-------|
| **Python 3.8+** | Primary programming language | ![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) |
| **scikit-learn** | Machine learning algorithms | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white) |
| **Gensim** | Topic modeling (LDA) | ![Gensim](https://img.shields.io/badge/Gensim-Topic%20Modeling-red) |
| **NLTK** | Natural language processing | ![NLTK](https://img.shields.io/badge/NLTK-NLP-green) |
| **pandas** | Data manipulation and analysis | ![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white) |
| **NumPy** | Numerical computing | ![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white) |
| **matplotlib** | Data visualization | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c) |
| **seaborn** | Statistical visualizations | ![Seaborn](https://img.shields.io/badge/Seaborn-3776AB) |
| **Jupyter** | Interactive development | ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white) |

### Additional Libraries
- `wordcloud` - Word cloud generation
- `plotly` - Interactive visualizations
- `spacy` - Advanced NLP (optional)

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step-by-step Installation

1. **Clone the repository**
```bash
git clone https://github.com/JayR1031/Unsupervised-News-Topic-Modeling.git
cd Unsupervised-News-Topic-Modeling
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data** (if required)
```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

### Requirements.txt

Create a `requirements.txt` file with the following dependencies:
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
gensim>=4.0.0
nltk>=3.6.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
wordcloud>=1.8.0
plotly>=5.0.0
```

## 💻 Usage

### Quick Start

1. **Open Jupyter Notebook**
```bash
jupyter notebook
```

2. **Run the main analysis notebook**
   - Navigate to the main notebook file
   - Execute cells sequentially to reproduce the analysis

### Workflow

```python
# 1. Load and preprocess data
from src.preprocessing import load_data, clean_text

data = load_data('path/to/dataset.csv')
cleaned_data = clean_text(data)

# 2. Topic Modeling with LDA
from src.topic_modeling import perform_lda

lda_model, topics = perform_lda(cleaned_data, num_topics=10)

# 3. Clustering Analysis
from src.clustering import kmeans_clustering

clusters = kmeans_clustering(cleaned_data, n_clusters=5)

# 4. Visualize Results
from src.visualization import plot_topics, plot_clusters

plot_topics(lda_model)
plot_clusters(clusters)
```

## 📊 Results

### Key Findings

🔍 **Topic Analysis:**
- Identified **X distinct topics** across the news corpus
- Fake news articles show higher concentration in specific topics
- Real news demonstrates more diverse topic distribution

📈 **Clustering Insights:**
- Articles naturally separate into **Y meaningful clusters**
- Strong correlation between cluster membership and news authenticity
- Outlier detection reveals unique narrative patterns

🎯 **Performance Metrics:**
- **Silhouette Score**: [0.XX] - indicating good cluster separation
- **Topic Coherence**: [0.XX] - measuring topic quality
- **Variance Explained**: [XX]% - by principal components

### Visualizations

The project includes comprehensive visualizations:
- 📊 Topic distribution heatmaps
- 🔴 Cluster scatter plots (PCA/t-SNE)
- ☁️ Word clouds for each topic
- 📈 Topic evolution over time
- 🎨 Confusion matrices and comparison charts

## 📁 Project Structure

```
Unsupervised-News-Topic-Modeling/
│
├── data/
│   ├── raw/                      # Original datasets
│   ├── processed/                # Cleaned and preprocessed data
│   └── README.md                 # Data documentation
│
├── notebooks/
│   ├── 01_data_exploration.ipynb # EDA and initial analysis
│   ├── 02_preprocessing.ipynb    # Text cleaning and preparation
│   ├── 03_topic_modeling.ipynb   # LDA implementation
│   ├── 04_clustering.ipynb       # Clustering algorithms
│   └── 05_results_visualization.ipynb  # Final visualizations
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          # Text preprocessing utilities
│   ├── topic_modeling.py         # LDA and topic analysis
│   ├── clustering.py             # Clustering algorithms
│   ├── visualization.py          # Plotting functions
│   └── utils.py                  # Helper functions
│
├── results/
│   ├── figures/                  # Generated plots and charts
│   ├── models/                   # Saved models
│   └── reports/                  # Analysis reports
│
├── tests/
│   └── test_*.py                 # Unit tests
│
├── .gitignore                    # Git ignore file
├── LICENSE                       # MIT License
├── README.md                     # This file
├── requirements.txt              # Python dependencies
└── setup.py                      # Package setup file
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### How to Contribute

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

### Contribution Ideas

- 🐛 Report bugs and issues
- 💡 Suggest new features or improvements
- 📝 Improve documentation
- 🧪 Add test coverage
- 🎨 Enhance visualizations
- 🔬 Experiment with new algorithms

### Code Style

- Follow PEP 8 guidelines
- Add docstrings to functions and classes
- Include type hints where appropriate
- Write unit tests for new features

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Jay Rodriguez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

## 👨‍💻 Author

**Jay Rodriguez**
- GitHub: [@JayR1031](https://github.com/JayR1031)

## 🙏 Acknowledgments

- Kaggle for providing the fake news datasets
- The open-source community for amazing ML/NLP libraries
- Research papers on topic modeling and fake news detection

## 📧 Contact

For questions or collaboration opportunities, please open an issue or reach out through GitHub.

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ and Python

</div>
