# Dual-Mode Intelligent Resume Screening Platform

##  Overview

An AI-powered resume screening application that uses advanced RAG (Retrieval Augmented Generation) technology to evaluate resumes against job descriptions through semantic similarity analysis. Unlike traditional keyword-matching systems, this platform provides domain-agnostic evaluation suitable for any industry or role.

##  Key Features

###  Single Resume Analysis Mode
- **Detailed Analysis**: Comprehensive evaluation of one resume against a job description
- **Section-wise Scoring**: Breaks down analysis by skills, experience, education, and overall content
- **Confidence Metrics**: Provides confidence levels for analysis quality
- **Key Insights**: Identifies strengths, missing skills, and actionable recommendations
- **Grade System**: Easy-to-understand A-F grading with percentage scores

###  Batch Shortlisting Mode
- **High-Volume Processing**: Efficiently process multiple resumes simultaneously
- **Intelligent Ranking**: Automatically ranks candidates by compatibility score
- **Top-N Selection**: Configurable shortlisting (e.g., top 10 candidates)
- **Batch Analytics**: Statistical summaries and score distributions
- **Export Capabilities**: Download detailed reports and candidate summaries

##  Technology Stack

- **LangChain**: Document processing and RAG pipeline orchestration
- **ChromaDB**: Vector database for efficient similarity search
- **Hugging Face Transformers**: Sentence-BERT embeddings for semantic understanding
- **Streamlit**: Interactive web interface
- **PyTorch**: Deep learning framework support
- **scikit-learn**: Additional ML utilities

##  Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Conda (recommended for environment management)

### Step 1: Create Virtual Environment
- Create new conda environment: conda create -n resume_analyzer python=3.10 -y
- Activate environment: conda activate resume_analyzer

### Step 2: Install Dependencies
- Install all required packages: pip install -r requirements.txt

### Step 3: Environment Configuration (Optional)
Create a `.env` file in the project root:
- EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
- CHROMA_PERSIST_DIRECTORY=./chroma_db
- MAX_BATCH_SIZE=100


##  Running the Application

### Start the Application
- From project root directory: streamlit run app/main.py

