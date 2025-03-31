# FAANG Market Analysis Agentic Workflow

This project implements a multi-agent workflow for market analysis on FAANG companies (Facebook/Meta, Apple, Amazon, Netflix, Google/Alphabet) using online content. The system collects news articles from Reuters, extracts relevant information and full text, vectorizes the data using embeddings, performs sentiment and economic classification via OpenAI models, and finally generates an investment report indicating which FAANG company is the best investment as of Q1 2025.

The project was developed with the assistance of AI to accelerate development and implement best practices throughout the workflow.

---

## 🧠 Overview

The workflow consists of the following steps:

1. **Extractor Agent**  
   Collects articles from Reuters based on a keyword, filters them by the last 90 days, and saves the data (`link`, `keyword`, `date`, `title`) as Parquet files in the `output/articles` folder.

2. **Text Extraction**  
   Reads the collected articles, extracts the full text using BeautifulSoup, and saves a new Parquet file (with an additional `text` column) in the same folder.

3. **Vectorization & Classification (Classificator Agent)**  
   Reads the Parquet file with full text, generates embeddings using OpenAI’s `text-embedding-3-small` model, creates a FAISS index, and performs sentiment/economic classification using a chat model (default: `o1-mini`). The results are saved as a timestamped Parquet file in `output/classification`.

4. **Final Report Generation (Final Report Agent)**  
   Aggregates classification results from the `output/classification` folder and calls an API to generate a final investment report. The report is saved in the `output/report` folder.

---

## ⚙️ Setup and Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd project
```

### 2. Create and Activate a Virtual Environment (recommended)

```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set the OpenAI API Key

#### On macOS/Linux:

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

#### On Windows:

```bash
set OPENAI_API_KEY="your-openai-api-key"
```

---

## 🚀 Running the Application

### Start the FastAPI Server

```bash
python -m app.main
```

> Access the API at [http://localhost:8000](http://localhost:8000)

---

### Run the Extractor Agent

```bash
python agents/extractor.py
```

> Collects articles and saves them in `output/articles`.

---

### Run the Classification Agent

```bash
python agents/classificator.py
```

> Vectorizes and classifies the articles. Results are saved in `output/classification`.

---

### Run the Final Report Agent

```bash
python agents/final_report.py
```

> Generates a final investment report based on the latest classification results and saves it in `output/report`.

---

## 📝 Notes

This project was developed with the support of AI tools to accelerate implementation and improve code quality. Feel free to adapt or extend it for your own use cases.
