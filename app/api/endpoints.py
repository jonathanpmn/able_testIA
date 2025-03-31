from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import os
import openai
import faiss

router = APIRouter()

# Constants for Reuters API and Base URL
REUTERS_API_URL = "https://www.reuters.com/pf/api/v3/content/fetch/articles-by-search-v2"
BASE_URL = "https://www.reuters.com"
HEADERS = {
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.reuters.com/site-search/?query=faang",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
}
# Number of articles to return
RETURNED_ARTICLES = 2

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

def ensure_directory(path: str):
    """Ensure that a directory exists."""
    os.makedirs(path, exist_ok=True)

def is_recent(publication_date_str: str, days: int = 90) -> bool:
    try:
        publication_date = datetime.strptime(publication_date_str, "%Y-%m-%d")
        cutoff_date = datetime.today() - timedelta(days=days)
        return publication_date >= cutoff_date
    except Exception:
        return False

def extract_text_from_url(url: str) -> str:
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            return ""
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("div", {"data-testid": lambda x: x and x.startswith("paragraph-")})
        if not paragraphs:
            return ""
        text = "\n".join([p.get_text() for p in paragraphs])
        return text
    except Exception:
        return ""

def get_embedding(text: str) -> list:
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-3-small"
    )
    embedding = response["data"][0]["embedding"]
    return embedding

# --- Model-Agnostic Classification Helper ---
def call_classification_model(model_name: str, full_prompt: str, temperature: float = 1.0) -> str:
    """
    Calls the classification model based on the model_name provided.
    If the model is a chat model (e.g., "o1-mini"), it uses ChatCompletion;
    otherwise, it uses Completion.
    
    This function can be adapted later to support other models.
    """
    if model_name == "o1-mini":
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=temperature
        )
        return response["choices"][0]["message"]["content"].strip()
    else:
        response = openai.Completion.create(
            model=model_name,
            prompt=full_prompt,
            temperature=temperature
        )
        return response["choices"][0]["text"].strip()

# --- Endpoints ---
@router.get("/api/collect_articles")
async def collect_articles(keyword: str = Query(..., description="Keyword for searching articles")):
    """
    Collect articles from the Reuters API by keyword, filter those from the last 90 days,
    and save them in a Parquet file with columns: link, keyword, date, and title.
    Files are saved in the folder output/articles.
    """
    query_str = f'{{"keyword":"{keyword}","offset":0,"orderby":"display_date:desc","size":{RETURNED_ARTICLES},"website":"reuters"}}'
    params = {
        "query": query_str,
        "d": "275",
        "mxId": "00000000",
        "_website": "reuters"
    }
    try:
        response = requests.get(REUTERS_API_URL, params=params, headers=HEADERS)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Error fetching articles from Reuters API")
        data = response.json()
        if "result" not in data or "articles" not in data["result"]:
            return JSONResponse(content={"message": "No articles found."})
        articles = data["result"]["articles"]
        records = []
        for article in articles:
            relative_link = article.get("canonical_url", "")
            full_link = BASE_URL + relative_link
            published_time = article.get("published_time", "")
            publication_date = published_time[:10] if published_time else ""
            title = article.get("title", "")
            if publication_date and is_recent(publication_date, 90):
                records.append({
                    "link": full_link,
                    "keyword": keyword,
                    "date": publication_date,
                    "title": title
                })
        if records:
            df = pd.DataFrame(records)
            directory = "output/articles"
            ensure_directory(directory)
            filename = f"{directory}/articles_{keyword}.parquet"
            df.to_parquet(filename, index=False)
            return JSONResponse(content={"message": f"Collected {len(records)} articles and saved to {filename}"})
        else:
            return JSONResponse(content={"message": "No recent articles found for the given keyword."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/extract_text")
async def extract_text(keyword: str = Query(..., description="Keyword for extracting text from saved articles")):
    """
    Load the Parquet file from collect_articles endpoint,
    extract article text using BeautifulSoup,
    and save a new Parquet file with an additional 'text' column.
    Files are saved in the folder output/articles.
    """
    filename = f"output/articles/articles_{keyword}.parquet"
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail=f"File {filename} not found. Please run the collect_articles endpoint first.")
    
    try:
        df = pd.read_parquet(filename)
        texts = []
        for index, row in df.iterrows():
            url = row["link"]
            article_text = extract_text_from_url(url)
            texts.append(article_text)
        df["text"] = texts
        directory = "output/articles"
        ensure_directory(directory)
        new_filename = f"{directory}/articles_text_{keyword}.parquet"
        df.to_parquet(new_filename, index=False)
        return JSONResponse(content={"message": f"Extracted text for {len(df)} articles and saved to {new_filename}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/vectorize")
async def vectorize_articles(keyword: str = Query(..., description="Keyword for vectorizing the articles")):
    """
    Generate embeddings for articles saved in the Parquet file (with text) for the given keyword,
    create a FAISS index, and save the index along with metadata.
    The index is saved in output/index and metadata in output/metadata.
    """
    parquet_file = f"output/articles/articles_text_{keyword}.parquet"
    if not os.path.exists(parquet_file):
        raise HTTPException(status_code=404, detail=f"Parquet file {parquet_file} not found. Please run the extract_text endpoint first.")
    
    try:
        df = pd.read_parquet(parquet_file)
        embeddings = []
        texts = df["text"].tolist()
        for idx, text in enumerate(texts):
            emb = get_embedding(text)
            embeddings.append(emb)
        embeddings_array = np.array(embeddings).astype("float32")
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        index_directory = "output/index"
        ensure_directory(index_directory)
        index_filename = f"{index_directory}/faiss_index_{keyword}.index"
        faiss.write_index(index, index_filename)
        metadata_directory = "output/metadata"
        ensure_directory(metadata_directory)
        metadata_filename = f"{metadata_directory}/metadata_{keyword}.csv"
        df.to_csv(metadata_filename, index=False)
        return JSONResponse(content={"message": f"Vectorization complete. Index saved to {index_filename} and metadata saved to {metadata_filename}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/classify")
async def classify_news(data: dict):
    """
    Classify news based on a query using the FAISS index and metadata.
    Expects a JSON body with:
      - 'keyword': the keyword corresponding to the articles.
      - 'query_text': the text used to query the vector index.
    Uses a model (default: o1-mini) via the call_classification_model helper to perform sentiment/economic analysis.
    The classification result is saved in a Parquet file in output/classification with a timestamp in its name.
    """
    keyword = data.get("keyword")
    query_text = data.get("query_text")
    if not keyword or not query_text:
        raise HTTPException(status_code=400, detail="Both 'keyword' and 'query_text' are required.")
    
    index_filename = f"output/index/faiss_index_{keyword}.index"
    metadata_filename = f"output/metadata/metadata_{keyword}.csv"
    if not os.path.exists(index_filename) or not os.path.exists(metadata_filename):
        raise HTTPException(status_code=404, detail="Index or metadata file not found. Please run the vectorize endpoint first.")
    
    try:
        index = faiss.read_index(index_filename)
        metadata_df = pd.read_csv(metadata_filename)
        query_embedding = np.array(get_embedding(query_text)).astype("float32").reshape(1, -1)
        k = 3
        distances, indices = index.search(query_embedding, k)
        top_results = metadata_df.iloc[indices[0]].to_dict(orient="records")
        
        prompt = (
            "You are an intermediate agent specialized in analyzing news articles and texts related to FAANG companies (Facebook/Meta, Apple, Amazon, Netflix, Google/Alphabet). "
            "Your role is to extract relevant and structured information to facilitate subsequent market analysis. Upon receiving a text, you must:\n\n"
            "Identify the mentioned FAANG company (Facebook/Meta, Apple, Amazon, Netflix, Google/Alphabet).\n\n"
            "Highlight significant events described in the text (e.g., new product launches, executive changes, quarterly financial results, acquisitions, partnerships, regulatory processes, etc.).\n\n"
            "Extract mentioned financial indicators (revenue, profit, margins, debts, investments).\n\n"
            "Summarize any mentioned market trends or competitive strategies.\n\n"
            "Return your analysis in a clear, structured, and objective format, containing:\n\n"
            "Company Name\n\n"
            "Date of the event or news (if mentioned)\n\n"
            "Type of event (Financial, Strategic, Regulatory, Product, etc.)\n\n"
            "Main facts and extracted data\n\n"
            "Potential implications or impact of the event for the company\n\n"
            "If the text mentions more than one FAANG company, analyze each company separately."
        )
        context = "\n".join([f"Title: {res['title']}\nText: {res.get('text', '')}" for res in top_results])
        full_prompt = f"{prompt}\n\nContext:\n{context}"
        
        model_name = "o1-mini"
        classification_result = call_classification_model(model_name, full_prompt, temperature=1.0)
        
        # Save classification result in a Parquet file with a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_df = pd.DataFrame([{
            "keyword": keyword,
            "classification": classification_result,
            "timestamp": timestamp
        }])
        classification_directory = "output/classification"
        ensure_directory(classification_directory)
        classification_filename = f"{classification_directory}/results_classification_{timestamp}.parquet"
        result_df.to_parquet(classification_filename, index=False)
        
        return JSONResponse(content={
            "message": "Classification complete.",
            "classification": classification_result,
            "saved_file": classification_filename
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/generate_report")
async def generate_report(filename: str = Query(..., description="Filename of the classification results Parquet file (including path)")):
    """
    Reads the classification results from the specified Parquet file,
    aggregates them, and calls a GPT model (o1-mini) to generate a final investment report.
    The report describes which FAANG company is the best investment as of Q1 2025.
    The final report is saved in output/report.
    """
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail=f"File {filename} not found. Please provide a valid classification file.")
    
    try:
        df = pd.read_parquet(filename)
        aggregated_classifications = "\n\n".join(df["classification"].tolist())
        
        prompt = (
            "You are an expert financial market analyst. Based on the following classification analyses of FAANG news, "
            "generate a final investment report that determines which FAANG company is the best investment as of Q1 2025. "
            "Consider financial indicators, strategic moves, and regulatory factors mentioned in the analyses. "
            "Provide a clear recommendation along with a brief explanation.\n\n"
            "Classification Analyses:\n" + aggregated_classifications + "\n\nFinal Report:"
        )
        
        final_report = call_classification_model("o1-mini", prompt, temperature=1.0)
        
        report_directory = "output/report"
        ensure_directory(report_directory)
        report_filename = f"{report_directory}/final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(final_report)
        
        return JSONResponse(content={
            "message": "Final report generated.",
            "final_report": final_report,
            "report_file": report_filename
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
