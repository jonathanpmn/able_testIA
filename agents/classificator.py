import requests
import time
import json

# Local API base URL (ensure your FastAPI server is running)
LOCAL_API_BASE = "http://localhost:8000"

def run_vectorize(keyword):
    url = f"{LOCAL_API_BASE}/api/vectorize"
    params = {"keyword": keyword}
    response = requests.get(url, params=params)
    print("Vectorize Response:")
    print(response.json())

def run_classify(keyword, query_text):
    url = f"{LOCAL_API_BASE}/api/classify"
    payload = {
        "keyword": keyword,
        "query_text": query_text
    }
    response = requests.post(url, json=payload)
    print("Classification Response:")
    print(json.dumps(response.json(), indent=2))

def main():
    # Define the keywords used in the extractor phase.
    keywords = ["FAANG", "Facebook"]#, "Meta", "Apple", "Amazon", "Netflix", "Google", "Alphabet"]
    
    # Define a query_text that will be used for classification.
    query_text = ("Analyze the latest news on FAANG companies for Q1 2025 and determine which company offers "
                  "the best investment potential based on financial indicators, strategic moves, and regulatory developments.")
    
    # Iterate over each keyword to vectorize and classify the articles.
    for keyword in keywords:
        print(f"Running vectorization for keyword: {keyword}")
        run_vectorize(keyword)
        time.sleep(2)  # Wait for a bit between calls
        
        print(f"Running classification for keyword: {keyword}")
        run_classify(keyword, query_text)
        time.sleep(2)

if __name__ == "__main__":
    main()
