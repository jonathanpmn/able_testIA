import requests

# Local API base URL
LOCAL_BASE_URL = "http://localhost:8000"

def run_collect_articles(keyword):
    url = f"{LOCAL_BASE_URL}/api/collect_articles"
    params = {"keyword": keyword}
    response = requests.get(url, params=params)
    print("Collect Articles Response:")
    print(response.json())

def run_extract_text(keyword):
    url = f"{LOCAL_BASE_URL}/api/extract_text"
    params = {"keyword": keyword}
    response = requests.get(url, params=params)
    print("Extract Text Response:")
    print(response.json())

def main():
    # The agent defines the keywords
    keywords = ["FAANG", "Facebook"]#, "Meta", "Apple", "Amazon", "Netflix", "Google", "Alphabet"]
    
    for keyword in keywords:
        print(f"Running collection for keyword: {keyword}")
        run_collect_articles(keyword)
        print(f"Running text extraction for keyword: {keyword}")
        run_extract_text(keyword)

if __name__ == "__main__":
    main()
