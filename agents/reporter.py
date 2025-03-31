import os
import requests
import json

# Local API base URL
LOCAL_API_BASE = "http://localhost:8000"

def get_latest_classification_file(directory: str) -> str:
    """
    Lists the files in the specified directory and returns the full path of the latest file (by modification time).
    """
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return ""
    
    files = os.listdir(directory)
    if not files:
        print("No classification files found in the directory.")
        return ""
    
    # Sort files by modification time descending
    files_sorted = sorted(files, key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
    latest_file = os.path.join(directory, files_sorted[0])
    return latest_file

def generate_final_report(filename: str) -> dict:
    """
    Calls the /api/generate_report endpoint with the provided classification filename.
    """
    url = f"{LOCAL_API_BASE}/api/generate_report"
    params = {"filename": filename}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def main():
    classification_directory = "output/classification"
    latest_file = get_latest_classification_file(classification_directory)
    
    if not latest_file:
        print("No classification file available. Please run the classification phase first.")
        return

    print("Using classification file:", latest_file)
    
    try:
        report_data = generate_final_report(latest_file)
        print("Final Report:")
        print(json.dumps(report_data, indent=2))
    except Exception as e:
        print("Error generating final report:", str(e))

if __name__ == "__main__":
    main()
