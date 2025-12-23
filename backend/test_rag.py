import requests
import os

API_URL = "http://127.0.0.1:8000"

def test_api():
    print("1. Testing Health...")
    try:
        r = requests.get(f"{API_URL}/")
        print(f"   Status: {r.status_code}")
        print(f"   Response: {r.json()}")
    except Exception as e:
        print(f"   FAILED to connect: {e}")
        return

    print("\n2. Testing Upload (doc1.md)...")
    file_path = r"D:\job\indecimal\doc1.md"
    if not os.path.exists(file_path):
        print(f"   File not found: {file_path}")
        return
        
    try:
        files = {'file': open(file_path, 'rb')}
        r = requests.post(f"{API_URL}/api/v1/upload", files=files)
        print(f"   Status: {r.status_code}")
        print(f"   Response: {r.json()}")
    except Exception as e:
        print(f"   Upload FAILED: {e}")

    print("\n3. Testing Query (Groq)...")
    try:
        payload = {"question": "What is Indecimal? Answer neatly.", "top_k": 3}
        r = requests.post(f"{API_URL}/api/v1/query", json=payload)
        print(f"   Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"   Answer Preview: {data['answer'][:200]}...")
            print(f"   Confidence: {data['confidence']}")
        else:
            print(f"   Error: {r.text}")
    except Exception as e:
        print(f"   Query FAILED: {e}")

if __name__ == "__main__":
    test_api()
