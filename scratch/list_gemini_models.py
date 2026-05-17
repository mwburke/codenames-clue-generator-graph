import os
import google.generativeai as genai

def list_models():
    # Attempt to read the key from the generator's state or env
    key = os.getenv("GOOGLE_API_KEY") # User's current env key
    if not key:
        print("No API key found in env")
        return
    
    genai.configure(api_key=key)
    print(f"Listing models for key starting with {key[:10]}...")
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)
    except Exception as e:
        print(f"Failed to list models: {e}")

if __name__ == "__main__":
    list_models()
