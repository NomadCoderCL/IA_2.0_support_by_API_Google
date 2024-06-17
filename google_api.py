import os
import google.generativeai as genai
from decrypt_util import decrypt_key

def predict_google_api(input_text):
    api_key_path = os.path.join(os.path.dirname(__file__), 'encrypted_api_key.txt')
    if os.path.exists(api_key_path):
        with open(api_key_path, 'rb') as key_file:
            encrypted_key = key_file.read()
            api_key = decrypt_key(encrypted_key)
            configure_api_key(api_key)
    else:
        raise FileNotFoundError(f"No se encontró el archivo {api_key_path}")
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config=generation_config,
    )
    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [input_text],
            }
        ]
    )
    response = chat_session.send_message(input_text)
    return response.text

def configure_api_key(api_key):
    genai.configure(api_key=api_key)
