import os
from google import genai
from fastapi import HTTPException
from dotenv import load_dotenv
 
load_dotenv()

# Configurar la API de Google
try:
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    print(f"Error al inicializar el cliente de Google GenAI: {e}")
    client = None

async def generate_content_with_prompt(prompt: str):
    if not client:
        raise HTTPException(status_code=500, detail="El cliente de Google AI no está inicializado.")

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al contactar a Google AI: {str(e)}")
