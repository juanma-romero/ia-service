import os
from google import genai
from groq import Groq
from fastapi import HTTPException
from dotenv import load_dotenv
 
load_dotenv()

# Configurar la API de Google (Fallback)
try:
    client_google = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    print(f"Error al inicializar el cliente de Google GenAI: {e}")
    client_google = None

# Configurar la API de Groq (Principal)
try:
    client_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    print(f"Error al inicializar el cliente de Groq: {e}")
    client_groq = None

async def generate_content_with_prompt(prompt: str):
    # Intentar PRIMERO con Groq (openai/gpt-oss-120b)
    if client_groq:
        try:
            print("[ia-service] Consultando a Groq (openai/gpt-oss-120b)...")
            chat_completion = client_groq.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model="openai/gpt-oss-120b",
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"[ia-service] Error o Timeout en Groq API: {e}. Activando Fallback a Gemini...")
    
    # FALLBACK a Google Gemini
    if not client_google:
        raise HTTPException(status_code=500, detail="Ni Groq ni Google AI están inicializados para procesar la petición.")

    try:
        print("[ia-service] Consultando a Google Gemini (gemini-flash-lite-latest   ) como salvavidas...")
        response = client_google.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fallo masivo de IA (Ambos proveedores inaccesibles): {str(e)}")
