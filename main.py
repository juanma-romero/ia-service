# ia-service/main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
# Esto es crucial para mantener tu API Key segura y fuera del código.
load_dotenv()

# Configurar la API de Google
# Asegúrate de que la variable GOOGLE_API_KEY esté en tu archivo .env
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


# Inicializar la aplicación FastAPI
app = FastAPI()

# Modelo Pydantic para validar los datos de entrada de la solicitud
class PromptRequest(BaseModel):
    prompt: str

# Endpoint de "salud" para verificar que el servicio está vivo
@app.get("/")
def read_root():
    return {"status": "IA Service is running"}

# Endpoint principal para hacer preguntas a la IA de Google
@app.post("/ask-google-ai")
async def ask_google(request: PromptRequest):
    """
    Recibe un prompt y devuelve la respuesta del modelo Gemini.
    """
    try:
        print(f"Recibido prompt para Google AI: '{request.prompt}'")
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=request.prompt,
        )                
        print(f"Respuesta de Google AI: '{response.text}'")
        
        # Devolver la respuesta en un formato JSON claro
        return {"answer": response.text}

    except Exception as e:
        print(f"Error al contactar a Google AI: {e}")
        # Si algo sale mal, devolvemos un error HTTP claro.
        raise HTTPException(status_code=500, detail=f"Error interno del servidor de IA: {str(e)}")

