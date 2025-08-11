import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configurar la API de Google
# Asegúrate de que la variable GOOGLE_API_KEY esté en tu archivo .env
try:
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    # Manejo de error si la API key no es válida o falta
    print(f"Error al inicializar el cliente de Google GenAI: {e}")
    client = None

# Inicializar la aplicación FastAPI
app = FastAPI(
    title="Servicio de IA para Asistente Voraz",
    description="Analiza conversaciones de WhatsApp y devuelve estado y resumen."
)

# Modelo Pydantic para validar los datos de entrada
class PromptRequest(BaseModel):
    prompt: str

# Endpoint de "salud"
@app.get("/", tags=["Status"])
def read_root():
    return {"status": "IA Service is running"}

@app.post("/analyze-conversation", tags=["Analysis"])
async def analyze_conversation(request: PromptRequest):
    """
    Recibe un historial de conversación, lo analiza con Gemini
    y devuelve el estado y un resumen en formato JSON.
    """
    if not client:
        raise HTTPException(status_code=500, detail="El cliente de Google AI no está inicializado. Verifica la API Key.")

    # El prompt del sistema es la clave. Le damos el contexto, las reglas y el formato de salida.
    system_prompt = f"""
    Eres un asistente experto para "Voraz", un negocio de bocaditos y empanadas.
    Tu tarea es analizar el siguiente historial de una conversación de WhatsApp y determinar dos cosas:
    1.  El "estado" actual de la conversación desde la perspectiva del administrador del negocio.
    2.  Un "resumen" contextual muy breve (máximo 15 palabras) que describa lo último relevante o la acción pendiente.

    **Historial de la Conversación:**
    ---
    {request.prompt}
    ---

    **Reglas para determinar el estado:**
    - "Requiere Acción del Admin": Usa este estado si el cliente hizo una pregunta, envió información que necesita ser confirmada, o está esperando una acción del negocio.
    - "Esperando Respuesta del Cliente": Usa este estado si la última acción fue del administrador haciendo una pregunta directa al cliente.
    - "Resuelto": Usa este estado si el pedido fue entregado, pagado y la conversación ha concluido amigablemente.
    - "Estancado/Inactivo": Usa este estado si el administrador preguntó algo pero el cliente no ha respondido en un tiempo considerable (basado en los timestamps).
    - "Sin Contestar": Usa este estado ÚNICAMENTE si la conversación consiste en un único mensaje inicial del cliente que aún no ha sido respondido por el administrador.

    **Formato de Respuesta Obligatorio:**
    Debes responder ÚNICAMENTE con un objeto JSON válido, sin texto adicional antes o después.
    El formato exacto debe ser:
    {{
      "state": "...",
      "summary": "..."
    }}
    """

    try:
        print("Enviando prompt a Gemini para análisis...")
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=system_prompt,
        )
        
        # Intentamos parsear la respuesta de la IA para asegurar que es un JSON válido
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
        print(f"Respuesta recibida de Gemini: {cleaned_text}")
        
        json_response = json.loads(cleaned_text)
        
        return json_response

    except json.JSONDecodeError as e:
        print(f"Error: La respuesta de la IA no es un JSON válido. Respuesta: {response.text}")
        raise HTTPException(status_code=500, detail=f"La respuesta de la IA no pudo ser parseada como JSON: {response.text}")
    except Exception as e:
        print(f"Error al contactar a Google AI: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor de IA: {str(e)}")