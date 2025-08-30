import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services import ai_model

router = APIRouter()

class PromptRequest(BaseModel):
    prompt: str

@router.post("/analyze-order")
async def analyze_order(request: PromptRequest):
    """
    Recibe un historial de conversación, extrae los detalles de un pedido
    y devuelve un objeto JSON con la información estructurada.
    """
    system_prompt = f"""
    Eres un asistente experto para "Voraz", un negocio de bocaditos y empanadas.
    Tu única tarea es analizar el siguiente historial de una conversación de WhatsApp
    y extraer la información de un pedido que ha sido confirmado por el administrador
    con la frase "Entonces te agendo:".
    - Si no encuentras la frase o el formato no es claro, no extraigas nada y devuelve una respuesta vacía o con un indicador de "no_pedido".
    - Si la detectas, debes extraer la siguiente información:
    1.  **Día y Hora:** Teniendo en cuenta la fecha y hora actual suministrada junto con el historial de mensajes, analiza la conversación para determinar la fecha y hora de entrega deseadas. Tu tarea es convertir términos relativos como "hoy", "mañana" o "el jueves" en una fecha absoluta y estructurada. El resultado final DEBE ESTAR en el formato estricto del tipo Date de MongoDB.
    2.  **Productos y Cantidades:** Identifica los productos y la cantidad solicitada para cada uno. Ejemplo: "1 combo Premium".
    3.  **Monto Total:** Extrae el monto total del pedido. Ejemplo: "335 mil gs".

    **Historial de la Conversación:**
    ---
    {request.prompt}
    ---

    **Formato de Respuesta Obligatorio:**
    - Debes responder ÚNICAMENTE con un objeto JSON válido.
    - Si se encontraron los datos, el formato exacto debe ser:

  ```json
  {{
    "pedido_detectado": true,
    "fecha_hora_entrega": "...",
    "productos": [
      {{
        "nombre": "...",
        "cantidad": "..."
      }}
    ],
    "monto_total": "..."
  }} 
    """

    try:
        response_text = await ai_model.generate_content_with_prompt(system_prompt)
        
        cleaned_text = response_text.strip().replace("```json", "").replace("```", "")
        json_response = json.loads(cleaned_text)
        
        return json_response

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"La respuesta de la IA no pudo ser parseada como JSON: {response_text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al contactar a Google AI: {str(e)}")
