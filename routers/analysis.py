import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services import ai_model # Importar el servicio que crearemos
 
router = APIRouter()

# Modelo Pydantic para validar los datos de entrada
class PromptRequest(BaseModel):
    prompt: str

@router.post("/analyze-conversation")
async def analyze_conversation(request: PromptRequest):
    """
    Recibe un historial de conversación, lo analiza con Gemini
    y devuelve el estado y un resumen en formato JSON.
    """
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
        response_text = await ai_model.generate_content_with_prompt(system_prompt)
        
        # Parsear la respuesta
        cleaned_text = response_text.strip().replace("```json", "").replace("```", "")
        json_response = json.loads(cleaned_text)
        
        return json_response

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"La respuesta de la IA no pudo ser parseada como JSON: {response_text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al contactar a Google AI: {str(e)}")
