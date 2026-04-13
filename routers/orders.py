import json
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services import ai_model

router = APIRouter()

class PromptRequest(BaseModel):
    prompt: str
    
def get_catalog():
    try:
        current_dir = os.path.dirname(os.path.realpath(__file__))
        catalog_path = os.path.join(current_dir, '..', 'catalog.json')
        with open(catalog_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print("Aviso: No se pudo cargar catalog.json:", e)
        return []

@router.post("/analyze-order")
async def analyze_order(request: PromptRequest):
    """
    Analiza una solicitud de agendamiento estructurada hecha por un administrador,
    mapeando informalidades ("hamburguesitas", "el de 198") a los item_code exactos de ERPNext.
    """
    catalog = get_catalog()
    catalog_str = json.dumps(catalog, ensure_ascii=False) if catalog else "Catálogo no disponible."
    
    system_prompt = f"""
    Eres un asistente experto para "Voraz". Tu única tarea es extraer información precisa 
    de un resumen de administrador de WhatsApp que comienza con "Entonces te agendo:".
    
    Esta información viaja a un sistema ERP, por lo que NO debes adivinar productos. **DEBES obligatoriamente** mapear lo que pide el admin al `item_code` exacto listado en tu catálogo. Si no tiene sentido, repórtalo en tu cabeza pero usa el código más cercano válido.

    --- CATÁLOGO DE REFERENCIA (ERPNext) ---
    {catalog_str}
    -----------------------------------------

    **Instrucciones de extracción:**
    1. **Día y Hora:** Analiza el texto para deducir la fecha y hora de entrega. Ejemplo: "mañana 10 am" + la fecha actual proveída en el texto = Date ISO. Formato obligatorio: `YYYY-MM-DD HH:MM:00`.
    2. **Monto Total:** Extrae el monto total del pedido (como un número, elimina símbolos o 'gs').
    3. **Productos:** Para cada producto mencionado en el chat, extrae su cantidad, y OBLIGATORIAMENTE combínalo con un `item_code` válido de la lista anterior, guiándote por el nombre sugerido (description). Ignora los precios acá ya que el total va abarcativo.

    Si es un delivery/envío, y no hay item code de "delivery" explícito en tu catálogo, omítelo o usa un genérico si existe.

    **Resumen del Administrador:**
    ---
    {request.prompt}
    ---

    **Formato de Respuesta Obligatorio:**
    Debes responder ÚNICAMENTE con un objeto JSON (NUNCA agregues markdown como ```json).
    Formato:
    {{
        "pedido_detectado": true,
        "fecha_hora_entrega": "2026-04-12 10:00:00",
        "monto_total": 223000,
        "productos": [
            {{ "item_code": "pre", "cantidad": 1 }}
        ]
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
