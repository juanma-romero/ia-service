"""
agent.py — Router del Agente IA de Voraz
=========================================
Expone el endpoint POST /agent-query.

Flujo:
  1. El admin envía una consulta en lenguaje natural.
  2. El LLM (Groq llama-3.3-70b-versatile) recibe la query + lista de tools disponibles.
  3. Si el LLM decide usar una tool, Python la ejecuta de forma controlada.
  4. El LLM recibe el resultado y formula una respuesta en lenguaje natural para WhatsApp.
"""

import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

from services.agent_tools import execute_tool, TOOL_DEFINITIONS
from services.mcp_client import mcp_manager

load_dotenv()

router = APIRouter()

SYSTEM_PROMPT = """Sos el asistente de gestión de "Voraz", un comercio de bocaditos y empanadas en Ciudad del Este, Paraguay.
Tu función es responder consultas del administrador sobre ventas, pedidos y operaciones del negocio.

Reglas para responder:
- Respondé siempre en español, de forma concisa y directa.
- Usá formato WhatsApp: *negrita* para títulos, números y datos importantes.
- Los montos son en guaraníes (₲). Formateá los números con puntos como separador de miles (ej: ₲ 1.250.000).
- Si la consulta solicita cantidades de productos o ventas de productos por día (ej: "cantidades dia hoy", "productos entregados", etc.), responde con un formato simple y limpio que liste únicamente las cantidades y los nombres de los productos, por ejemplo:
  *Ventas por producto de dia:*
  4 Combo Premium
  1 Coca Cola
- Si no tenés los datos para responder, decilo claramente y sugerí qué consulta puede hacer el admin.
- No inventes datos. Si la tool devolvió un error, informalo con un mensaje claro.
- Sé breve: el admin lee en WhatsApp, no en una pantalla grande.

Reglas para el uso de ERPNext:
- Si el usuario pide listar o buscar varios documentos (ej: facturas, clientes, pedidos), usa SIEMPRE la herramienta `get_documents`. Podes usar los parámetros `limit_page_length` para limitar (ej: 5).
- Usa `get_document` (singular) SOLO si ya conoces el ID o `name` exacto del documento. NUNCA adivines nombres como "1", "2".
- El doctype para facturas de ventas es "Sales Invoice". Para clientes es "Customer". Para artículos es "Item".
"""

class AgentQueryRequest(BaseModel):
    query: str


@router.post("/agent-query")
async def agent_query(request: AgentQueryRequest):
    """
    Endpoint principal del agente. Recibe una consulta en lenguaje natural
    del administrador y devuelve una respuesta formateada para WhatsApp.
    """
    openrouter_api_key = os.getenv("OPEN_ROUTER_API_KEY")
    if not openrouter_api_key:
        raise HTTPException(status_code=500, detail="OPEN_ROUTER_API_KEY no configurada.")

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
    )
    model = "anthropic/claude-3.5-haiku"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": request.query}
    ]

    print(f"[Agent] Query recibida: '{request.query}'")

    try:
        # Obtener las tools del MCP y combinarlas
        mcp_tools = await mcp_manager.get_tools_schema()
        all_tools = TOOL_DEFINITIONS + mcp_tools

        # ── Paso 1: Primera llamada al LLM — decide si usa una tool ──────────
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=all_tools,
            tool_choice="auto",
            max_tokens=1024,
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # ── Paso 2: Si el LLM quiere usar una tool, ejecutarla ───────────────
        if tool_calls:
            # Agregamos la respuesta del LLM al historial de mensajes
            messages.append(response_message)

            # Procesamos cada tool_call (generalmente es solo 1 en Fase 1)
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                print(f"[Agent] LLM eligió tool='{tool_name}' args={tool_args}")

                mcp_tool_names = [t["function"]["name"] for t in mcp_tools]
                if tool_name in mcp_tool_names:
                    # Ejecutar tool del servidor MCP
                    tool_result = await mcp_manager.call_tool(tool_name, tool_args)
                else:
                    # Ejecutar la función Python nativa
                    tool_result = await execute_tool(tool_name, tool_args)

                # Agregar el resultado de la tool al historial
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result, ensure_ascii=False, default=str)
                })

            # ── Paso 3: Segunda llamada al LLM — formula la respuesta final ──
            final_response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1024,
            )

            answer = final_response.choices[0].message.content
            print(f"[Agent] Respuesta generada (con tool): {answer[:100]}...")
            return {"response": answer}

        # ── Sin tool: el LLM respondió directo (saludo, consulta simple, etc.) ─
        answer = response_message.content
        print(f"[Agent] Respuesta directa (sin tool): {answer[:100]}...")
        return {"response": answer}

    except Exception as e:
        print(f"[Agent] Error en el agente: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la consulta con el agente: {str(e)}"
        )
