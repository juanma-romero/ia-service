"""
agent.py — Router del Agente IA Autónomo de Voraz
================================================
Expone el endpoint POST /agent-query con bucle de razonamiento multi-paso (ReAct).

Flujo:
  1. Recibe la consulta del admin en lenguaje natural.
  2. Inyecta contexto temporal dinámico (hora Paraguay UTC-3) y Manual de Datos ERPNext.
  3. Ejecuta un bucle de razonamiento (hasta 5 iteraciones):
     - El LLM decide si necesita datos (tools de ERPNext, cálculos matemáticos, etc.).
     - Python ejecuta las tools solicitadas y devuelve los resultados.
     - El LLM evalúa los nuevos datos y decide si necesita más información o formula la respuesta.
  4. Devuelve la respuesta final formateada para WhatsApp.
"""

import os
import json
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv

from services.agent_tools import execute_tool, TOOL_DEFINITIONS
from services.mcp_client import mcp_manager

load_dotenv()

router = APIRouter()

PRIMARY_MODEL = os.getenv("AGENT_MODEL", "openai/gpt-5")
FALLBACK_MODEL = os.getenv("AGENT_FALLBACK_MODEL", "anthropic/claude-sonnet-4.5")
MAX_ITERATIONS = 5

SYSTEM_PROMPT_TEMPLATE = """Sos el asistente de gestión inteligente de "Voraz", un comercio gastronómico de bocaditos y empanadas en Ciudad del Este, Paraguay.
Tu función es responder con total precisión consultas del administrador sobre ventas, pedidos, stock, clientes y operaciones del negocio.

{temporal_context}

══════════════════════════════════════════════════════════════════════════
REGLAS GENERALES DE RESPUESTA
══════════════════════════════════════════════════════════════════════════
- Respondé siempre en español, de forma concisa, ejecutiva y directa.
- Usá formato WhatsApp: *negrita* para títulos, números y datos destacados.
- Moneda: Guaraníes (₲). Formateá SIEMPRE los montos con puntos de miles (ej: ₲ 1.250.000).
- Si la consulta pide cantidades de productos (ej: "cantidades dia hoy", "productos entregados"), responde con un formato simple:
  *Ventas por producto:*
  • 4 Combo Premium
  • 1 Coca Cola
- No inventes datos. Si una consulta no arrojó resultados o la tool falló, informalo claramente.
- Razoná paso a paso antes de responder: si te falta un dato intermedio (ej: ID de cliente o lista de ítems), utilizá las herramientas necesarias para averiguarlo.

══════════════════════════════════════════════════════════════════════════
MANUAL DE DATOS Y ESTRUCTURA DE ERPNEXT (VORAZ)
══════════════════════════════════════════════════════════════════════════
En el flujo comercial de Voraz:
1. Primero se emite el **Sales Order** (Pedido de Venta).
2. Posteriormente se factura con **Sales Invoice** (Factura de Venta).

DocTypes y campos clave en ERPNext:
• `Sales Order` (Pedidos de Venta):
  - Campos: `name` (ID del pedido), `customer` (ID cliente), `customer_name`, `transaction_date` (fecha pedido), `delivery_date` (fecha entrega), `custom_dia_y_hora_entrega` (campo personalizado clave con la fecha y hora exacta programada de entrega), `grand_total`, `status` ('Draft', 'To Deliver and Bill', 'Completed', 'Cancelled'), `docstatus` (0=Borrador, 1=Confirmado, 2=Cancelado).
  - Ítems del pedido (tabla hija): `items` con `item_code`, `item_name`, `qty`, `rate`, `amount`.

• `Sales Invoice` (Facturas de Venta):
  - Campos: `name`, `customer`, `customer_name`, `posting_date` (fecha factura), `grand_total`, `outstanding_amount` (saldo pendiente de cobro), `status` ('Paid', 'Unpaid', 'Overdue', 'Cancelled'), `docstatus` (1=Confirmada).

• `Customer` (Clientes):
  - Campos: `name`, `customer_name`, `mobile_no`, `territory`, `customer_group`.

• `Item` (Productos / Combos / Empanadas):
  - Campos: `item_code`, `item_name`, `item_group`, `standard_rate`, `description`.

• `Bin` (Stock Actual por Depósito):
  - Campos: `item_code`, `warehouse`, `actual_qty`.

• `Payment Entry` (Recibos de Cobro / Pagos):
  - Campos: `posting_date`, `party` (cliente), `paid_amount`, `mode_of_payment` ('Efectivo', 'Transferencia', etc.).

══════════════════════════════════════════════════════════════════════════
GUÍA DE HERRAMIENTAS (TOOLS)
══════════════════════════════════════════════════════════════════════════
1. **Atajos rápidos de Voraz:**
   - `get_sales_summary`: Resumen de ventas rápidas para períodos: 'hoy', 'semana', 'mes', 'mes_pasado', 'anio'.
   - `get_sales_by_product`: Resumen por producto para los períodos anteriores.
   - `get_pending_orders`: Lista de pedidos pendientes de entrega.

2. **Consultas libres a ERPNext (MCP):**
   - `get_documents`: Para buscar y listar documentos.
     * Siempre especifica `fields` relevantes para no sobrecargar la respuesta (ej: `["name", "customer_name", "grand_total", "transaction_date"]`).
     * Usa `filters` en formato de objeto o lista (ej: `{{"docstatus": 1, "transaction_date": [">=", "YYYY-MM-DD"]}}`).
     * Usa `limit_page_length` (ej: 10 o 20) cuando no necesites todos los registros.
   - `get_document`: Úsalo SOLO cuando ya tengas el `name` / ID exacto del documento para ver su detalle completo (incluyendo ítems).
   - `get_doctype_fields`: Si necesitas conocer campos de un DocType que no recuerdes.

3. **Cálculos matemáticos:**
   - `calculate`: Usá esta tool para cualquier suma, resta, promedio o porcentaje exacto (ej: `calculate(expression="150000 * 4 + 75000")`).
"""


def _get_temporal_context() -> str:
    """Genera el texto con la fecha, hora y día actual en zona horaria de Paraguay (UTC-3)."""
    tz_py = timezone(timedelta(hours=-3))
    now = datetime.now(tz_py)

    dias_semana = {
        0: "Lunes", 1: "Martes", 2: "Miércoles",
        3: "Jueves", 4: "Viernes", 5: "Sábado", 6: "Domingo"
    }
    meses = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
        5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
        9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
    }

    dia_nombre = dias_semana[now.weekday()]
    mes_nombre = meses[now.month]
    fecha_iso = now.strftime("%Y-%m-%d")
    hora_str = now.strftime("%H:%M")

    return (
        f"══════════════════════════════════════════════════════════════════════════\n"
        f"CONTEXTO TEMPORAL ACTUAL (PARAGUAY UTC-3):\n"
        f"• Hoy es: {dia_nombre}, {now.day} de {mes_nombre} de {now.year} ({fecha_iso}).\n"
        f"• Hora actual: {hora_str} hs.\n"
        f"• Mes en curso: {mes_nombre} {now.year}.\n"
        f"══════════════════════════════════════════════════════════════════════════"
    )


class AgentQueryRequest(BaseModel):
    query: str


@router.post("/agent-query")
async def agent_query(request: AgentQueryRequest):
    """
    Endpoint principal del agente IA con razonamiento multi-paso (ReAct).
    """
    openrouter_api_key = os.getenv("OPEN_ROUTER_API_KEY")
    if not openrouter_api_key:
        raise HTTPException(status_code=500, detail="OPEN_ROUTER_API_KEY no configurada.")

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
    )

    temporal_context = _get_temporal_context()
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(temporal_context=temporal_context)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.query}
    ]

    print(f"[Agent] Nueva consulta recibida: '{request.query}'")

    try:
        # Obtener las tools disponibles (Python nativas + MCP ERPNext)
        mcp_tools = await mcp_manager.get_tools_schema()
        all_tools = TOOL_DEFINITIONS + mcp_tools
        mcp_tool_names = [t["function"]["name"] for t in mcp_tools]

        current_model = PRIMARY_MODEL

        # ── Bucle de Razonamiento Multi-paso (ReAct Loop) ───────────────────
        for step in range(1, MAX_ITERATIONS + 1):
            print(f"[Agent] [Paso {step}/{MAX_ITERATIONS}] Consultando modelo '{current_model}'...")

            try:
                response = await client.chat.completions.create(
                    model=current_model,
                    messages=messages,
                    tools=all_tools,
                    tool_choice="auto",
                    max_tokens=1500,
                )
            except Exception as model_err:
                print(f"[Agent] Error con modelo '{current_model}': {model_err}")
                if current_model != FALLBACK_MODEL:
                    print(f"[Agent] Reintentando con modelo fallback '{FALLBACK_MODEL}'...")
                    current_model = FALLBACK_MODEL
                    response = await client.chat.completions.create(
                        model=current_model,
                        messages=messages,
                        tools=all_tools,
                        tool_choice="auto",
                        max_tokens=1500,
                    )
                else:
                    raise model_err

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            # Si el modelo no pide ninguna tool, ya tiene la respuesta final
            if not tool_calls:
                answer = response_message.content or "No se pudo formular una respuesta."
                print(f"[Agent] Respuesta final lista en paso {step}: {answer[:100]}...")
                return {"response": answer}

            # Si el modelo decidió llamar a una o más herramientas:
            messages.append(response_message)

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                print(f"[Agent] Tool call: {tool_name}({tool_args})")

                # Ejecutar según sea tool de MCP o nativa
                try:
                    if tool_name in mcp_tool_names:
                        tool_result = await mcp_manager.call_tool(tool_name, tool_args)
                    else:
                        tool_result = await execute_tool(tool_name, tool_args)
                except Exception as exec_err:
                    tool_result = {"error": f"Fallo al ejecutar tool '{tool_name}': {str(exec_err)}"}

                # Añadir el resultado de la tool al historial de mensajes
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result, ensure_ascii=False, default=str)
                })

        # ── Si se alcanza el límite máximo de iteraciones, forzar respuesta final ──
        print(f"[Agent] Límite de {MAX_ITERATIONS} iteraciones alcanzado. Forzando respuesta final...")
        final_response = await client.chat.completions.create(
            model=current_model,
            messages=messages,
            max_tokens=1500,
        )
        answer = final_response.choices[0].message.content or "Consulta procesada con datos parciales."
        return {"response": answer}

    except Exception as e:
        print(f"[Agent] Error general en el agente: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la consulta con el agente: {str(e)}"
        )
