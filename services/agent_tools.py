"""
agent_tools.py
==============
Define las "tools" disponibles para el agente IA de Voraz.
Cada tool tiene:
  - TOOL_DEFINITIONS: la especificación que se le envía al LLM (schema JSON)
  - execute_tool(): el dispatcher que llama a la función Python real

El LLM NUNCA ejecuta código — solo elige el nombre de la tool y los parámetros.
Este módulo ejecuta la lógica real de forma controlada y auditada.
"""

import os
import json
import httpx
from datetime import datetime, timedelta, timezone
from typing import Any

# URL del erp-service (misma que usa el backend)
ERP_SERVICE_URL = os.getenv("ERP_SERVICE_URL", "http://localhost:8001")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS DE FECHAS
# ─────────────────────────────────────────────────────────────────────────────

def _get_period_dates(period: str) -> tuple[str, str]:
    """
    Convierte un período textual en (fecha_desde, fecha_hasta) en formato YYYY-MM-DD.
    Usa la zona horaria de Paraguay (UTC-3).
    """
    tz = timezone(timedelta(hours=-3))
    now = datetime.now(tz)

    if period == "hoy":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = now
    elif period == "semana":
        start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        end = now
    elif period == "mes":
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end = now
    elif period == "mes_pasado":
        first_of_this_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end = first_of_this_month - timedelta(seconds=1)
        start = end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    elif period == "anio":
        start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        end = now
    else:
        # Default: hoy
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = now

    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES DE TOOLS (lógica real)
# ─────────────────────────────────────────────────────────────────────────────

async def _get_sales_summary(period: str) -> dict:
    """Consulta el resumen de ventas del período al erp-service."""
    date_from, date_to = _get_period_dates(period)
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(
            f"{ERP_SERVICE_URL}/api/sales/summary",
            params={"date_from": date_from, "date_to": date_to}
        )
        response.raise_for_status()
        return response.json()


async def _get_sales_by_product(period: str) -> dict:
    """Consulta ventas agrupadas por producto al erp-service."""
    date_from, date_to = _get_period_dates(period)
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(
            f"{ERP_SERVICE_URL}/api/sales/by-product",
            params={"date_from": date_from, "date_to": date_to}
        )
        response.raise_for_status()
        return response.json()


async def _get_pending_orders() -> dict:
    """Consulta los pedidos pendientes de entrega (reutiliza endpoint existente)."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(f"{ERP_SERVICE_URL}/api/orders/pending")
        response.raise_for_status()
        orders = response.json()
        return {
            "total_pendientes": len(orders),
            "pedidos": orders[:10]  # Limitar a 10 para no saturar el contexto del LLM
        }


# ─────────────────────────────────────────────────────────────────────────────
# DISPATCHER PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

async def execute_tool(tool_name: str, args: dict) -> Any:
    """
    Ejecuta la tool correspondiente y devuelve el resultado.
    Esto es lo único que el LLM puede "activar" — nunca ejecuta código directamente.
    """
    print(f"[AgentTools] Ejecutando tool='{tool_name}' con args={args}")

    try:
        if tool_name == "get_sales_summary":
            period = args.get("period", "mes")
            return await _get_sales_summary(period)

        elif tool_name == "get_sales_by_product":
            period = args.get("period", "mes")
            return await _get_sales_by_product(period)

        elif tool_name == "get_pending_orders":
            return await _get_pending_orders()

        else:
            return {"error": f"Tool '{tool_name}' no reconocida."}

    except httpx.HTTPStatusError as e:
        return {"error": f"Error HTTP al consultar datos: {e.response.status_code} - {e.response.text[:200]}"}
    except Exception as e:
        return {"error": f"Error al ejecutar la tool: {str(e)}"}


# ─────────────────────────────────────────────────────────────────────────────
# DEFINICIONES DE TOOLS PARA EL LLM (schema JSON — Groq/OpenAI format)
# ─────────────────────────────────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_sales_summary",
            "description": (
                "Obtiene el resumen de ventas para un período determinado: "
                "total en dinero, cantidad de pedidos. "
                "Usá esta tool cuando pregunten por ventas totales, ingresos, recaudación, "
                "cuánto se vendió, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "enum": ["hoy", "semana", "mes", "mes_pasado", "anio"],
                        "description": (
                            "El período de tiempo. "
                            "'hoy' = desde medianoche hasta ahora. "
                            "'semana' = esta semana (lunes a hoy). "
                            "'mes' = desde el 1 del mes actual. "
                            "'mes_pasado' = el mes calendario anterior completo. "
                            "'anio' = desde el 1 de enero."
                        )
                    }
                },
                "required": ["period"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_sales_by_product",
            "description": (
                "Obtiene el detalle de ventas desagregado por producto: "
                "qué productos se vendieron más, cantidades y montos por ítem. "
                "Usá esta tool cuando pregunten por productos más vendidos, "
                "cantidad de combos vendidos, ventas por producto, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "enum": ["hoy", "semana", "mes", "mes_pasado", "anio"],
                        "description": "El período de tiempo a consultar."
                    }
                },
                "required": ["period"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_pending_orders",
            "description": (
                "Lista los pedidos que están pendientes de entrega en este momento. "
                "Usá esta tool cuando pregunten cuántos pedidos hay pendientes, "
                "qué pedidos faltan entregar, cuál es la carga de trabajo actual."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]
