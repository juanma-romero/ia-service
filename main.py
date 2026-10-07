import os
from fastapi import FastAPI
from dotenv import load_dotenv
 
# Cargar variables de entorno
load_dotenv()

# Importar los routers que contienen los endpoints
from routers import analysis, orders, agent

from contextlib import asynccontextmanager
from services.mcp_client import mcp_manager

# El MCP de ERPNext arranca un servidor Node y necesita un ERP real.
# En local se apaga con ENABLE_MCP=false; en producción no se define, así que
# el valor por defecto mantiene el comportamiento actual (encendido).
ENABLE_MCP = os.getenv("ENABLE_MCP", "true").strip().lower() not in ("0", "false", "no", "off")

@asynccontextmanager
async def lifespan(app: FastAPI):
    if ENABLE_MCP:
        print("Iniciando dependencias (MCP)...")
        await mcp_manager.start()
    else:
        print("[MCP] Deshabilitado (ENABLE_MCP=false): el agente funcionará sin herramientas de ERP.")
    yield
    if ENABLE_MCP:
        print("Apagando dependencias (MCP)...")
        await mcp_manager.stop()

# Crear la aplicación FastAPI
app = FastAPI(
    title="Servicio de IA para Asistente Voraz",
    description="Analiza conversaciones de WhatsApp y devuelve estado y resumen.",
    lifespan=lifespan
)

# Incluir los routers
app.include_router(analysis.router, tags=["Analysis"])
app.include_router(orders.router, tags=["Orders"])
app.include_router(agent.router, tags=["Agent"])

# Endpoint de "salud"
@app.get("/", tags=["Status"])
def read_root():
    return {"status": "IA Service is running"}
