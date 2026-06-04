import os
from fastapi import FastAPI
from dotenv import load_dotenv
 
# Cargar variables de entorno
load_dotenv()

# Importar los routers que contienen los endpoints
from routers import analysis, orders, agent

from contextlib import asynccontextmanager
from services.mcp_client import mcp_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Iniciando dependencias (MCP)...")
    await mcp_manager.start()
    yield
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
