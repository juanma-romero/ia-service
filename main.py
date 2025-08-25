import os
from fastapi import FastAPI
from dotenv import load_dotenv
 
# Cargar variables de entorno
load_dotenv()

# Importar los routers que contienen los endpoints
from routers import analysis, orders

# Crear la aplicación FastAPI
app = FastAPI(
    title="Servicio de IA para Asistente Voraz",
    description="Analiza conversaciones de WhatsApp y devuelve estado y resumen."
)

# Incluir los routers
app.include_router(analysis.router, tags=["Analysis"])
app.include_router(orders.router, tags=["Orders"])

# Endpoint de "salud"
@app.get("/", tags=["Status"])
def read_root():
    return {"status": "IA Service is running"}
