import asyncio
import os
import sys

# Añadir el directorio actual al path para que los imports locales funcionen
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.mcp_client import mcp_manager

async def test():
    print("Iniciando test de MCP...")
    await mcp_manager.start()
    tools = await mcp_manager.get_tools_schema()
    for t in tools:
        if t['function']['name'] == 'get_documents':
            import json
            print("Schema de get_documents:")
            print(json.dumps(t, indent=2))
    await mcp_manager.stop()
    print("Test finalizado.")

if __name__ == "__main__":
    asyncio.run(test())
