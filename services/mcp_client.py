import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

# Asegurarse de cargar las variables de entorno de erp-service donde están las credenciales
load_dotenv(os.path.join(os.path.dirname(__file__), "../../erp-service/.env"))

# Path al index.js del MCP de ERPNext
MCP_SERVER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../erpnext-mcp-server/build/index.js"))

class MCPManager:
    def __init__(self):
        self.session = None
        self._client = None
        self._process_context = None

    async def start(self):
        env = os.environ.copy()
        
        server_params = StdioServerParameters(
            command="node",
            args=[MCP_SERVER_PATH],
            env=env
        )
        
        print(f"[MCP] Iniciando MCP Server en {MCP_SERVER_PATH}")
        self._process_context = stdio_client(server_params)
        read, write = await self._process_context.__aenter__()
        
        self._client = ClientSession(read, write)
        self.session = await self._client.__aenter__()
        
        await self.session.initialize()
        print("[MCP] Conectado exitosamente al ERPNext MCP Server")

    async def stop(self):
        if self._client:
            await self._client.__aexit__(None, None, None)
        if self._process_context:
            await self._process_context.__aexit__(None, None, None)

    async def get_tools_schema(self):
        if not self.session:
            return []
        
        response = await self.session.list_tools()
        tools = []
        for tool in response.tools:
            tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema
                }
            })
        return tools

    async def call_tool(self, name: str, args: dict):
        if not self.session:
            raise Exception("MCP session not initialized")
        
        result = await self.session.call_tool(name, arguments=args)
        
        # Formatear la respuesta del MCP para devolverla como diccionario/texto
        # Las tools de MCP devuelven content, que es una lista de objetos (ej: TextContent)
        # Extraemos el texto:
        outputs = []
        for content in result.content:
            if content.type == "text":
                outputs.append(content.text)
        
        # Lo unimos y tratamos de parsear como JSON si es posible, sino como string
        output_str = "\n".join(outputs)
        import json
        try:
            return json.loads(output_str)
        except:
            return {"result": output_str}

mcp_manager = MCPManager()
