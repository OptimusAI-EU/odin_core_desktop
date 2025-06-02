import asyncio
import asyncio
import os
import logging
import yaml
from mcp.server.fastmcp import FastMCP
from metagpt.team import Team
from metagpt.roles import ProductManager, Architect, ProjectManager, Engineer
from metagpt.logs import logger
from metagpt.roles.di.data_interpreter import DataInterpreter
from metagpt.utils.recovery_util import save_history

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("data_interpreter_executor")
@mcp.tool(name="execute_data_interpreter", description="Executes data interpreter project generation")
async def execute_data_interpreter(requirement: str) -> str:
    """Execute data interpreter project with proper configuration"""
    try:
                    
        # Initialize data interpreter 
        di = DataInterpreter()
        rsp = await di.run(requirement)
        save_history(role=di)
        
        return rsp
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
    