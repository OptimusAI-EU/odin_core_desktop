import asyncio
import os
import sys
from pathlib import Path
from typing import Optional, List, Any, Dict
from contextlib import AsyncExitStack

# MCP Imports
# Note: ClientSession and StdioServerParameters might not be directly needed
# if MultiServerMCPClient handles the underlying connections.
# Keep them if needed for specific server types or future expansion.
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.client import MultiServerMCPClient # Key import

# LangChain Imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
# load_mcp_tools might not be needed if MultiServerMCPClient provides get_tools()
# from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

# Environment Loading
from dotenv import load_dotenv
load_dotenv()

class MultiServerMCPClientWrapper:
    def __init__(self, model: BaseChatModel):
        """
        Initializes the Multi-Server MCP Client Wrapper.

        Args:
            model: A LangChain compatible chat model instance.
        """
        self.model: BaseChatModel = model
        self.client: Optional[MultiServerMCPClient] = None
        self.tools: List[BaseTool] = []
        self.agent: Optional[Any] = None
        # Use AsyncExitStack to manage the MultiServerMCPClient context
        self.exit_stack = AsyncExitStack()

    async def connect_to_servers(self, server_configs: Dict[str, Dict]):
        """
        Connects to multiple MCP servers based on the provided configurations,
        loads tools, and creates the LangChain agent.

        Args:
            server_configs: A dictionary where keys are server names and
                            values are configuration dictionaries for MultiServerMCPClient.
                            Example:
                            {
                                "calculator": {"command": "python", "args": ["path/to/server.py"], "transport": "stdio"},
                                "weather": {"url": "http://localhost:8000", "transport": "sse"}
                            }
        """
        print("Attempting to connect to multiple servers...")
        if not server_configs:
            raise ValueError("Server configurations dictionary cannot be empty.")

        try:
            # Enter the MultiServerMCPClient context using AsyncExitStack
            self.client = await self.exit_stack.enter_async_context(
                MultiServerMCPClient(server_configs)
            )
            print("MultiServerMCPClient initialized.")

            # Get tools from all connected servers
            print("Loading tools from all connected servers...")
            self.tools = self.client.get_tools() # Use the client's method
            print(f"Tools loaded: {[tool.name for tool in self.tools]}")

            if not self.tools:
                print("Warning: No tools loaded from any MCP server.")
                self.agent = None
                return # Exit if no tools

            # Create the LangChain agent
            print("Creating LangChain agent...")
            self.agent = create_react_agent(self.model, self.tools)
            if self.agent:
                 print("LangChain agent created successfully.")
            else:
                 print("Error: Failed to create LangChain agent.")
                 # Ensure client is cleaned up if agent creation fails
                 await self.cleanup()
                 raise RuntimeError("Failed to create LangChain agent.")

        except Exception as e:
            print(f"Error during multi-server connection/setup: {e}")
            await self.cleanup() # Ensure cleanup on error
            raise # Re-raise the exception

    async def process_query(self, query: str) -> str:
        """
        Processes a query using the configured LangChain agent and tools from multiple servers.

        Args:
            query: The user's query string.

        Returns:
            The agent's final response string.
        """
        if not self.agent:
            return "Error: Agent not initialized. Please connect to servers first."
        if not self.client:
             return "Error: MultiServerMCPClient not initialized."

        messages = [HumanMessage(content=query)]
        print(f"\nInvoking agent with query: '{query}'")
        try:
            response = await self.agent.ainvoke({"messages": messages})

            if isinstance(response, dict) and "messages" in response and response["messages"]:
                 final_response = response["messages"][-1].content
                 print(f"Agent raw response: {response}")
                 print(f"Agent final answer: {final_response}")
                 return final_response
            else:
                 print(f"Unexpected agent response structure: {response}")
                 return f"Agent finished, but response format was unexpected: {response}"

        except Exception as e:
            print(f"Error during agent invocation: {e}")
            return f"Error processing query: {e}"

    async def chat_loop(self):
        """Runs an interactive chat loop."""
        if not self.agent:
             print("Cannot start chat loop: Agent not initialized.")
             return

        print("\nMulti-Server MCP Client Started (LangChain Mode)!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if not query:
                    continue
                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat loop.")
                break
            except Exception as e:
                print(f"\nAn error occurred in the chat loop: {str(e)}")

    async def cleanup(self):
        """Cleans up resources by closing the MultiServerMCPClient context."""
        print("\nCleaning up resources...")
        # AsyncExitStack handles calling the __aexit__ of MultiServerMCPClient
        await self.exit_stack.aclose()
        self.client = None # Clear the client reference
        self.agent = None # Clear the agent reference
        print("Resources cleaned up.")


async def main():
    # --- Server Configuration ---
    # Define the configurations for each server you want to connect to.
    # Ensure paths are correct (absolute paths are often safer).
    # Make sure any non-stdio servers (like SSE/HTTP) are running independently.
    current_dir = Path(__file__).parent
    server_configs = {
        "calculator": {
            "command": sys.executable, # Use sys.executable for python path
            "args": [str(current_dir / "math_server.py")], # Example: server in same dir
            "cwd": str(current_dir), # Set working directory
            "transport": "stdio",
            "env": os.environ.copy(), # Pass environment
        },
        # Add other servers here, e.g.:
        "weather": {
            "url": "http://localhost:8000/sse", # Ensure this server is running
            "transport": "sse",
        },
        # "researcher": {
        #     "command": sys.executable,
        #     "args": [str(current_dir / "mcp_researcher.py")],
        #     "cwd": str(current_dir),
        #     "transport": "stdio",
        #     "env": os.environ.copy(),
        # }
    }
    print(f"Server configurations: {server_configs}")

    # --- Model Configuration ---
    try:
        # Choose and configure your LangChain model
        model = ChatOllama(model="qwen2.5-coder") # Example: Ollama
        # model = ChatAnthropic(model="claude-3-5-sonnet-20240620") # Example: Anthropic
        print("Language model initialized.")

    except ImportError as e:
         print(f"Error importing model integration: {e}. Make sure the required package is installed.")
         sys.exit(1)
    except Exception as e:
         print(f"Error initializing the language model: {e}")
         sys.exit(1)

    # --- Client Initialization and Execution ---
    client_wrapper = MultiServerMCPClientWrapper(model=model)
    try:
        print("Connecting client to servers...")
        await client_wrapper.connect_to_servers(server_configs)
        print("Client connection attempt finished.")

        if client_wrapper.agent:
             print("Agent found, starting chat loop...")
             await client_wrapper.chat_loop()
        else:
             print("Agent not initialized (check connection/tool loading logs). Exiting.")

    except FileNotFoundError as e:
         print(f"Setup failed: Server script not found - {e}")
    except Exception as e:
        print(f"\nAn unhandled error occurred during client execution: {e}")
    finally:
        # Cleanup is handled by the AsyncExitStack when connect_to_servers completes
        # or raises an error, but we call it again ensure closure if chat_loop exits.
        await client_wrapper.cleanup()

if __name__ == "__main__":
    asyncio.run(main())

