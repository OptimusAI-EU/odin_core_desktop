import asyncio
import os
import sys
from pathlib import Path
from typing import Optional, List, Any
from contextlib import AsyncExitStack

# MCP Imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# LangChain Imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic # Example import
# from langchain_openai import ChatOpenAI     # Example import
# from langchain_groq import ChatGroq         # Example import
from langchain_ollama import ChatOllama     # Example import

# Environment Loading (optional, if API keys are needed)
from dotenv import load_dotenv
load_dotenv()

class MCPClient:
    def __init__(self, model: BaseChatModel):
        """
        Initializes the MCP Client.

        Args:
            model: A LangChain compatible chat model instance (e.g., ChatAnthropic, ChatOpenAI).
        """
        self.model: BaseChatModel = model
        self.session: Optional[ClientSession] = None
        self.tools: List[BaseTool] = []
        self.agent: Optional[Any] = None # To store the LangChain agent executor
        self.exit_stack = AsyncExitStack()
        self.stdio = None # For type hinting/clarity
        self.write = None # For type hinting/clarity

    async def connect_to_server(self, server_script_path: str):
        """
        Connects to an MCP server, loads tools, and creates the LangChain agent.

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        print("Attempting to connect to server...") # Added
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_path = Path(server_script_path)
        if not server_path.is_absolute():
             server_path = Path(__file__).parent / server_script_path

        if not server_path.exists(): # Added check
             print(f"Error: Server script not found at {server_path}")
             raise FileNotFoundError(f"Server script not found at {server_path}")

        server_params = StdioServerParameters(
            command=command,
            args=[str(server_path)],
            cwd=str(server_path.parent),
            env=os.environ.copy()
        )
        print(f"Starting server with: cmd='{command}', args='{[str(server_path)]}', cwd='{str(server_path.parent)}'")

        try:
            print("Initializing stdio transport...") # Added
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.write = stdio_transport
            print("Stdio transport initialized.") # Added

            print("Initializing MCP session...") # Added
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
            await self.session.initialize()
            print("MCP Session initialized.")

            print("Loading MCP tools via LangChain adapter...") # Added
            self.tools = await load_mcp_tools(self.session)
            print(f"Tools loaded: {[tool.name for tool in self.tools]}")

            if not self.tools:
                print("Warning: No tools loaded from the MCP server.")
                # Agent creation might fail or be skipped if no tools are present
                self.agent = None # Explicitly set agent to None if no tools
                return # Exit connection attempt if no tools

            print("Creating LangChain agent...") # Added
            self.agent = create_react_agent(self.model, self.tools)
            if self.agent:
                 print("LangChain agent created successfully.") # Added
            else:
                 print("Error: Failed to create LangChain agent.") # Added

        except Exception as e:
            print(f"Error during connection/setup: {e}") # Modified
            # Ensure cleanup happens if connection fails partially
            await self.cleanup() # Consider if cleanup is appropriate here or should be left to main's finally
            raise # Re-raise the exception to be caught by main

    async def process_query(self, query: str) -> str:
        """
        Processes a query using the configured LangChain agent and MCP tools.

        Args:
            query: The user's query string.

        Returns:
            The agent's final response string.
        """
        if not self.agent:
            return "Error: Agent not initialized. Please connect to the server first."
        # Removed the check for self.session.is_initialized as it caused an AttributeError
        # If self.agent exists, self.session should also exist and be initialized.
        # We can add a basic check for self.session just in case.
        if not self.session:
             return "Error: MCP session is not available."

        messages = [HumanMessage(content=query)]
        print(f"\nInvoking agent with query: '{query}'")
        try:
            # Use agent.ainvoke for asynchronous execution
            # The input format depends on the agent type (create_react_agent expects {"messages": ...})
            response = await self.agent.ainvoke({"messages": messages})

            # The output format also depends on the agent.
            # For create_react_agent, the final answer is often in response['messages'][-1].content
            # Adjust extraction logic based on observed agent output structure if needed.
            if isinstance(response, dict) and "messages" in response and response["messages"]:
                 final_response = response["messages"][-1].content
                 print(f"Agent raw response: {response}") # Log raw for debugging
                 print(f"Agent final answer: {final_response}")
                 return final_response
            else:
                 # Fallback or error handling if the structure is unexpected
                 print(f"Unexpected agent response structure: {response}")
                 return f"Agent finished, but response format was unexpected: {response}"

        except Exception as e:
            print(f"Error during agent invocation: {e}")
            # Consider more specific error handling based on potential LangChain/MCP errors
            return f"Error processing query: {e}"


    async def chat_loop(self):
        """Runs an interactive chat loop."""
        if not self.agent:
             print("Cannot start chat loop: Agent not initialized.")
             return

        print("\nMCP Client Started (LangChain Mode)!")
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

            except (EOFError, KeyboardInterrupt): # Handle Ctrl+D/Ctrl+C gracefully
                print("\nExiting chat loop.")
                break
            except Exception as e:
                # Catch errors from input() or other unexpected issues
                print(f"\nAn error occurred in the chat loop: {str(e)}")


    async def cleanup(self):
        """Cleans up resources."""
        print("\nCleaning up resources...")
        await self.exit_stack.aclose()
        print("Resources cleaned up.")


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client_single_server.py <path_to_server_script>") # Corrected usage message
        sys.exit(1)

    server_script_path = sys.argv[1]

    # --- Model Configuration ---
    try:
        # Example: Using Anthropic Claude 3.5 Sonnet
        # model = ChatAnthropic(
        #      model="claude-3-5-sonnet-20240620",
        #      temperature=0 # Adjust temperature as needed
        # )

        # Example: Using OpenAI GPT-4o (requires langchain-openai, OPENAI_API_KEY)
        # from langchain_openai import ChatOpenAI
        # model = ChatOpenAI(model="gpt-4o", temperature=0)

        # Example: Using a local Ollama model (requires langchain-ollama, Ollama running)
        # from langchain_ollama import ChatOllama
        model = ChatOllama(model="qwen2.5-coder") # Replace 'llama3' with your model
        print("Language model initialized.") # Added

    except ImportError as e:
         print(f"Error importing model integration: {e}. Make sure the required package is installed.")
         sys.exit(1)
    except Exception as e:
         print(f"Error initializing the language model: {e}")
         sys.exit(1)

    # --- Client Initialization and Execution ---
    client = MCPClient(model=model)
    try:
        print("Connecting client to server...") # Added
        await client.connect_to_server(server_script_path)
        print("Client connection attempt finished.") # Added

        # Only start chat loop if connection and agent creation succeeded
        if client.agent:
             print("Agent found, starting chat loop...") # Added
             await client.chat_loop()
        else:
             # Added more specific message
             print("Agent not initialized (check connection/tool loading logs). Exiting.")
    except FileNotFoundError as e: # Catch specific error from connect_to_server
         print(f"Setup failed: {e}")
    except Exception as e:
        print(f"\nAn unhandled error occurred during client execution: {e}") # Modified
    finally:
        await client.cleanup()

if __name__ == "__main__":
    # Note: No need to import sys again here
    asyncio.run(main())