# File: odin_core.py
# (Place this in odin_core_desktop/python_src/)

import asyncio
import os
import sys
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict, Any

from dotenv import load_dotenv

# LangChain and MCP Imports (ensure these are all needed at this core level)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient # For BrowserUse (SSE)
from langgraph.prebuilt import create_react_agent

# Assuming client_multi_server.py (with MultiServerMCPClientWrapper) is in the same directory
# or SCRIPT_DIR is correctly added to sys.path by the importing script.
from client_multi_server import MultiServerMCPClientWrapper

# LLM Imports
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# --- Global Setup (Load .env, SCRIPT_DIR) ---
SCRIPT_DIR = Path(__file__).resolve().parent
dotenv_path = SCRIPT_DIR / '.env' # Assumes .env is next to odin_core.py
load_dotenv(dotenv_path=dotenv_path)
print(f"ODIN_CORE: Loaded .env file from: {dotenv_path.resolve()}")


# --- Base Agent Class ---
class BaseODINAgent(ABC):
    """Base class for all ODIN agents."""

    def __init__(self, agent_llm: BaseChatModel, agent_name: str):
        """Initialize the base agent."""
        self.agent_llm = agent_llm
        self.agent_name = agent_name
        self.progress_callback: Optional[Callable[[float, str], None]] = None
        print(f"ODIN_CORE: Initialized {self.agent_name} with LLM: {agent_llm.__class__.__name__}")

    def set_progress_callback(self, callback: Callable[[float, str], None]) -> None:
        """Set callback for progress updates (e.g., for Gradio)."""
        self.progress_callback = callback

    def _report_progress(self, value: float, description: str = "") -> None:
        """Report progress through callback if set."""
        if self.progress_callback:
            self.progress_callback(value, description)

    @abstractmethod
    async def execute(self, main_input: str, **kwargs) -> str:
        """
        Execute the agent's primary task.
        kwargs will hold agent-specific parameters.
        """
        pass

# --- Stdio-Based Agent Mixin ---
"""This handles the common logic for stdio-based agents."""

class StdioAgentMixin:
    async def _execute_stdio_task(
        self: BaseODINAgent,
        server_script_name: str,
        server_key: str,
        tool_invocation_message: str
    ) -> str:
        self._report_progress(0.2, f"Configuring {self.agent_name} (stdio)...")
        print(f"ODIN_CORE: Handling {self.agent_name} task via stdio server: {server_script_name}")

        server_script_full_path = SCRIPT_DIR / server_script_name
        if not server_script_full_path.exists():
            error_msg = f"Error: Server script '{server_script_name}' not found at '{server_script_full_path}'."
            print(error_msg, file=sys.stderr)
            return error_msg

        # Initialize the client wrapper with the LLM model
        stdio_client_wrapper = MultiServerMCPClientWrapper(model=self.agent_llm)
        
        server_configs = {
            server_key: {
                "command": sys.executable,
                "args": [str(server_script_full_path)],
                "transport": "stdio",
                "env": os.environ.copy()
            }
        }
        # If CWD is truly essential for the mcp_*.py script and it's not SCRIPT_DIR,
        # the mcp_*.py script itself would need to os.chdir() at its beginning.
        # However, for finding other files relative to itself, Path(__file__).parent works within the script.

        print(f"ODIN_CORE: Stdio Server Config: {server_configs}")
        print(f"ODIN_CORE: Tool invocation message: {tool_invocation_message}")

        final_result = f"Error: {self.agent_name} stdio task did not complete."
        try:
            self._report_progress(0.3, "Connecting to stdio server...")
            await stdio_client_wrapper.connect_to_servers(server_configs)
            print(f"ODIN_CORE: {self.agent_name} client wrapper connected.")

            if stdio_client_wrapper.agent:
                self._report_progress(0.5, "Processing query via stdio...")
                response = await stdio_client_wrapper.process_query(tool_invocation_message)
                final_result = response
                print(f"ODIN_CORE: {self.agent_name} stdio query processed.")
            else:
                final_result = f"Error: {self.agent_name} stdio agent not initialized."
        except Exception as e_stdio:
            error_msg = f"Error during {self.agent_name} stdio execution: {e_stdio}"
            print(error_msg, file=sys.stderr)
            # import traceback # Already imported in calling function if this is top level error
            # traceback.print_exc(file=sys.stderr)
            final_result = error_msg # Return the specific error
        finally:
            self._report_progress(0.9, "Cleaning up stdio client resources...")
            await stdio_client_wrapper.cleanup()
            print(f"ODIN_CORE: {self.agent_name} stdio client wrapper cleanup complete.")
        
        # No longer reporting progress(1.0) here, as the calling function run_odin_task does
        return final_result

# ... (rest of odin_core.py) ...

# --- Concrete Agent Implementations ---

class SoftwareCompanyAgent(BaseODINAgent, StdioAgentMixin):
    def __init__(self, agent_llm: BaseChatModel):
        super().__init__(agent_llm, "SoftwareCompanyAgent")

    async def execute(self, main_input: str, **kwargs) -> str:
        self._report_progress(0, f"Starting Software Company task for: {main_input}")
        # Extract MetaGPT specific params from kwargs with defaults
        project_name = kwargs.get("project_name", "")
        investment = kwargs.get("investment", 3.0)
        n_round = kwargs.get("n_round", 5)
        code_review = kwargs.get("code_review", True)
        run_tests = kwargs.get("run_tests", False)
        implement = kwargs.get("implement", True)
        inc = kwargs.get("inc", False)
        project_path = kwargs.get("project_path", "")
        reqa_file = kwargs.get("reqa_file", "")
        max_auto_summarize_code = kwargs.get("max_auto_summarize_code", 0)
        recover_path = kwargs.get("recover_path", None)

        param_details = f"idea='{main_input}'"
        if project_name: param_details += f", project_name='{project_name}'"
        if project_path: param_details += f", project_path='{project_path}'"
        if reqa_file: param_details += f", reqa_file='{reqa_file}'"
        if recover_path: param_details += f", recover_path='{recover_path}'"
        param_details += (f", investment={investment}, n_round={n_round}, code_review={code_review}, "
                          f"run_tests={run_tests}, implement={implement}, inc={inc}, "
                          f"max_auto_summarize_code={max_auto_summarize_code}")
        
        tool_message = f"Use the execute_metagpt tool with parameters: {param_details}."
        
        return await self._execute_stdio_task(
            server_script_name="mcp_software_company.py",
            server_key="metagpt_server", # Key used in MultiServerMCPClientWrapper config
            tool_invocation_message=tool_message
        )

class DeepResearcherAgent(BaseODINAgent, StdioAgentMixin):
    def __init__(self, agent_llm: BaseChatModel):
        super().__init__(agent_llm, "DeepResearcherAgent")
        print(f"ODIN_CORE: Initialized {self.__class__.__name__} with LLM: {agent_llm.__class__.__name__}")

    async def execute(self, main_input: str, **kwargs) -> str:
        print(f"ODIN_CORE: DeepResearcher executing with input: {main_input}")
        print(f"ODIN_CORE: DeepResearcher kwargs: {kwargs}")
        
        self._report_progress(0, f"Starting Deep Researcher task for: {main_input}")
        report_type = kwargs.get("report_type", "research_report")
        print(f"ODIN_CORE: DeepResearcher report_type: {report_type}")
        
        tool_message = f"Use the execute_deep_research tool with query='{main_input}' and report_type='{report_type}'."
        print(f"ODIN_CORE: DeepResearcher tool message: {tool_message}")
        
        return await self._execute_stdio_task(
            server_script_name="mcp_deep_researcher.py",
            server_key="deep_researcher_server",
            tool_invocation_message=tool_message
        )

class DataInterpreterAgent(BaseODINAgent, StdioAgentMixin):
    def __init__(self, agent_llm: BaseChatModel):
        super().__init__(agent_llm, "DataInterpreterAgent")

    async def execute(self, main_input: str, **kwargs) -> str:
        self._report_progress(0, f"Starting Data Interpreter task for: {main_input}")
        # No specific kwargs for DataInterpreter in your app.py
        tool_message = f"Use the execute_data_interpreter tool with requirement='{main_input}'."
        
        return await self._execute_stdio_task(
            server_script_name="mcp_data_interpreter.py",
            server_key="data_interpreter_server",
            tool_invocation_message=tool_message
        )

class BrowserUseAgent(BaseODINAgent): # Does not use StdioAgentMixin
    def __init__(self, agent_llm: BaseChatModel):
        super().__init__(agent_llm, "BrowserUseAgent")

    async def execute(self, main_input: str, **kwargs) -> str:
        self._report_progress(0, f"Starting Browser Use task for: {main_input}")
        platform = kwargs.get("platform", "upwork") # Get platform from kwargs
        
        browser_server_url = os.getenv("BROWSER_SERVER_URL", "http://localhost:8000/sse")
        sse_server_config = {"browser-use": {"url": browser_server_url, "transport": "sse"}}
        tool_message = f"Use the execute_browseruse tool with task='{main_input}' and platform='{platform}'."
        messages = [HumanMessage(content=tool_message)]
        
        print(f"ODIN_CORE: Expecting Browser SSE server at: {browser_server_url}, platform: {platform}")

        # This logic handles the MultiServerMCPClient API versioning
        final_result = "Error: BrowserUse task did not complete."
        mcp_client_for_sse = None # For direct instantiation path
        try:
            # Try context manager style (for older langchain-mcp-adapters)
            print("ODIN_CORE: Attempting Browser SSE connection with MultiServerMCPClient as context manager...")
            # The client here is the MultiServerMCPClient instance itself
            async with MultiServerMCPClient(sse_server_config) as client_session_provider:
                self._report_progress(0.4, "Loading tools from SSE server...")
                # get_tools might be sync or async depending on adapter version
                try: tools = await client_session_provider.get_tools()
                except TypeError: tools = client_session_provider.get_tools()

                if not tools: return "Error: No tools loaded from browser SSE server."
                print(f"ODIN_CORE: Browser tools loaded: {[tool.name for tool in tools]}")
                agent = create_react_agent(self.agent_llm, tools)
                agent_response = await agent.ainvoke({"messages": messages})
                final_result = agent_response["messages"][-1].content
        except TypeError as te: # Likely "MultiServerMCPClient is not an async context manager"
            print(f"ODIN_CORE: Browser SSE TypeError (likely new adapter API): {te}. Trying direct instantiation...")
            mcp_client_for_sse = MultiServerMCPClient(sse_server_config)
            try:
                self._report_progress(0.4, "Loading tools from SSE server (direct)...")
                try: tools = await mcp_client_for_sse.get_tools()
                except TypeError: tools = mcp_client_for_sse.get_tools()
                
                if not tools: return "Error: No tools loaded (direct SSE)."
                print(f"ODIN_CORE: Browser tools loaded (direct): {[tool.name for tool in tools]}")
                agent = create_react_agent(self.agent_llm, tools)
                agent_response = await agent.ainvoke({"messages": messages})
                final_result = agent_response["messages"][-1].content
            finally:
                if hasattr(mcp_client_for_sse, 'aclose'): await mcp_client_for_sse.aclose()
                elif hasattr(mcp_client_for_sse, 'cleanup'): await mcp_client_for_sse.cleanup()
                print("ODIN_CORE: Browser SSE Client (direct) cleaned up.")
        except ConnectionRefusedError:
            final_result = f"Error: Connection refused for SSE server at {browser_server_url}."
            print(final_result, file=sys.stderr)
        except Exception as e_sse:
            error_msg = f"Error during Browser Use SSE: {e_sse}"
            print(error_msg, file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            final_result = error_msg
        
        self._report_progress(1.0, "Task Complete.")
        return final_result


# --- LLM Instance Factory (Moved from app.py's run_mcp_task) ---
def create_llm_instance(agent_model_name: str) -> BaseChatModel:
    """Creates and returns a LangChain LLM instance based on model name."""
    print(f"ODIN_CORE: Creating LLM instance for: {agent_model_name}")
    groq_model_list = [
        "llama-3.3-70b-versatile", "llama3-8b-8192", "llama3-70b-8192",
        "mixtral-8x7b-32768", "gemma-7b-it"
    ]
    openrouter_model_patterns = ["deepseek/", "meta-llama/", "google/gemini"]
    openai_direct_model_patterns = ["gpt-4", "gpt-3.5"]

    if agent_model_name in groq_model_list:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key: raise ValueError("GROQ_API_KEY not set for Groq model.")
        llm = ChatGroq(model_name=agent_model_name, api_key=groq_api_key, temperature=0)
    elif any(pattern in agent_model_name.lower() for pattern in openrouter_model_patterns):
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key: raise ValueError("OPENROUTER_API_KEY not set for OpenRouter model.")
        llm = ChatOpenAI(model=agent_model_name, api_key=api_key, base_url=base_url, temperature=0)
    elif any(pattern in agent_model_name.lower() for pattern in openai_direct_model_patterns):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key: raise ValueError("OPENAI_API_KEY not set for direct OpenAI model.")
        llm = ChatOpenAI(model=agent_model_name, api_key=openai_api_key, temperature=0)
    else: # Fallback to Ollama
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        llm = ChatOllama(model=agent_model_name, base_url=ollama_base_url, temperature=0)
    
    if llm is None:
        raise ValueError(f"Could not create LLM for model name: {agent_model_name}")
    print(f"ODIN_CORE: LLM instance created: {llm.__class__.__name__} for model {agent_model_name}")
    return llm


# --- Agent Factory Function ---
AGENT_CLASSES: Dict[str, type[BaseODINAgent]] = {
    "softwarecompany": SoftwareCompanyAgent,
    "deepresearcher": DeepResearcherAgent,
    "datainterpreter": DataInterpreterAgent,
    "browseruse": BrowserUseAgent
}

def create_odin_agent(agent_type: str, agent_llm_instance: BaseChatModel) -> BaseODINAgent:
    """Factory function to create ODIN agents."""
    normalized_agent_type = agent_type.lower().replace(" ", "").replace("_", "").replace("-", "")
    print(f"ODIN_CORE: Creating agent for type='{agent_type}' (normalized='{normalized_agent_type}')")
    print(f"ODIN_CORE: Available agent types: {list(AGENT_CLASSES.keys())}")
    
    agent_class = AGENT_CLASSES.get(normalized_agent_type)
    if not agent_class:
        raise ValueError(f"Unknown agent type: '{agent_type}' (normalized: '{normalized_agent_type}'). Available: {list(AGENT_CLASSES.keys())}")
    
    print(f"ODIN_CORE: Selected agent class: {agent_class.__name__}")
    return agent_class(agent_llm_instance)


# --- Main ODIN Core Execution Function (replaces run_mcp_task from app.py) ---
async def run_odin_task(
    agent_type: str,
    main_input: str,
    agent_model_name: str, # Name of the LLM model to use for the agent
    progress_callback: Optional[Callable[[float, str], None]] = None, # For Gradio progress
    **agent_specific_kwargs # All other params like investment, report_type, platform
) -> str:
    """
    Main entry point to run an ODIN agent task.
    """
    if progress_callback:
        progress_callback(0, "Initializing ODIN task...")
    
    print(f"\nODIN_CORE: ====== Task Execution Start ======")
    print(f"ODIN_CORE: Agent Type: '{agent_type}'")
    print(f"ODIN_CORE: Input: '{main_input[:50]}...'")
    print(f"ODIN_CORE: LLM Model: '{agent_model_name}'")
    print(f"ODIN_CORE: Agent kwargs: {agent_specific_kwargs}")

    try:
        # 1. Create LLM instance
        if progress_callback: progress_callback(0.1, f"Initializing LLM ({agent_model_name})...")
        print(f"ODIN_CORE: Creating LLM instance for model {agent_model_name}...")
        llm_instance = create_llm_instance(agent_model_name)

        # 2. Create the specific ODIN agent
        if progress_callback: progress_callback(0.15, f"Creating {agent_type} agent...")
        odin_agent = create_odin_agent(agent_type, llm_instance)
        
        if progress_callback:
            odin_agent.set_progress_callback(progress_callback) # Pass Gradio progress to agent

        # 3. Execute the task
        # The agent_specific_kwargs dictionary already contains all relevant parameters
        # like investment, n_round, report_type, platform, etc.
        # The individual agent's execute method will pick what it needs from kwargs.
        result = await odin_agent.execute(main_input, **agent_specific_kwargs)
        
        print(f"ODIN_CORE: Task for {agent_type} completed. Result: {result[:100]}...")
        if progress_callback: progress_callback(1, "Task complete.")
        return result

    except Exception as e:
        error_msg = f"Error in ODIN Core task execution: {e}"
        print(error_msg, file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        if progress_callback: progress_callback(1, f"Error: {e}")
        return error_msg

# --- Expose symbols for import ---
__all__ = [
    'BaseODINAgent',
    'SoftwareCompanyAgent',
    'DeepResearcherAgent',
    'DataInterpreterAgent',
    'BrowserUseAgent',
    'create_llm_instance',
    'create_odin_agent',
    'run_odin_task',
    'SCRIPT_DIR' # Export SCRIPT_DIR if other modules need to reference it relative to odin_core
]