import asyncio
import os
import logging
import json # Keep for saving error details
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Optional # Keep Optional for BrowserConfig

# --- MCP/BrowserUse/LangChain Imports ---
from mcp.server.fastmcp import FastMCP
# Remove Controller import
from browser_use import Agent, Browser, BrowserConfig
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic

# --- Configuration Loading ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
mcp = FastMCP("browseruse_executor")

# --- Helper Functions (get_llm_instance, get_browser_config - remain the same) ---
def get_llm_instance():
    # ... (implementation as before) ...
    """Initializes the LangChain LLM based on environment variables."""
    provider = os.getenv("BROWSER_LLM_PROVIDER", "ollama").lower()
    model_name = os.getenv("BROWSER_LLM_MODEL")
    api_key = os.getenv("BROWSER_API_KEY") # General key, specific keys checked below

    if not model_name:
        raise ValueError("BROWSER_LLM_MODEL environment variable is not set.")

    logger.info(f"Attempting to initialize LLM provider: {provider}, model: {model_name}")

    if provider == "groq":
        groq_api_key = api_key or os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY or BROWSER_API_KEY must be set for Groq.")
        return ChatGroq(model=model_name, api_key=groq_api_key)
    elif provider == "ollama":
        base_url = os.getenv("BROWSER_OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model_name, base_url=base_url)
    elif provider == "openai":
        openai_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY or BROWSER_API_KEY must be set for OpenAI.")
        return ChatOpenAI(model=model_name, api_key=openai_api_key)
    elif provider == "anthropic":
        anthropic_api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY or BROWSER_API_KEY must be set for Anthropic.")
        return ChatAnthropic(model=model_name, api_key=anthropic_api_key)
    elif provider == "openrouter":
        base_url = os.getenv("BROWSER_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        openrouter_api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY or BROWSER_API_KEY must be set for OpenRouter.") # Corrected error message
        # Assuming ChatOpenAI compatible API for OpenRouter
        return ChatOpenAI(model=model_name, api_key=openrouter_api_key, base_url=base_url)
    elif provider == "lmstudio":
        base_url = os.getenv("BROWSER_LMSTUDIO_BASE_URL", "http://localhost:1234/v1") # Ensure /v1 for OpenAI compatibility
        # LM Studio often doesn't require an API key
        return ChatOpenAI(model=model_name, base_url=base_url, api_key="dummy-key") # Use a dummy key
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def get_browser_config() -> BrowserConfig:
    # ... (implementation as before) ...
    """Creates BrowserConfig based on environment variables."""
    use_builtin_str = os.getenv("BROWSER_USE_BUILTIN_CHROME", "false").lower()
    headless_str = os.getenv("BROWSER_HEADLESS", "true").lower()
    disable_sec_str = os.getenv("BROWSER_DISABLE_SECURITY", "true").lower()

    use_builtin = use_builtin_str == "true"
    headless = headless_str == "true"
    disable_security = disable_sec_str == "true"

    chrome_path: Optional[str] = None
    if not use_builtin:
        chrome_path = os.getenv("BROWSER_CHROME_PATH")
        if not chrome_path:
            raise ValueError("BROWSER_CHROME_PATH must be set if BROWSER_USE_BUILTIN_CHROME is false.")
        logger.info(f"Using specified Chrome path: {chrome_path}")
    else:
        logger.info("Using built-in Chromium.")

    return BrowserConfig(
        chrome_instance_path=chrome_path, # Will be None if use_builtin is True
        headless=headless,
        disable_security=disable_security
    )


# --- MCP Tool Implementation ---
@mcp.tool(name="execute_browseruse", description="Executes browseruse tasks (scraping, automation) and saves the execution history to JSON.")
async def execute_browseruse(task: str) -> str:
    """
    Execute browseruse app, save the full execution history to a JSON file,
    and return the file path.
    """
    output_dir = Path(__file__).parent / "output_browser"
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure dir exists

    try:
        # --- Initialize LLM ---
        llm = get_llm_instance()
        logger.info(f"LLM Initialized: {llm.__class__.__name__}")

        # --- Initialize Browser ---
        browser_config = get_browser_config()
        browser = Browser(config=browser_config)
        logger.info(f"Browser Initialized with config: {browser_config}")

        # --- Initialize Agent (without Controller) ---
        agent = Agent(
            llm=llm,
            task=task,
            browser=browser,
            # controller=controller, # Removed controller
            use_vision=False
        )
        logger.info("BrowserUse Agent Initialized.")

        # --- Run Agent ---
        logger.info(f"Running agent with task: '{task}'...")
        # Add max_steps if desired, otherwise it runs until completion/error
        history = await agent.run()
        logger.info("Agent run completed.")

        # --- Save History Output ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_task_part = "".join(c if c.isalnum() else "_" for c in task[:30])
        # Use a more descriptive filename for history
        output_filename = f"browser_history_{timestamp}_{safe_task_part}.json"
        output_filepath = output_dir / output_filename

        try:
            history.save_to_file(str(output_filepath)) # Use the save_to_file method
            logger.info(f"Execution history saved to: {output_filepath}")
            # You might still want the final summary message from the agent
            final_summary = history.final_result()
            logger.info(f"Agent final summary: {final_summary}")
            return f"Success: Browser task completed. Full history saved to {output_filepath}"
        except Exception as save_error:
            logger.error(f"Failed to save history file: {save_error}", exc_info=True)
            return f"Error: Browser task completed but failed to save history file: {save_error}"


    except Exception as e:
        # ... (error handling remains largely the same, saves error details) ...
        logger.error(f"Browser task execution failed: {str(e)}", exc_info=True)
        error_result = {
            "error": str(e),
            "task": task,
            "timestamp": datetime.now().isoformat()
        }
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            error_filename = f"browser_error_{timestamp}.json"
            error_filepath = output_dir / error_filename
            with open(error_filepath, 'w', encoding='utf-8') as f:
                json.dump(error_result, f, indent=4, ensure_ascii=False)
            return f"Error: {str(e)}. Details saved to {error_filepath}"
        except Exception as save_err:
            logger.error(f"Failed to save error details: {save_err}")
            return f"Error: {str(e)}. Failed to save error details."


if __name__ == "__main__":
    # ... (main execution remains the same) ...
    logger.info("Starting BrowserUse MCP server...")
    mcp.run(transport="sse")
    logger.info("BrowserUse MCP server stopped.")