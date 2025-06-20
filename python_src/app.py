# File: app.py (derived from odin_core_ui.py, aiming for minimal necessary changes)
import asyncio
import os
import sys
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv
import time

# --- Path Setup (from odin_core_ui.py) ---
# Assuming app.py, client_multi_server.py, and mcp_*.py scripts are all
# in the SCRIPT_DIR (e.g., your python_src or odin_core directory).
# If client_multi_server.py has a different relative location, this might need adjustment.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# The grandparent_dir logic from odin_core_ui.py might be important if
# client_multi_server.py is indeed located in a parent directory relative
# to where the mcp_*.py scripts (and this app.py) reside.
# Let's include it as it was in your working version, but comment if not needed.
# original_parent_dir = Path(__file__).resolve().parent # This is SCRIPT_DIR
# original_grandparent_dir = original_parent_dir.parent
# if str(original_grandparent_dir) not in sys.path:
# sys.path.insert(0, str(original_grandparent_dir))

try:
    from client_multi_server import MultiServerMCPClientWrapper
    from langchain_mcp_adapters.client import MultiServerMCPClient # For BrowserUse
    from langgraph.prebuilt import create_react_agent
    from langchain_core.messages import HumanMessage
except ImportError as e:
    print(f"CRITICAL Error importing base modules: {e}", file=sys.stderr)
    print(f"Current sys.path: {sys.path}", file=sys.stderr)
    print("Ensure client_multi_server.py and langchain_mcp_adapters are correctly installed and accessible.", file=sys.stderr)
    sys.exit(1)

# --- LLM Imports (from odin_core_ui.py, add ChatGroq for future flexibility) ---
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq # Added for future use

# --- Load .env (from odin_core_ui.py) ---
# Assumes .env is in the same directory as this app.py (SCRIPT_DIR)
dotenv_path = SCRIPT_DIR / '.env'
load_dotenv(dotenv_path=dotenv_path)
print(f"Loaded .env file from: {dotenv_path.resolve()}")


# --- Backend Logic (run_mcp_task - ALMOST IDENTICAL to odin_core_ui.py) ---
# Only change is adding more robust LLM selection.
async def run_mcp_task(
    agent_type: str,
    main_input: str,
    agent_model_name: str, # This will be used by the enhanced LLM selection
    # MetaGPT specific (from odin_core_ui.py)
    investment: float,
    n_round: int,
    project_name: str,
    code_review: bool,
    run_tests: bool,
    implement: bool,
    inc: bool,
    project_path: str,
    reqa_file: str,
    max_auto_summarize_code: int,
    recover_path: str,
    # Researcher specific (from odin_core_ui.py)
    report_type: str,
    progress: gr.Progress,
    platform: str = "upwork" # From odin_core_ui.py
) -> str:
    progress(0, desc="Initializing...")
    print(f"\n--- Received Task ---")
    print(f"Agent Type: {agent_type}")
    print(f"Main Input: {main_input}")
    print(f"Selected Agent LLM: {agent_model_name}")

    # --- Agent LLM Configuration (Enhanced for flexibility) ---
    agent_llm_instance = None # Renamed from agent_model in odin_core_ui.py
    try:
        progress(0.1, desc=f"Initializing agent LLM ({agent_model_name})...")

        # Define model categories for clarity
        groq_model_list = [
            "llama-3.3-70b-versatile", "llama3-8b-8192", "llama3-70b-8192",
            "mixtral-8x7b-32768", "gemma-7b-it"
        ]
        # OpenRouter specific prefixes/patterns - adjust if needed
        # These are examples; your dropdown might have more direct OpenRouter model names.
        openrouter_model_patterns = ["deepseek/", "meta-llama/", "google/gemini"]
        openai_direct_model_patterns = ["gpt-4", "gpt-3.5"] # For direct OpenAI

        # Logic to select LLM based on agent_model_name
        if agent_model_name in groq_model_list:
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key: raise ValueError("GROQ_API_KEY not set for Groq model.")
            agent_llm_instance = ChatGroq(model_name=agent_model_name, api_key=groq_api_key, temperature=0)
            print(f"Using Groq model: {agent_model_name}")
        elif any(pattern in agent_model_name.lower() for pattern in openrouter_model_patterns):
            # This handles names like "deepseek/deepseek-r1:free"
            base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key: raise ValueError("OPENROUTER_API_KEY not set for OpenRouter model.")
            agent_llm_instance = ChatOpenAI(model=agent_model_name, api_key=api_key, base_url=base_url, temperature=0)
            print(f"Using OpenRouter model: {agent_model_name} via {base_url}")
        elif any(pattern in agent_model_name.lower() for pattern in openai_direct_model_patterns):
            # This handles "gpt-4o", "gpt-3.5-turbo"
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key: raise ValueError("OPENAI_API_KEY not set for direct OpenAI model.")
            agent_llm_instance = ChatOpenAI(model=agent_model_name, api_key=openai_api_key, temperature=0)
            print(f"Using direct OpenAI model: {agent_model_name}")
        else: # Fallback to Ollama (this includes "qwen2.5-coder", "phi4-mini", etc.)
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            agent_llm_instance = ChatOllama(model=agent_model_name, base_url=ollama_base_url, temperature=0)
            print(f"Using Ollama model: {agent_model_name} from {ollama_base_url}")

        if agent_llm_instance is None:
             raise ValueError(f"LLM instance could not be created for model name: {agent_model_name}")
        print(f"Agent LLM initialized: {agent_llm_instance.__class__.__name__}")

    except Exception as e:
        error_msg = f"Error initializing LLM '{agent_model_name}': {e}"
        print(error_msg, file=sys.stderr)
        progress(1, desc="Error")
        return error_msg

    # --- Server Script Directory (from odin_core_ui.py) ---
    server_script_dir = SCRIPT_DIR # Assumes mcp_*.py scripts are in SCRIPT_DIR

    # --- Main Logic (Identical to odin_core_ui.py, but uses agent_llm_instance) ---
    try:
        # --- Handle Browser Use (SSE) ---
        if agent_type == "Browser Use":
            progress(0.2, desc="Configuring Browser Use (SSE)...")
            print("Handling Browser Use task (SSE)...")
            browser_server_url = os.getenv("BROWSER_SERVER_URL", "http://localhost:8000/sse")
            sse_server_config = {"browser-use": {"url": browser_server_url, "transport": "sse"}}
            message_content = f"Use the execute_browseruse tool with task='{main_input}' and platform='{platform}'"
            messages = [HumanMessage(content=message_content)]
            print(f"Expecting SSE server at: {browser_server_url}, platform: {platform}")

            # This section needs to be robust to the MultiServerMCPClient API version
            # For odin_core_ui.py working in 'odin' env, it implies older adapters (<0.1.0)
            # So, the 'async with' should work. If you move this app.py to an env
            # with newer adapters, this part will need the try/except TypeError logic.
            try:
                print("Attempting SSE connection with MultiServerMCPClient as context manager...")
                # Pass sse_server_config here
                async with MultiServerMCPClient(sse_server_config) as client: # This assumes older adapter
                    progress(0.4, desc="Loading tools from SSE server...")
                    try:
                        # Try async version first
                        tools = await client.get_tools()
                    except TypeError:
                        # Fallback to sync version if get_tools is not async
                        tools = client.get_tools()
                    
                    if not tools: return "Error: No tools loaded from browser SSE server."
                    print(f"Browser tools loaded: {[tool.name for tool in tools]}")
                    # Use agent_llm_instance here
                    agent = create_react_agent(agent_llm_instance, tools)
                    agent_response = await agent.ainvoke({"messages": messages})
                    final_result = agent_response["messages"][-1].content
                    progress(1, desc="Task Complete.")
                    return final_result
            except TypeError as te:
                # This block is for newer langchain-mcp-adapters (>= 0.1.0)
                print(f"SSE TypeError (likely new adapter API): {te}. Trying direct instantiation for SSE...")
                mcp_client_for_sse_direct = MultiServerMCPClient(sse_server_config)
                try:
                    try:
                        # Try async version first
                        tools = await mcp_client_for_sse_direct.get_tools()
                    except TypeError:
                        # Fallback to sync version if get_tools is not async
                        tools = mcp_client_for_sse_direct.get_tools()

                    if not tools: return "Error: No tools loaded (direct SSE)."
                    agent = create_react_agent(agent_llm_instance, tools)
                    agent_response = await agent.ainvoke({"messages": messages})
                    final_result = agent_response["messages"][-1].content
                    return final_result
                finally:
                    if hasattr(mcp_client_for_sse_direct, 'aclose'): await mcp_client_for_sse_direct.aclose()
                    print("SSE Client (direct) cleaned up.")
            except ConnectionRefusedError:
                 return f"Error: Connection refused for SSE server at {browser_server_url}."
            except Exception as e_sse:
                print(f"Error during Browser Use SSE: {e_sse}", file=sys.stderr)
                return f"Error during Browser Use SSE: {str(e_sse)}"

        # --- Handle Stdio-based Servers ---
        else:
            progress(0.2, desc=f"Configuring {agent_type} (stdio)...")
            print(f"Handling {agent_type} task (stdio)...")
            # client_wrapper is instantiated with agent_llm_instance
            stdio_client_wrapper = MultiServerMCPClientWrapper(model=agent_llm_instance)
            server_script_name = ""
            server_key = ""
            message_content = ""

            # Logic to determine server_script_name, server_key, message_content
            # (This is identical to odin_core_ui.py)
            if agent_type == "Software Company":
                server_script_name = "mcp_software_company.py"; server_key = "metagpt_server"
                param_details = f"idea='{main_input}'"; # ... (build full param_details for MetaGPT)
                param_details += f", investment={investment}, n_round={n_round}, code_review={code_review}, run_tests={run_tests}, implement={implement}, inc={inc}, max_auto_summarize_code={max_auto_summarize_code}"
                if project_name: param_details += f", project_name='{project_name}'"
                if project_path: param_details += f", project_path='{project_path}'"
                if reqa_file: param_details += f", reqa_file='{reqa_file}'"
                if recover_path: param_details += f", recover_path='{recover_path}'"
                message_content = f"Use the execute_metagpt tool with parameters: {param_details}."
            elif agent_type == "Deep Researcher":
                server_script_name = "mcp_deep_researcher.py"; server_key = "deep_researcher_server"
                message_content = f"Use the execute_deep_research tool with query='{main_input}' and report_type='{report_type}'"
            elif agent_type == "Data Interpreter":
                server_script_name = "mcp_data_interpreter.py"; server_key = "data_interpreter_server"
                message_content = f"Use the execute_data_interpreter tool with requirement='{main_input}'"
            else:
                return f"Error: Unknown agent type '{agent_type}' for stdio."

            server_script_full_path = server_script_dir / server_script_name
            if not server_script_full_path.exists():
                return f"Error: Server script '{server_script_name}' not found at '{server_script_full_path}'."

            # Stdio server_configs (exactly as in odin_core_ui.py)
            server_configs = {
                server_key: {
                    "command": sys.executable,
                    "args": [str(server_script_full_path)],
                    "transport": "stdio",
                    "env": os.environ.copy(),
                }
            }
            print(f"Server Config: {server_configs}")
            print(f"Constructed message for Stdio Agent: {message_content}")

            final_result = "Error: Stdio task did not complete."
            try:
                progress(0.3, desc="Connecting to stdio server...")
                print("Connecting client wrapper to server...")
                await stdio_client_wrapper.connect_to_servers(server_configs)  # Use server_configs instead of stdio_server_configs
                print("Client wrapper connected.")

                if stdio_client_wrapper.agent:
                    progress(0.5, desc="Processing query via stdio...")
                    response = await stdio_client_wrapper.process_query(message_content)
                    final_result = response
                    print("Stdio query processed.")
                else:
                    final_result = "Error: Stdio agent not initialized."
            except Exception as e_stdio:
                print(f"Error during {agent_type} stdio execution: {e_stdio}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                final_result = f"Error during {agent_type} stdio: {str(e_stdio)}"
            finally:
                progress(0.9, desc="Cleaning up stdio client resources...")
                await stdio_client_wrapper.cleanup() # Wrapper's cleanup
                print("Stdio client wrapper cleanup complete.")
                progress(1, desc="Task Complete.")
            return final_result

    except Exception as e_main_task:
        error_msg = f"An unexpected error occurred in run_mcp_task: {e_main_task}"
        print(error_msg, file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        progress(1, desc="Error")
        return error_msg

# --- Gradio UI Definition (Identical to odin_core_ui.py, but with updated LLM choices) ---
def create_ui():
    with gr.Blocks(title="ODIN Core UI") as interface:
        gr.Markdown("# ODIN Core")
        gr.Markdown("Select an agent type, provide inputs, and run the task. \n**Note:** For 'Browser Use', ensure the SSE server is running.")

        with gr.Row():
            with gr.Column(scale=1):
                agent_type_dropdown = gr.Dropdown(
                    label="Agent Type",
                    choices=["Software Company", "Deep Researcher", "Data Interpreter", "Browser Use"],
                    value="Software Company"
                )
                agent_model_name_dropdown = gr.Dropdown( # Changed variable name for clarity
                    label="Agent LLM Model",
                    choices=[
                        # Ollama models
                        "qwen2.5-coder", "phi4-mini", "llama3", "MFDoom/deepseek-r1-tool-calling:7b",
                        # Groq models (direct names)
                        "llama-3.3-70b-versatile", "llama3-8b-8192", "llama3-70b-8192",
                        "mixtral-8x7b-32768", "gemma-7b-it",
                        # OpenRouter models (with prefixes as your logic expects)
                        "deepseek/deepseek-r1:free", "meta-llama/llama-4-scout:free",
                        "google/gemini-2.5-pro-exp-03-25:free",
                        # Direct OpenAI models
                        "gpt-4o", "gpt-3.5-turbo"
                    ],
                    value="qwen2.5-coder", # Default to a local model
                )
                main_input_textbox = gr.Textbox(label="Primary Input (Idea/Query/Requirement/Task)", lines=3) # Changed variable name

                # UI Groups (from odin_core_ui.py)
                with gr.Group(visible=True) as metagpt_group:
                     gr.Markdown("### Software Company Options")
                     metagpt_project_name = gr.Textbox(label="Project Name (Optional)")
                     metagpt_investment = gr.Slider(label="Investment ($)", minimum=1.0, maximum=100.0, value=3.0, step=0.5)
                     metagpt_n_round = gr.Slider(label="Rounds", minimum=1, maximum=20, value=5, step=1)
                     metagpt_code_review = gr.Checkbox(label="Enable Code Review", value=True)
                     metagpt_run_tests = gr.Checkbox(label="Enable Run Tests", value=False)
                     metagpt_implement = gr.Checkbox(label="Enable Implementation", value=True)
                     metagpt_inc = gr.Checkbox(label="Incremental Mode", value=False)
                     metagpt_project_path = gr.Textbox(label="Project Path (Optional)")
                     metagpt_reqa_file = gr.Textbox(label="Requirements File Path (Optional)")
                     metagpt_max_auto_summarize = gr.Number(label="Max Auto Summarize Code", value=0, precision=0)
                     metagpt_recover_path = gr.Textbox(label="Recover Path (Optional)")

                with gr.Group(visible=False) as researcher_group:
                     gr.Markdown("### Deep Researcher Options")
                     researcher_report_type_dropdown = gr.Dropdown( # Changed variable name
                         label="Report Type",
                         choices=['research_report', 'resource_report', 'outline_report', 'custom_report', 'subtopic_report'],
                         value='research_report'
                     )
                with gr.Group(visible=False) as browser_group:
                     gr.Markdown("### Browser Use Options")
                     platform_select_dropdown = gr.Dropdown( # Changed variable name
                         label="Platform", choices=["upwork", "freelancer"], value="upwork"
                     )
                submit_btn = gr.Button("Run Task")
            with gr.Column(scale=2):
                output_textbox = gr.Textbox(label="Output", lines=30, interactive=False) # Changed variable name

        def update_visibility_logic(selected_agent_type_from_ui): # Renamed param
            is_metagpt = selected_agent_type_from_ui == "Software Company"
            is_researcher = selected_agent_type_from_ui == "Deep Researcher"
            is_browser = selected_agent_type_from_ui == "Browser Use"
            return {
                metagpt_group: gr.update(visible=is_metagpt),
                researcher_group: gr.update(visible=is_researcher),
                browser_group: gr.update(visible=is_browser),
                main_input_textbox: gr.update(label={ # Use updated textbox name
                    "Software Company": "Idea", "Deep Researcher": "Query",
                    "Data Interpreter": "Requirement", "Browser Use": "Task"
                }.get(selected_agent_type_from_ui, "Primary Input"))
            }
        agent_type_dropdown.change( # Use updated dropdown name
            fn=update_visibility_logic,
            inputs=[agent_type_dropdown], # Use updated dropdown name
            outputs=[metagpt_group, researcher_group, browser_group, main_input_textbox] # Use updated textbox name
        )

        # Wrapper function that Gradio calls (from odin_core_ui.py)
        async def run_task_for_ui_wrapper( # Renamed from run_task_with_progress in odin_core_ui.py
            agent_type_val, main_input_val, agent_model_val,
            investment_val, n_round_val, project_name_val, code_review_val, run_tests_val, implement_val,
            inc_val, project_path_val, reqa_file_val, max_auto_summarize_val, recover_path_val,
            report_type_val, platform_val,
            progress=gr.Progress(track_tqdm=True)
        ):
            sys.stdout.reconfigure(line_buffering=True)
            sys.stderr.reconfigure(line_buffering=True)
            print("Gradio UI: Submit button clicked. Calling backend run_mcp_task...")
            
            result = await run_mcp_task( # Call the main backend logic
                agent_type=agent_type_val, main_input=main_input_val, agent_model_name=agent_model_val,
                investment=investment_val, n_round=n_round_val, project_name=project_name_val,
                code_review=code_review_val, run_tests=run_tests_val, implement=implement_val,
                inc=inc_val, project_path=project_path_val, reqa_file=reqa_file_val,
                max_auto_summarize_code=max_auto_summarize_val, recover_path=recover_path_val,
                report_type=report_type_val, platform=platform_val,
                progress=progress
            )
            print(f"Gradio UI: Result from backend: {result!r}")
            return result

        submit_btn.click(
            fn=run_task_for_ui_wrapper,
            inputs=[ # Ensure this list matches params of run_task_for_ui_wrapper IN ORDER
                agent_type_dropdown, main_input_textbox, agent_model_name_dropdown, # Use updated component names
                metagpt_investment, metagpt_n_round, metagpt_project_name, metagpt_code_review, metagpt_run_tests, metagpt_implement,
                metagpt_inc, metagpt_project_path, metagpt_reqa_file, metagpt_max_auto_summarize, metagpt_recover_path,
                researcher_report_type_dropdown, platform_select_dropdown, # Use updated component names
            ],
            outputs=output_textbox, # Use updated component name
        )
    return interface

# --- Main Application Runner (from odin_core_ui.py) ---
if __name__ == '__main__':
    print(f"Launching ODIN Core UI (based on {Path(__file__).name})...")
    print(f"Script directory (SCRIPT_DIR): {SCRIPT_DIR}")
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    print(f"Python executable: {sys.executable}")
    print(f"PYTHONPATH (at app start): {os.environ.get('PYTHONPATH', 'Not set')}")

    gradio_app_instance = create_ui()
    gradio_app_instance.launch(server_name="127.0.0.1", server_port=7860, share=False)
    # Original launch from odin_core_ui.py was just demo.launch()
    # Specifying server_name and port is good practice for clarity.