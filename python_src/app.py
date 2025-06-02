# File: python_src/app.py
import asyncio
import os
import sys
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv
import time # Keep for potential delays if needed

# --- Path Setup for Packaged App ---
# When packaged, __file__ will point to app.py inside the app's resources.
# We assume client_multi_server.py and langchain_mcp_adapters/ are in the same directory.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# The original script added grandparent_dir. If client_multi_server and
# langchain_mcp_adapters are directly within SCRIPT_DIR (python_src),
# this might not be necessary or could be simplified.
# For this example, let's assume they are directly in SCRIPT_DIR.
# If client_multi_server.py is in SCRIPT_DIR, and langchain_mcp_adapters is a folder in SCRIPT_DIR,
# the single sys.path.insert(0, str(SCRIPT_DIR)) should be enough.

try:
    from client_multi_server import MultiServerMCPClientWrapper
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langgraph.prebuilt import create_react_agent
    from langchain_core.messages import HumanMessage
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Attempted to load from SCRIPT_DIR: {SCRIPT_DIR}")
    print("Ensure client_multi_server.py and langchain_mcp_adapters directory are in the same directory as app.py.")
    sys.exit(1)

# Choose your LLM integration (used for the agent)
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

# Load .env from the same directory as app.py
dotenv_path = SCRIPT_DIR / '.env' # Changed from Path(__file__).parent
load_dotenv(dotenv_path=dotenv_path)
print(f"Loaded .env file from: {dotenv_path}")


# --- Backend Logic ---
# (Your run_mcp_task function remains largely the same)
# Important: Ensure paths to mcp_*.py scripts are correct
async def run_mcp_task(
    agent_type: str,
    main_input: str,
    agent_model_name: str,
    # MetaGPT specific
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
    # Researcher specific
    report_type: str,
    # Progress tracker argument
    progress: gr.Progress,
    # Platform argument
    platform: str = "upwork"
) -> str:
    progress(0, desc="Initializing...")

    print(f"\n--- Received Task ---")
    print(f"Agent Type: {agent_type}")
    print(f"Main Input: {main_input}")
    print(f"Agent Model: {agent_model_name}")

    try:
        progress(0.1, desc=f"Initializing agent LLM ({agent_model_name})...")
        if any(name in agent_model_name.lower() for name in ["deepseek", "llama-4", "gemini"]): # Corrected "llama4" to "llama-4"
            base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY not set in environment")
            agent_model = ChatOpenAI(
                model=agent_model_name,
                api_key=api_key,
                base_url=base_url
            )
        else:
            agent_model = ChatOllama(model=agent_model_name)
        print(f"Using agent model: {agent_model.__class__.__name__} (model={agent_model_name})")
    except Exception as e:
        error_msg = f"Error initializing the agent's language model '{agent_model_name}': {e}"
        print(error_msg)
        return error_msg

    # SCRIPT_DIR is defined at the top of the file
    server_script_dir = SCRIPT_DIR # Assume servers are in the same directory as app.py

    try:
        if agent_type == "Browser Use":
            progress(0.2, desc="Configuring Browser Use (SSE)...")
            print("Handling Browser Use task (SSE)...")
            browser_server_url = os.getenv("BROWSER_SERVER_URL", "http://localhost:8000/sse")
            server_configs = {
                "browser-use": {
                    "url": browser_server_url,
                    "transport": "sse",
                }
            }
            message_content = f"Use the execute_browseruse tool with task='{main_input}' and platform='{platform}'"
            messages = [HumanMessage(content=message_content)]
            
            print(f"Expecting SSE server at: {server_configs['browser-use']['url']}")
            print(f"Platform: {platform}")
            print(f"Constructed message: {message_content}")

            try:
                progress(0.3, desc="Connecting to SSE server...")
                async with MultiServerMCPClient(server_configs) as client:
                    progress(0.4, desc="Loading tools from SSE server...")
                    print("Browser SSE Client initialized.")
                    tools = client.get_tools()
                    if not tools:
                        return "Error: No tools loaded from the browser SSE server. Is it running?"

                    progress(0.5, desc="Creating agent...")
                    print(f"Tools loaded: {[tool.name for tool in tools]}")
                    agent = create_react_agent(agent_model, tools)

                    progress(0.6, desc="Invoking browser task (may take time)...")
                    print("Browser agent created. Invoking...")
                    agent_response = await agent.ainvoke({"messages": messages})
                    print("Browser agent invocation complete.")

                    progress(0.9, desc="Processing response...")
                    final_result = agent_response["messages"][-1].content
                    progress(1, desc="Task Complete.")
                    return final_result

            except ConnectionRefusedError:
                 return f"Error: Connection refused. Is the browser-use SSE server running at {server_configs['browser-use']['url']}?"
            except Exception as e:
                print(f"Error during Browser Use SSE execution: {e}")
                return f"Error during Browser Use SSE execution: {str(e)}"

        else:
            progress(0.2, desc=f"Configuring {agent_type} (stdio)...")
            print(f"Handling {agent_type} task (stdio)...")
            client_wrapper = MultiServerMCPClientWrapper(model=agent_model)
            server_script_name = ""
            server_key = ""
            message_content = ""

            if agent_type == "Software Company":
                server_script_name = "mcp_software_company.py"
                server_key = "metagpt_server"
                param_details = f"idea='{main_input}'"
                if project_name: param_details += f", project_name='{project_name}'"
                if project_path: param_details += f", project_path='{project_path}'"
                if reqa_file: param_details += f", reqa_file='{reqa_file}'"
                if recover_path: param_details += f", recover_path='{recover_path}'"
                param_details += f", investment={investment}"
                param_details += f", n_round={n_round}"
                param_details += f", code_review={code_review}"
                param_details += f", run_tests={run_tests}"
                param_details += f", implement={implement}"
                param_details += f", inc={inc}"
                param_details += f", max_auto_summarize_code={max_auto_summarize_code}"
                message_content = f"Use the execute_metagpt tool with parameters: {param_details}."
            elif agent_type == "Deep Researcher":
                server_script_name = "mcp_deep_researcher.py"
                server_key = "deep_researcher_server"
                message_content = f"Use the execute_deep_research tool with query='{main_input}' and report_type='{report_type}'"
            elif agent_type == "Data Interpreter":
                server_script_name = "mcp_data_interpreter.py"
                server_key = "data_interpreter_server"
                message_content = f"Use the execute_data_interpreter tool with requirement='{main_input}'"
            else:
                return f"Error: Unknown agent type '{agent_type}'"

            server_script_full_path = server_script_dir / server_script_name
            if not server_script_full_path.exists():
                return f"Error: Server script '{server_script_name}' not found in '{server_script_dir}'. Searched: {server_script_full_path}"

            python_executable = sys.executable # Use the same python that's running this script
            server_configs = {
                server_key: {
                    "command": python_executable, # Use determined python executable
                    "args": [str(server_script_full_path)],
                    "transport": "stdio",
                    "env": os.environ.copy(),
                }
            }
            print(f"Server Config: {server_configs}")
            print(f"Constructed message: {message_content}")

            client_wrapper = MultiServerMCPClientWrapper(model=agent_model)
            final_result = "Error: Task did not complete."
            try:
                progress(0.3, desc="Connecting to server (starting subprocess)...")
                print("Connecting client wrapper to server...")
                await client_wrapper.connect_to_servers(server_configs)
                print("Client wrapper connected.")

                if client_wrapper.agent:
                    progress(0.5, desc="Processing query (may take time)...")
                    print("Processing query...")
                    response = await client_wrapper.process_query(message_content)
                    print("Query processed.")
                    final_result = response
                else:
                    final_result = "Error: Agent could not be initialized after connecting to the server."

            except Exception as e:
                 print(f"Error during {agent_type} stdio execution: {e}")
                 final_result = f"Error during {agent_type} stdio execution: {str(e)}"
            finally:
                progress(0.9, desc="Cleaning up resources...")
                print("Cleaning up client wrapper resources...")
                await client_wrapper.cleanup()
                print("Cleanup complete.")
                progress(1, desc="Task Complete.")
            return final_result
    except Exception as e:
        error_msg = f"An unexpected error occurred in run_mcp_task: {str(e)}"
        print(error_msg)
        return error_msg

# --- Gradio UI Definition (remains the same) ---
def create_ui():
    with gr.Blocks(title="ODIN Core UI") as interface: # Updated title
        gr.Markdown("# ODIN Core") # Updated title
        gr.Markdown("Select an agent type, provide inputs, and run the task. \n**Note:** For 'Browser Use', ensure the SSE server (`python mcp_browseruse.py`) is running manually beforehand.") # Simplified note

        with gr.Row():
            with gr.Column(scale=1):
                agent_type = gr.Dropdown(
                    label="Agent Type",
                    choices=["Software Company", "Deep Researcher", "Data Interpreter", "Browser Use"],
                    value="Software Company"
                )
                agent_model_name = gr.Dropdown(
                    label="Agent LLM Model (for UI Agent)",
                    choices=[
                        # Local Ollama models
                        "qwen2.5-coder", "phi4-mini", "llama3",
                        # OpenRouter models
                        "deepseek/deepseek-r1:free", "meta-llama/llama-4-scout:free", "google/gemini-2.5-pro-exp-03-25:free",
                        # Other models
                        "gpt-4o", "gpt-3.5-turbo"
                    ],
                    value="qwen2.5-coder"
                )
                main_input = gr.Textbox(label="Primary Input (Idea/Query/Requirement/Task)", lines=3)

                with gr.Group(visible=True) as metagpt_group:
                     gr.Markdown("### Software Company Options")
                     metagpt_project_name = gr.Textbox(label="Project Name (Optional)")
                     metagpt_investment = gr.Slider(label="Investment", minimum=1.0, maximum=10.0, value=3.0, step=0.5)
                     metagpt_n_round = gr.Slider(label="Rounds", minimum=1, maximum=20, value=5, step=1)
                     metagpt_code_review = gr.Checkbox(label="Enable Code Review", value=True)
                     metagpt_run_tests = gr.Checkbox(label="Enable Run Tests", value=False)
                     metagpt_implement = gr.Checkbox(label="Enable Implementation", value=True)
                     metagpt_inc = gr.Checkbox(label="Incremental Mode", value=False)
                     metagpt_project_path = gr.Textbox(label="Project Path (Optional, overrides default)")
                     metagpt_reqa_file = gr.Textbox(label="Requirements File Path (Optional)")
                     metagpt_max_auto_summarize = gr.Number(label="Max Auto Summarize Code (0=disable)", value=0, precision=0)
                     metagpt_recover_path = gr.Textbox(label="Recover Path (Optional)")

                with gr.Group(visible=False) as researcher_group:
                     gr.Markdown("### Deep Researcher Options")
                     researcher_report_type = gr.Dropdown(
                         label="Report Type",
                         choices=['research_report', 'resource_report', 'outline_report', 'custom_report', 'subtopic_report'],
                         value='research_report'
                     )

                with gr.Group(visible=False) as browser_group:
                     gr.Markdown("### Browser Use Options")
                     gr.Markdown("*(Configuration is now handled by environment variables in `.env`)*")
                     platform_select = gr.Dropdown(
                         label="Platform",
                         choices=["upwork", "freelancer"],
                         value="upwork"
                     )
                submit_btn = gr.Button("Run Task")

            with gr.Column(scale=2):
                output = gr.Textbox(label="Output", lines=25, interactive=False)

        def update_visibility(selected_agent):
            is_metagpt = selected_agent == "Software Company"
            is_researcher = selected_agent == "Deep Researcher"
            is_browser = selected_agent == "Browser Use"
            return {
                metagpt_group: gr.update(visible=is_metagpt),
                researcher_group: gr.update(visible=is_researcher),
                browser_group: gr.update(visible=is_browser),
                main_input: gr.update(label="Idea" if is_metagpt else \
                                            "Query" if is_researcher else \
                                            "Requirement" if selected_agent == "Data Interpreter" else \
                                            "Task" if is_browser else "Primary Input")
            }

        agent_type.change(
            fn=update_visibility,
            inputs=[agent_type],
            outputs=[metagpt_group, researcher_group, browser_group, main_input]
        )

        async def run_task_with_progress(
            agent_type_val, main_input_val, agent_model_val,
            investment_val, n_round_val, project_name_val, code_review_val, run_tests_val, implement_val,
            inc_val, project_path_val, reqa_file_val, max_auto_summarize_val, recover_path_val,
            report_type_val,
            platform_val,
            progress=gr.Progress(track_tqdm=True)
        ):
            result = await run_mcp_task(
                agent_type=agent_type_val, main_input=main_input_val, agent_model_name=agent_model_val,
                investment=investment_val, n_round=n_round_val, project_name=project_name_val,
                code_review=code_review_val, run_tests=run_tests_val, implement=implement_val,
                inc=inc_val, project_path=project_path_val, reqa_file=reqa_file_val,
                max_auto_summarize_code=max_auto_summarize_val, recover_path=recover_path_val,
                report_type=report_type_val, platform=platform_val,
                progress=progress
            )
            return result

        submit_btn.click(
            fn=run_task_with_progress,
            inputs=[
                agent_type, main_input, agent_model_name,
                metagpt_investment, metagpt_n_round, metagpt_project_name, metagpt_code_review, metagpt_run_tests, metagpt_implement,
                metagpt_inc, metagpt_project_path, metagpt_reqa_file, metagpt_max_auto_summarize, metagpt_recover_path,
                researcher_report_type,
                platform_select,
            ],
            outputs=output,
        )
    return interface

if __name__ == '__main__':
    print("Launching ODIN Core Test UI (Python Server)...")
    print(f"Script directory (SCRIPT_DIR): {SCRIPT_DIR}")
    print(f"Attempting to load .env from: {dotenv_path}")
    # print(f"BROWSER_LLM_PROVIDER: {os.getenv('BROWSER_LLM_PROVIDER')}") # Example debug
    
    # Ensure Gradio listens on a port accessible to Electron
    # 0.0.0.0 makes it accessible from any IP on the machine, localhost included.
    # You can set a specific port if needed using server_port=xxxx
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860) # Default Gradio port