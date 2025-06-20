# File: app_ui.py
# (Place this in odin_core_desktop/python_src/, sibling to odin_core.py)

import asyncio
import os
import sys
import gradio as gr
from pathlib import Path

# --- Path Setup (ensure odin_core.py is importable) ---
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# --- Import from odin_core ---
try:
    # Import the main execution function and SCRIPT_DIR if needed for .env by odin_core
    from odin_core import run_odin_task, SCRIPT_DIR as ODIN_CORE_SCRIPT_DIR
    print("Successfully imported from odin_core module.")
except ImportError as e:
    print(f"CRITICAL Error importing from odin_core: {e}", file=sys.stderr)
    print("Ensure odin_core.py is in the same directory or PYTHONPATH is set.", file=sys.stderr)
    sys.exit(1)

# --- Main Gradio UI and App Logic ---

# --- Gradio UI Definition (Mostly from your app.py) ---
def create_gradio_ui(): # Renamed from create_ui to avoid conflict if you run both files
    with gr.Blocks(title="ODIN Core UI (Powered by odin_core.py)") as interface:
        gr.Markdown("# ODIN Core Interface")
        gr.Markdown("Select an agent, provide inputs, configure LLM, and run the task.")

        with gr.Row():
            with gr.Column(scale=1):
                agent_type_dropdown = gr.Dropdown(
                    label="Agent Type",
                    choices=["Software Company", "Deep Researcher", "Data Interpreter", "Browser Use"],
                    value="Software Company"
                )
                agent_model_name_dropdown = gr.Dropdown(
                    label="Agent LLM Model",
                    choices=[
                        "qwen2.5-coder", "phi4-mini", "llama3", "MFDoom/deepseek-r1-tool-calling:7b",
                        "llama-3.3-70b-versatile", "llama3-8b-8192", "llama3-70b-8192",
                        "mixtral-8x7b-32768", "gemma-7b-it",
                        "deepseek/deepseek-r1:free", "meta-llama/llama-4-scout:free",
                        "google/gemini-2.5-pro-exp-03-25:free",
                        "gpt-4o", "gpt-3.5-turbo"
                    ],
                    value="qwen2.5-coder",
                )
                main_input_textbox = gr.Textbox(label="Primary Input", lines=3)

                with gr.Group(visible=True) as metagpt_group:
                     gr.Markdown("### Software Company Options")
                     metagpt_project_name = gr.Textbox(label="Project Name", value="")
                     metagpt_investment = gr.Slider(label="Investment ($)", minimum=1.0, maximum=100.0, value=3.0, step=0.5)
                     metagpt_n_round = gr.Slider(label="Rounds", minimum=1, maximum=20, value=5, step=1)
                     # ... (all other metagpt params from your app.py) ...
                     metagpt_code_review = gr.Checkbox(label="Code Review", value=True)
                     metagpt_run_tests = gr.Checkbox(label="Run Tests", value=False)
                     metagpt_implement = gr.Checkbox(label="Implement", value=True)
                     metagpt_inc = gr.Checkbox(label="Incremental", value=False)
                     metagpt_project_path = gr.Textbox(label="Project Path", value="")
                     metagpt_reqa_file = gr.Textbox(label="Requirements File", value="")
                     metagpt_max_auto_summarize = gr.Number(label="Max Auto Summarize", value=0, precision=0)
                     metagpt_recover_path = gr.Textbox(label="Recover Path", value="")


                with gr.Group(visible=False) as researcher_group:
                     gr.Markdown("### Deep Researcher Options")
                     researcher_report_type_dropdown = gr.Dropdown(
                         label="Report Type",
                         choices=['research_report', 'resource_report', 'outline_report', 'custom_report', 'subtopic_report'],
                         value='research_report'
                     )
                with gr.Group(visible=False) as browser_group:
                     gr.Markdown("### Browser Use Options")
                     platform_select_dropdown = gr.Dropdown(
                         label="Platform", choices=["upwork", "freelancer"], value="upwork"
                     )
                submit_btn = gr.Button("Run Task")
            with gr.Column(scale=2):
                output_textbox = gr.Textbox(label="Output", lines=30, interactive=False)

        def update_visibility_logic(selected_agent_type_from_ui):
            # ... (same visibility logic as your app.py) ...
            is_metagpt = selected_agent_type_from_ui == "Software Company"
            is_researcher = selected_agent_type_from_ui == "Deep Researcher"
            is_browser = selected_agent_type_from_ui == "Browser Use"
            return {
                metagpt_group: gr.update(visible=is_metagpt),
                researcher_group: gr.update(visible=is_researcher),
                browser_group: gr.update(visible=is_browser),
                main_input_textbox: gr.update(label={
                    "Software Company": "Idea", "Deep Researcher": "Query",
                    "Data Interpreter": "Requirement", "Browser Use": "Task"
                }.get(selected_agent_type_from_ui, "Primary Input"))
            }
        agent_type_dropdown.change(
            fn=update_visibility_logic,
            inputs=[agent_type_dropdown],
            outputs=[metagpt_group, researcher_group, browser_group, main_input_textbox]
        )

        async def handle_submit_task(
            # These parameters MUST match the 'inputs' list of submit_btn.click IN ORDER
            agent_type_val, main_input_val, agent_model_name_val,
            # MetaGPT args
            investment_val, n_round_val, project_name_val, code_review_val, run_tests_val, implement_val,
            inc_val, project_path_val, reqa_file_val, max_auto_summarize_val, recover_path_val,
            # Researcher args
            report_type_val,
            # Browser args
            platform_val,
            progress=gr.Progress(track_tqdm=True) # Gradio provides this
        ):
            sys.stdout.reconfigure(line_buffering=True)
            sys.stderr.reconfigure(line_buffering=True)
            print("APP_UI: Submit button clicked. Preparing to call odin_core.run_odin_task...")

            # Prepare kwargs for agent-specific parameters
            agent_kwargs = {
                "investment": investment_val,
                "n_round": n_round_val,
                "project_name": project_name_val,
                "code_review": code_review_val,
                "run_tests": run_tests_val,
                "implement": implement_val,
                "inc": inc_val,
                "project_path": project_path_val,
                "reqa_file": reqa_file_val,
                "max_auto_summarize_code": max_auto_summarize_val,
                "recover_path": recover_path_val,
                "report_type": report_type_val,
                "platform": platform_val,
            }
            
            # Define the progress callback for odin_core
            def progress_updater(value, desc):
                progress(value, desc=desc)

            result = await run_odin_task(
                agent_type=agent_type_val,
                main_input=main_input_val,
                agent_model_name=agent_model_name_val,
                progress_callback=progress_updater, # Pass the callback
                **agent_kwargs # Pass all other params as kwargs
            )
            print(f"APP_UI: Result from odin_core: {result!r}")
            return result

        submit_btn.click(
            fn=handle_submit_task,
            inputs=[
                agent_type_dropdown, main_input_textbox, agent_model_name_dropdown,
                metagpt_investment, metagpt_n_round, metagpt_project_name, metagpt_code_review, metagpt_run_tests, metagpt_implement,
                metagpt_inc, metagpt_project_path, metagpt_reqa_file, metagpt_max_auto_summarize, metagpt_recover_path,
                researcher_report_type_dropdown, platform_select_dropdown,
            ],
            outputs=output_textbox,
        )
    return interface

# --- Main Application Runner ---
if __name__ == "__main__":
    print(f"Launching Gradio UI from {Path(__file__).name}...")
    # Ensure odin_core.py also has its .env loaded correctly if it doesn't do it itself
    # SCRIPT_DIR here refers to app_ui.py's directory
    print(f"APP_UI Script directory: {SCRIPT_DIR}")
    if ODIN_CORE_SCRIPT_DIR: # Check if it was imported
         print(f"ODIN_CORE Script directory (for .env): {ODIN_CORE_SCRIPT_DIR}")

    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    gradio_app_instance = create_gradio_ui()
    gradio_app_instance.launch(server_name="127.0.0.1", server_port=7860, share=False)