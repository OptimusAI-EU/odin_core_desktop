# File: odin_core_desktop/python_src/tests/test_odin_agents.py
import asyncio
import sys
from pathlib import Path
import os

# --- Add python_src to sys.path to find odin_core and client_multi_server ---
# This assumes 'tests' is a subdir of 'python_src'
PYTHON_SRC_DIR = Path(__file__).resolve().parent.parent
if str(PYTHON_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC_DIR))
print(f"Test Script: Added to sys.path: {PYTHON_SRC_DIR}")
# ---

try:
    from odin_core import run_odin_task, create_llm_instance # Key function to test
    print("Test Script: Successfully imported from odin_core.")
except ImportError as e:
    print(f"Test Script: CRITICAL Error importing from odin_core: {e}", file=sys.stderr)
    print(f"Test Script: Current sys.path: {sys.path}", file=sys.stderr)
    sys.exit(1)

# Dummy progress callback for testing
def dummy_progress_callback(value: float, description: str):
    print(f"TEST PROGRESS: {value*100:.0f}% - {description}")

# In odin_core_desktop/python_src/tests/test_odin_agents.py

async def test_software_company():
    print("\n--- Testing SoftwareCompanyAgent ---")
    try:
        llm_model_name_for_test = os.getenv("TEST_LLM_MODEL", "qwen2.5-coder")
        print(f"Test Script: Using LLM: {llm_model_name_for_test} for Software Company test.")

        result = await run_odin_task(
            agent_type="Software Company",
            main_input="Create a very simple Python script that prints 'Hello, ODIN!'",
            agent_model_name=llm_model_name_for_test,
            progress_callback=dummy_progress_callback,
            investment=1.0,
            n_round=3,
            project_name="odin_hello_test_v2", # Changed name for fresh run
            code_review=False,
            run_tests=False,
            implement=True,
            inc=False, project_path="", reqa_file="", max_auto_summarize_code=0, recover_path=None
        )
        print("\n--- SoftwareCompanyAgent Test Result (Full Content) ---")
        print(f"RESULT_STRING_START>>>\n{result}\n<<<RESULT_STRING_END") # Print the full result clearly

        # Now the assertion
        assert "error" not in result.lower(), f"Software Company task reported an error. Full result: {result!r}" # Include result in assertion message

        # ... (rest of your assertions, like checking workspace) ...

    except Exception as e:
        print(f"Test Script: SoftwareCompanyAgent test FAILED with exception: {e}")
        import traceback
        traceback.print_exc()

    except Exception as e:
        print(f"Test Script: SoftwareCompanyAgent test FAILED: {e}")
        import traceback
        traceback.print_exc()

async def test_deep_researcher():
    print("\n--- Testing DeepResearcherAgent ---")
    try:
        llm_model_name_for_test = os.getenv("TEST_LLM_MODEL", "qwen2.5-coder")
        print(f"Test Script: Using LLM: {llm_model_name_for_test} for Deep Researcher test.")

        result = await run_odin_task(
            agent_type="Deep Researcher",
            main_input="What are the recent advancements in AI-powered code generation?",
            agent_model_name=llm_model_name_for_test,
            progress_callback=dummy_progress_callback,
            # Researcher specific kwargs
            report_type="research_report"
            # Other params will use defaults or be empty strings if not provided
        )
        print("\n--- DeepResearcherAgent Test Result ---")
        print(result)
        assert "error" not in result.lower(), "Deep Researcher task reported an error."
        # Add assertion: check if a report file was mentioned or created
        # For now, just checking for "error".
        # report_dir = PYTHON_SRC_DIR / "workspace" # Or wherever DeepResearcher saves reports
        # Found_report = any(f.name.endswith(".md") or f.name.endswith(".pdf") for f in report_dir.iterdir() if f.is_file())
        # assert Found_report, "No report file found for Deep Researcher test."
        # print(f"Test Script: Deep Researcher test passed. Report content printed above.")

    except Exception as e:
        print(f"Test Script: DeepResearcherAgent test FAILED: {e}")
        import traceback
        traceback.print_exc()


# Add more test functions for DataInterpreterAgent and BrowserUseAgent if desired

async def main_tests():
    # Ensure any required servers for BrowserUse are running if you test that
    # For stdio, mcp_*.py scripts will be launched by odin_core
    
    # Run tests sequentially
    await test_software_company()
    # await test_deep_researcher() # Uncomment to test
    # await test_data_interpreter()
    # await test_browser_use()

if __name__ == "__main__":
    print("Test Script: Starting ODIN Core agent tests...")
    # Ensure event loop policy is set for Windows if needed for subprocesses,
    # though usually not necessary for basic asyncio.run()
    # if sys.platform == "win32":
    # asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main_tests())
    print("Test Script: All ODIN Core agent tests finished.")