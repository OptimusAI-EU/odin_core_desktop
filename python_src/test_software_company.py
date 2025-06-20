"""Test the Software Company agent functionality."""
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
import sys

# Add parent directory to Python path for imports
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(SCRIPT_DIR))

# Load environment variables
from odin_core import create_agent  # Import after adding to path
load_dotenv(SCRIPT_DIR / '.env')


def create_progress_callback():
    """Create a standard progress callback."""
    def progress_callback(value, desc=""):
        print(f"Progress: {value*100:.0f}% - {desc}")
    return progress_callback


async def test_software_company():
    """Test the Software Company agent."""
    print("\nTesting Software Company agent...")
    llm = ChatOllama(model="qwen2.5-coder")
    agent = create_agent("Software Company", llm)
    agent.set_progress_callback(create_progress_callback())
    
    try:
        result = await agent.execute(
            task="Write a simple Python script that calculates factorial using both recursive and iterative approaches"
        )
        print("\nSoftware Company Results:")
        print("=====================")
        print(result)
        
        # Validate result contains key indicators of success
        success = (
            isinstance(result, str) and
            len(result) > 0 and
            any(keyword in result.lower() for keyword in 
                ['factorial', 'recursive', 'iterative', 'implementation', 'code'])
        )
        
        return success
    except Exception as e:
        print(f"Error in Software Company test: {e}")
        return False
    finally:
        if hasattr(agent, 'cleanup'):
            await agent.cleanup()


async def main():
    """Run all Software Company agent tests."""
    success = await test_software_company()
    print(f"\nOverall test {'succeeded' if success else 'failed'}")
    return success


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nReceived interrupt signal, shutting down...")
    finally:
        print("\nTest completed, exiting...")
