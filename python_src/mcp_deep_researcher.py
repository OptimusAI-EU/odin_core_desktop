import asyncio
import logging
import os # Added for path operations
import re # Added for filename sanitization
from pathlib import Path # Added for path handling
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP # Import FastMCP
# Removed unused LLM imports from server file
# from langchain_ollama import ChatOllama
# from langchain_openai import ChatOpenAI
# from langchain_groq import ChatGroq
from typing import Dict
from gpt_researcher import GPTResearcher

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP
mcp = FastMCP("gpt_researcher_executor")

def sanitize_filename(name):
    """Removes or replaces characters unsafe for filenames."""
    # Remove characters that are definitely problematic
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    # Replace spaces with underscores (optional, but common)
    name = name.replace(" ", "_")
    # Limit length (optional)
    return name[:100] # Limit to 100 chars

@mcp.tool(
    name="execute_deep_research",
    description="Conducts in-depth research on a given query using GPTResearcher and generates and saves a report." # Updated description
)
async def execute_deep_research(query: str, report_type: str = "research_report") -> str:
    """
    Performs research using GPTResearcher and saves the report to a markdown file.

    Args:
        query: The research query string.
        report_type: The type of report to generate (e.g., 'research_report', 'resource_report', 'outline_report').
                     Defaults to 'research_report'.

    Returns:
        A confirmation message indicating the path to the saved report, or an error message.
    """
    logger.info(f"Received research request: query='{query}', report_type='{report_type}'")
    try:
        # Initialize the researcher
        researcher = GPTResearcher(query=query, report_type=report_type, config_path=None)

        # Conduct research
        logger.info("Conducting research...")
        await researcher.conduct_research()
        logger.info("Research complete. Writing report...")

        # Get the report content as a string
        report_content = await researcher.write_report()
        logger.info("Report content generated.")

        # --- Add file saving logic ---
        try:
            output_dir = Path("./outputs") # Define output directory relative to server script
            output_dir.mkdir(parents=True, exist_ok=True) # Create directory if it doesn't exist

            # Create a safe filename from the query
            safe_filename_base = sanitize_filename(query)
            output_filename = f"{safe_filename_base}_{report_type}.md"
            output_filepath = output_dir / output_filename

            # Write the report content to the file
            output_filepath.write_text(report_content, encoding='utf-8')
            logger.info(f"Report successfully saved to: {output_filepath}")
            # Return the path instead of the full report content
            return f"Report successfully generated and saved to: {output_filepath}"

        except Exception as e_save:
            logger.error(f"Failed to save report file: {str(e_save)}", exc_info=True)
            # Return the report content anyway if saving failed, along with an error
            return f"Error saving report file: {str(e_save)}. Report content:\n\n{report_content}"
        # --- End file saving logic ---

    except Exception as e:
        logger.error(f"Deep research execution failed: {str(e)}", exc_info=True)
        return f"Error during research: {str(e)}"

if __name__ == "__main__":
    # Run the MCP server using stdio transport
    print("Starting GPT Researcher MCP server via stdio...")
    mcp.run(transport="stdio")
    
    print("GPT Researcher MCP server stopped.")


