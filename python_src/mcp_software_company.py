import asyncio
import os
import logging
from pathlib import Path
import agentops
from mcp.server.fastmcp import FastMCP
from metagpt.const import CONFIG_ROOT
from metagpt.utils.project_repo import ProjectRepo
from metagpt.roles import (
    Architect,
    Engineer,
    ProductManager,
    ProjectManager,
    QaEngineer,
)
from metagpt.team import Team
from metagpt.config2 import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("metagpt_executor")

async def generate_project(
    idea: str,
    investment: float = 3.0,
    n_round: int = 5,
    code_review: bool = True,
    run_tests: bool = False,
    implement: bool = True,
    project_name: str = "",
    inc: bool = False,
    project_path: str = "",
    reqa_file: str = "",
    max_auto_summarize_code: int = 0,
    recover_path: str = None
) -> str:
    """Generate a software project using MetaGPT with all original parameters"""
    try:
        # Update config if project path exists
        if project_path:
            config.update_via_cli(project_path, project_name, inc, reqa_file, max_auto_summarize_code)

        # Initialize team based on recover path
        if recover_path:
            stg_path = Path(recover_path)
            if not stg_path.exists() or not str(stg_path).endswith("team"):
                raise FileNotFoundError(f"{recover_path} not exists or not endswith `team`")
            company = Team.deserialize(stg_path=stg_path)
        else:
            company = Team()
            
            # Base team members
            team = [
                ProductManager(),
                Architect(),
                ProjectManager(),
            ]
            
            # Add Engineer if implementation or code review is needed
            if implement or code_review:
                team.append(Engineer(n_borg=5, use_code_review=code_review))
            
            # Add QA Engineer if tests are enabled
            if run_tests:
                team.append(QaEngineer())
                if n_round < 8:
                    n_round = 8  # Minimum rounds needed for QA
                    
            company.hire(team)

        company.invest(investment)
        company.run_project(idea)
        await company.run(n_round=n_round)
        
        return "Project generation completed successfully"
    except Exception as e:
        logger.error(f"Project generation failed: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool(
    name="execute_metagpt",
    description="Executes MetaGPT project generation with full parameter support"
)
async def execute_metagpt(
    idea: str,
    investment: float = 3.0,
    n_round: int = 5,
    code_review: bool = True,
    run_tests: bool = False,
    implement: bool = True,
    project_name: str = "",
    inc: bool = False,
    project_path: str = "",
    reqa_file: str = "",
    max_auto_summarize_code: int = 0,
    recover_path: str = None
) -> str:
    """
    MCP Tool for MetaGPT project generation
    
    Args:
        idea: Your innovative idea, such as 'Create a 2048 game'
        investment: Dollar amount to invest in the AI company
        n_round: Number of rounds for the simulation
        code_review: Whether to use code review
        run_tests: Whether to enable QA for adding & running tests
        implement: Enable or disable code implementation
        project_name: Unique project name, such as 'game_2048'
        inc: Incremental mode for existing repo cooperation
        project_path: Directory path for incremental project
        reqa_file: Source file for rewriting QA code
        max_auto_summarize_code: Max auto summarize iterations (-1 for unlimited)
        recover_path: Path to recover project from serialized storage
    """
    return await generate_project(
        idea=idea,
        investment=investment,
        n_round=n_round,
        code_review=code_review,
        run_tests=run_tests,
        implement=implement,
        project_name=project_name,
        inc=inc,
        project_path=project_path,
        reqa_file=reqa_file,
        max_auto_summarize_code=max_auto_summarize_code,
        recover_path=recover_path
    )

if __name__ == "__main__":
    mcp.run(transport='stdio')
    