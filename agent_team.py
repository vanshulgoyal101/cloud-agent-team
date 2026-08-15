import os
import sys
import json
import time
import subprocess
from pathlib import Path
from pydantic import BaseModel, Field
import google.generativeai as genai
from openai import OpenAI
from groq import Groq

# Define workspace directories
WORKSPACE_DIR = Path("workspace")
WORKSPACE_DIR.mkdir(exist_ok=True)
(WORKSPACE_DIR / "src").mkdir(exist_ok=True)
(WORKSPACE_DIR / "tests").mkdir(exist_ok=True)

STATE_FILE = WORKSPACE_DIR / "state.json"

# Structured Pydantic Schemas
class FileAction(BaseModel):
    path: str = Field(description="Relative path of the file from workspace root (e.g. 'src/utils.py')")
    content: str = Field(description="Full text contents to write or replace in the file")
    action: str = Field(description="Action to perform: 'create', 'modify', or 'delete'")

class BacklogTask(BaseModel):
    id: str = Field(description="Unique task identifier (e.g. 'task_001')")
    title: str = Field(description="Short title describing the task")
    description: str = Field(description="Detailed instructions and contracts for the Coder agent")
    status: str = Field(default="todo", description="Status of the task: 'todo', 'completed', or 'failed'")
    files_affected: list[str] = Field(description="List of paths of files this task targets")

class Backlog(BaseModel):
    tasks: list[BacklogTask] = Field(description="Sequence of development tasks to execute")

class CoderResponse(BaseModel):
    explanation: str = Field(description="Brief explanation of the changes written")
    files: list[FileAction] = Field(description="File modifications/creations to perform")

class DebuggerResponse(BaseModel):
    explanation: str = Field(description="Explanation of the fix")
    files: list[FileAction] = Field(description="Corrected file contents")


class APIKeyRotator:
    def __init__(self):
        self.keys = []
        # Attempt to load rotation key list from environment
        keys_env = os.environ.get("AGENT_API_KEYS")
        if keys_env:
            try:
                self.keys = json.loads(keys_env)
                print(f"Loaded {len(self.keys)} API keys from AGENT_API_KEYS config.")
            except Exception as e:
                print(f"Error parsing AGENT_API_KEYS environment variable: {e}")
        
        # Attempt to load from local keys.json file
        if not self.keys:
            local_keys_path = Path(__file__).parent / "keys.json"
            if local_keys_path.exists():
                try:
                    self.keys = json.loads(local_keys_path.read_text(encoding="utf-8"))
                    print(f"Loaded {len(self.keys)} API keys from local keys.json file.")
                except Exception as e:
                    print(f"Error parsing local keys.json: {e}")
        
        # Fallback to standard individual env vars if key list is empty
        if not self.keys:
            if os.environ.get("GEMINI_API_KEY"):
                self.keys.append({"provider": "gemini", "key": os.environ.get("GEMINI_API_KEY")})
            if os.environ.get("OPENAI_API_KEY"):
                self.keys.append({"provider": "openai", "key": os.environ.get("OPENAI_API_KEY")})
            if os.environ.get("GROQ_API_KEY"):
                self.keys.append({"provider": "groq", "key": os.environ.get("GROQ_API_KEY")})
        
        self.current_idx = 0
        self.blocked_keys = {} # key_index -> unix_timestamp_until_unblocked

    def get_current_provider(self):
        if not self.keys:
            raise ValueError("No API keys configured! Please set GEMINI_API_KEY, AGENT_API_KEYS, or other provider keys.")
        
        # Find next unblocked key
        now = time.time()
        for i in range(len(self.keys)):
            idx = (self.current_idx + i) % len(self.keys)
            unblock_time = self.blocked_keys.get(idx, 0)
            if now >= unblock_time:
                self.current_idx = idx
                return self.keys[idx]
        
        # All keys are blocked!
        raise BlockingIOError("All configured API keys are currently rate-limited/exhausted. Exiting loop.")

    def mark_current_blocked(self, cooldown_sec=1800):
        """Marks the current key as rate-limited/exhausted for the cooldown period (default 30 mins)."""
        now = time.time()
        self.blocked_keys[self.current_idx] = now + cooldown_sec
        print(f"⚠️ Marked Key Index {self.current_idx} ({self.keys[self.current_idx]['provider']}) as blocked/exhausted.")
        self.current_idx = (self.current_idx + 1) % len(self.keys)

    def generate_content(self, system_prompt: str, prompt: str, schema: BaseModel = None) -> str:
        """Call LLM using the current active API key provider, handling rotation automatically."""
        retries = len(self.keys)
        while retries > 0:
            provider_info = self.get_current_provider()
            provider = provider_info["provider"]
            key = provider_info["key"]
            
            try:
                if provider == "gemini":
                    genai.configure(api_key=key)
                    # Use gemini-2.5-flash as default, or fall back to gemini-1.5-flash if needed
                    model = genai.GenerativeModel(
                        model_name="gemini-2.5-flash",
                        system_instruction=system_prompt
                    )
                    
                    generation_config = {}
                    if schema:
                        generation_config = {
                            "response_mime_type": "application/json",
                            "response_schema": schema,
                        }
                    
                    response = model.generate_content(
                        prompt,
                        generation_config=generation_config
                    )
                    return response.text
                
                elif provider == "openai":
                    client = OpenAI(api_key=key)
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                    
                    # Call OpenAI
                    kwargs = {"model": "gpt-4o-mini", "messages": messages}
                    if schema:
                        kwargs["response_format"] = {"type": "json_object"}
                        # Inject schema requirements into prompt to ensure JSON schema compliance
                        messages[0]["content"] += f"\nOutput MUST conform to this JSON schema: {json.dumps(schema.model_json_schema())}"

                    response = client.chat.completions.create(**kwargs)
                    return response.choices[0].message.content
                
                elif provider == "groq":
                    client = Groq(api_key=key)
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                    
                    kwargs = {"model": "llama-3.3-70b-versatile", "messages": messages}
                    if schema:
                        kwargs["response_format"] = {"type": "json_object"}
                        messages[0]["content"] += f"\nOutput MUST conform to this JSON schema: {json.dumps(schema.model_json_schema())}"
                        
                    response = client.chat.completions.create(**kwargs)
                    return response.choices[0].message.content
                
                else:
                    raise ValueError(f"Unknown provider: {provider}")
                    
            except Exception as e:
                err_str = str(e).lower()
                # Rotate on rate limits, quota limits, leaked keys, or invalid permissions
                if any(x in err_str for x in ["rate", "quota", "429", "exhaust", "limit", "permission", "denied", "leaked", "key", "auth"]):
                    print(f"⚠️ Provider error on {provider} (Key Index {self.current_idx}): {e}. Rotating key...")
                    self.mark_current_blocked()
                    retries -= 1
                else:
                    # Reraise other errors (e.g. prompt formatting) immediately
                    raise e
                    
        raise BlockingIOError("All providers failed or returned rate limits.")


class ProductionAgentTeam:
    def __init__(self):
        self.rotator = APIKeyRotator()
        self.state = {
            "current_step": "innovate", # innovate -> plan -> decompose -> code -> verify
            "active_task_id": None,
            "project_name": "Autonomous App",
            "backlog": []
        }
        self.load_state()

    def load_state(self):
        if STATE_FILE.exists():
            try:
                self.state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
                print(f"Loaded existing checkpoint. Current step: {self.state['current_step']}")
            except Exception as e:
                print(f"Error loading state.json: {e}. Starting fresh.")

    def save_state(self):
        STATE_FILE.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        print("💾 State checkpoint saved.")

    def get_workspace_state(self) -> str:
        state = []
        for path in WORKSPACE_DIR.rglob("*"):
            if path.is_file() and not path.name.startswith(".") and "venv" not in path.parts:
                try:
                    content = path.read_text(encoding="utf-8")
                    state.append(f"=== File: {path.relative_to(WORKSPACE_DIR)} ===\n{content}\n")
                except Exception:
                    state.append(f"=== File: {path.relative_to(WORKSPACE_DIR)} (Binary/Unreadable) ===\n")
        return "\n".join(state) if state else "Workspace is empty."

    def run_cycle(self):
        """Orchestrates agent execution in stages, allowing seamless pause/resume on failure."""
        try:
            if self.state["current_step"] == "innovate":
                self.run_innovator()
                self.state["current_step"] = "plan"
                self.save_state()

            if self.state["current_step"] == "plan":
                self.run_architect()
                self.state["current_step"] = "decompose"
                self.save_state()

            if self.state["current_step"] == "decompose":
                self.run_planner()
                self.state["current_step"] = "code"
                self.save_state()

            if self.state["current_step"] == "code":
                self.run_coder_loop()
                self.state["current_step"] = "verify"
                self.save_state()

            if self.state["current_step"] == "verify":
                self.run_tester()
                # Cycle completes! Reset step to innovate to build/improve further on next trigger
                self.state["current_step"] = "innovate"
                self.save_state()
                print("🎉 Agent execution cycle completed successfully!")
                
        except BlockingIOError as bioe:
            print(f"\n⏸️ Execution paused: {bioe}")
            print("The runner will exit safely and resume from the current step on the next cron run.")
            sys.exit(0)

    def run_innovator(self):
        print("💡 Running Innovator Agent...")
        workspace_state = self.get_workspace_state()
        ideas_path = WORKSPACE_DIR / "ideas.md"
        current_ideas = ideas_path.read_text(encoding="utf-8") if ideas_path.exists() else "No ideas yet."

        sys_prompt = "You are an Innovator Agent designing novel, high-impact features or projects."
        prompt = f"""
We want to expand the current codebase into a highly complex, production-ready system.
Look at the current state of files:
{workspace_state}

Existing Ideas:
{current_ideas}

Brainstorm 3 new highly scalable production-ready project features or complete modules.
Write the result to `ideas.md` in clean markdown format, including descriptions, tech stack details, and expected components.
"""
        response_text = self.rotator.generate_content(sys_prompt, prompt)
        ideas_path.write_text(response_text, encoding="utf-8")
        print("✅ Innovator updated ideas.md")

    def run_architect(self):
        print("🏛️ Running Principal Architect...")
        workspace_state = self.get_workspace_state()
        ideas = (WORKSPACE_DIR / "ideas.md").read_text(encoding="utf-8")

        sys_prompt = "You are a Principal Software Architect who designs modular, clean, and highly robust folder structures and application flows."
        prompt = f"""
Read the proposed ideas:
{ideas}

Here is the current workspace:
{workspace_state}

Select the most impact/complex feature/idea to build.
Design a production-ready folder architecture, listing:
1. File structure design (e.g. config/, src/controllers/, src/models/, src/routes/, tests/)
2. Interaction flows and components contracts.
3. System design guidelines.

Save your architecture specification in `system_architecture.md`. Output only markdown content.
"""
        response_text = self.rotator.generate_content(sys_prompt, prompt)
        (WORKSPACE_DIR / "system_architecture.md").write_text(response_text, encoding="utf-8")
        print("✅ Architect created system_architecture.md")

    def run_planner(self):
        print("📋 Running Task Planner...")
        architecture = (WORKSPACE_DIR / "system_architecture.md").read_text(encoding="utf-8")
        
        sys_prompt = "You are a Task Planner. You translate architectures into detailed step-by-step issue trackers."
        prompt = f"""
Look at the designed architecture:
{architecture}

Generate a backlog of tasks. Each task should have:
1. An ID (e.g. task_001)
2. Detailed instructions for writing code (contracts, API shapes)
3. Target file paths

Return the backlog in JSON format matching the schema provided.
"""
        response_text = self.rotator.generate_content(sys_prompt, prompt, schema=Backlog)
        try:
            data = json.loads(response_text)
            backlog = Backlog(**data)
            self.state["backlog"] = [task.model_dump() for task in backlog.tasks]
            print(f"✅ Decomposed into {len(self.state['backlog'])} tasks.")
        except Exception as e:
            print(f"❌ Failed to parse backlog schema: {e}")
            raise e

    def run_coder_loop(self):
        print("💻 Running Coding Subagents Loop...")
        for i, task_data in enumerate(self.state["backlog"]):
            task = BacklogTask(**task_data)
            if task.status == "completed":
                continue
            
            print(f"🔨 Implementing Task {task.id}: {task.title}")
            self.state["active_task_id"] = task.id
            self.save_state()
            
            workspace_state = self.get_workspace_state()
            architecture = (WORKSPACE_DIR / "system_architecture.md").read_text(encoding="utf-8")
            
            sys_prompt = "You are a Senior Software Developer Subagent. You write bug-free, clean code matching a precise task description."
            prompt = f"""
Architecture Context:
{architecture}

Your Specific Task:
Task Title: {task.title}
Task Description: {task.description}
Files Affected: {', '.join(task.files_affected)}

Current Workspace State:
{workspace_state}

Write the code implementations. Return a structured JSON of files to write or modify.
"""
            response_text = self.rotator.generate_content(sys_prompt, prompt, schema=CoderResponse)
            try:
                data = json.loads(response_text)
                coder_res = CoderResponse(**data)
                self.apply_file_actions(coder_res.files)
                
                # Mark current task as completed in memory state
                self.state["backlog"][i]["status"] = "completed"
                self.save_state()
            except Exception as e:
                print(f"❌ Coding subagent failed on task {task.id}: {e}")
                self.state["backlog"][i]["status"] = "failed"
                self.save_state()
                raise e

    def apply_file_actions(self, files: list[FileAction]):
        for file in files:
            target_path = WORKSPACE_DIR / file.path
            if not target_path.resolve().is_relative_to(WORKSPACE_DIR.resolve()):
                print(f"⚠️ Path {file.path} is outside workspace boundaries. Skipping.")
                continue

            if file.action == "delete":
                if target_path.exists():
                    target_path.unlink()
                    print(f"🗑️ Deleted {file.path}")
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(file.content, encoding="utf-8")
                print(f"💾 Saved {file.path} ({file.action})")

    def run_tester(self):
        print("🔍 Running Integration Tester & Debugger Agent...")
        tests = list(WORKSPACE_DIR.glob("tests/test_*.py"))
        run_cmd = ["pytest", str(WORKSPACE_DIR / "tests")] if tests else ["python3", "-m", "compileall", str(WORKSPACE_DIR)]
        
        print(f"🏃 Executing checks: {' '.join(run_cmd)}")
        result = subprocess.run(run_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("🎉 All test verification passed successfully!")
            return

        print("❌ Verification failed! Running Debugger...")
        workspace_state = self.get_workspace_state()
        sys_prompt = "You are a Debugger Agent. You diagnose failing builds or test logs and generate files to fix them."
        prompt = f"""
Recent check failure logs:
Stdout:
{result.stdout}
Stderr:
{result.stderr}

Current workspace:
{workspace_state}

Provide the corrections in standard JSON format matching the schema to fix this build failure.
"""
        response_text = self.rotator.generate_content(sys_prompt, prompt, schema=DebuggerResponse)
        try:
            data = json.loads(response_text)
            debug_res = DebuggerResponse(**data)
            self.apply_file_actions(debug_res.files)
            print(f"🔧 Debugger applied fixes: {debug_res.explanation}")
            
            # Re-verify
            print("🔄 Re-verifying...")
            new_res = subprocess.run(run_cmd, capture_output=True, text=True)
            if new_res.returncode == 0:
                print("🎉 Verification passed after bug fixes!")
            else:
                print("⚠️ Errors still persist. Will attempt debugging again during next run cycle.")
        except Exception as e:
            print(f"❌ Debugger execution failed: {e}")
            raise e

if __name__ == "__main__":
    team = ProductionAgentTeam()
    team.run_cycle()
