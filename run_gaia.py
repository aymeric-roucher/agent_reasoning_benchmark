import os
import asyncio
import datasets
import pandas as pd
from openai import OpenAI
from transformers.agents import HfEngine, PythonInterpreterTool
from transformers.agents.llm_engine import MessageRole, get_clean_message_list
from dotenv import load_dotenv

load_dotenv(".env", override=True)

from manager import Manager, CodeManager
from tools import (
    WebSearchTool,
    VisualQATool,
    VisualQAGPT4Tool,
    FileInspectorTool,
    AudioQATool
)
from run_agents import answer_questions

OUTPUT_DIR = "gaia_output"
n_samples = 40
USE_OAI = True
USE_CODE = False
output_fname = f"plan_exec_{'llama_3_70b_instr' if not USE_OAI else 'gpt4o'}_{'code' if USE_CODE else 'json'}_{n_samples}_6"

eval_ds = datasets.load_dataset("gaia-benchmark/GAIA", "2023_all", split=f"validation[:{n_samples}]", token=os.environ["HF_TOKEN"])
eval_ds = eval_ds.rename_columns(
    {"Question": "question", "Final answer": "true_answer", "Level": "task"}
)

def preprocess_file_paths(row):
    if len(row["file_name"]) > 0:
        row["file_name"] = "data/gaia/validation/" + row["file_name"]
    return row


eval_ds = eval_ds.map(preprocess_file_paths)
eval_df = pd.DataFrame(eval_ds)

openai_role_conversions = {
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}

class OpenAIModel:
    def __init__(self, model_name="gpt-4o"):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    def __call__(self, messages, stop_sequences=[]):
        messages = get_clean_message_list(messages, role_conversions=openai_role_conversions)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            stop=stop_sequences,
        )
        return response.choices[0].message.content


if USE_OAI:
    main_engine = OpenAIModel()
    tool_engine = OpenAIModel()
else:
    main_engine = HfEngine(model="meta-llama/Meta-Llama-3-70B-Instruct")
    tool_engine = HfEngine("CohereForAI/c4ai-command-r-plus")

if USE_CODE:
    agent = CodeManager(
        llm_engine=main_engine,
        tools=[
            WebSearchTool(llm_engine=tool_engine),
            VisualQAGPT4Tool() if USE_OAI else VisualQATool(),
            FileInspectorTool(llm_engine=tool_engine),
            AudioQATool(llm_engine=tool_engine, device="cpu"),
        ],
        additional_authorized_imports=["requests", "zipfile", "os", "pandas", "numpy", "json", "bs4"],
        max_iterations=12,
        verbose=1
    )
else:
    agent = Manager(
        llm_engine=main_engine,
        tools=[
            WebSearchTool(llm_engine=tool_engine),
            VisualQAGPT4Tool() if USE_OAI else VisualQATool(),
            FileInspectorTool(llm_engine=tool_engine),
            AudioQATool(llm_engine=tool_engine, device="cpu"),
            PythonInterpreterTool()
        ],
        max_iterations=12,
        verbose=1
    )

# result = agent.run("If Eliud Kipchoge could maintain his record-making marathon pace indefinitely, how many thousand hours would it take him to run the distance between the Earth and the Moon its closest approach? Please use the minimum perigee value on the Wikipedia page for the Moon when carrying out your calculation. Round your result to the nearest 1000 hours and do not use any comma separators if necessary.")
# print(result)
# print(type(result))

async def call_transformers(agent, question: str, **kwargs):
    result = agent.run(question, **kwargs)
    return {
        "output": str(result),
        "intermediate_steps": [
            {key: value for key, value in log.items() if key != "agent_memory"}
            for log in agent.logs
        ],
    }

def call_transformers_sync(agent, question: str, **kwargs):
    result = agent.run(question, **kwargs)
    return {
        "output": str(result),
        "intermediate_steps": [
            {key: value for key, value in log.items() if key != "agent_memory"}
            for log in agent.logs
        ],
    }

os.makedirs(OUTPUT_DIR, exist_ok=True)

results = asyncio.run(answer_questions(
    eval_ds,
    agent,
    output_fname,
    output_folder=OUTPUT_DIR,
    agent_call_function=call_transformers,
    add_optional_visualizer_tool=True,
))

