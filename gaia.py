import pandas as pd
from dotenv import load_dotenv
import datasets

load_dotenv(override=True)
pd.set_option("max_colwidth", None)

OUTPUT_DIR = "output_gaia"

from huggingface_hub import login
import os

login(os.getenv("HUGGINGFACEHUB_API_TOKEN"))

from scripts.tools.web_surfer import (
    SearchInformationTool,
    NavigationalSearchTool,
    VisitTool,
    DownloadTool,
    PageUpTool,
    PageDownTool,
    FinderTool,
    FindNextTool,
    browser,
)
from scripts.tools.mdconvert import MarkdownConverter

from scripts.run_agents import answer_questions
from openai import OpenAI
from transformers.agents.llm_engine import MessageRole, get_clean_message_list
from transformers.agents import HfEngine
from transformers.agents import ReactCodeAgent, HfEngine
from transformers.agents.prompts import DEFAULT_REACT_CODE_SYSTEM_PROMPT, DEFAULT_REACT_JSON_SYSTEM_PROMPT
from transformers.agents.default_tools import Tool
from scripts.tools.visual_qa import VisualQATool, VisualQAGPT4Tool
from transformers.agents import SpeechToTextTool, PythonInterpreterTool
import shutil
import asyncio

### IMPORTANT: EVALUATION SWITCHES

USE_OS_MODELS = False
USE_JSON_AGENT = False

### BUILD LLM ENGINES

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
            stop=stop_sequences,
        )
        return response.choices[0].message.content


oai_llm_engine = OpenAIModel()

url_llama3 = "meta-llama/Meta-Llama-3-70B-Instruct"
url_qwen2 = "https://azbwihkodyacoe54.us-east-1.aws.endpoints.huggingface.cloud"
url_command_r = "CohereForAI/c4ai-command-r-plus"

### LOAD EVALUATION DATASET

eval_ds = datasets.load_dataset("gaia-benchmark/GAIA", "2023_all")["validation"]
eval_ds = eval_ds.rename_columns(
    {"Question": "question", "Final answer": "true_answer", "Level": "task"}
)


def preprocess_file_paths(row):
    if len(row["file_name"]) > 0:
        row["file_name"] = "data/gaia/validation/" + row["file_name"]

    replaced_name = row["file_name"].replace(".xlsx", ".png").replace('.pdf', '.png')
    if os.path.exists(replaced_name):
        row["file_name"] = replaced_name
    return row


eval_ds = eval_ds.map(preprocess_file_paths)

eval_df = pd.DataFrame(eval_ds)
print("Loaded evaluation dataset:")
print(pd.Series(eval_ds["task"]).value_counts())


from transformers.agents import ReactJsonAgent, HfEngine

websurfer_llm_engine = HfEngine(
    model=url_qwen2,
)  # chosen for its high context length

# Replace with OAI if needed
if not USE_OS_MODELS:
    websurfer_llm_engine = oai_llm_engine

### BUILD AGENTS & TOOLS

WEB_TOOLS = [
    SearchInformationTool(),
    NavigationalSearchTool(),
    VisitTool(),
    DownloadTool(),
    PageUpTool(),
    PageDownTool(),
    FinderTool(),
    FindNextTool(),
]

class TextInspectorTool(Tool):
    name = "inspect_file_as_text"
    description = """
You cannot load files yourself: instead call this tool to read a file as markdown text and ask questions about it.
This tool handles the following file extensions: [".html", ".htm", ".xlsx", ".pptx", ".wav", ".mp3", ".flac", ".pdf", ".docx"], and all other types of text files. IT DOES NOT HANDLE IMAGES."""

    inputs = {
        "question": {
            "description": "Your question, as a natural language sentence. Provide as much context as possible.",
            "type": "text",
        },
        "file_path": {
            "description": "The path to the file you want to read as text. Must be a '.something' file, like '.pdf'. If it is an image, use the visualizer tool instead! DO NOT USE THIS TOOL FOR A WEBPAGE: use the search tool instead!",
            "type": "text",
        },
    }
    output_type = "text"
    md_converter = MarkdownConverter()

    def forward(self, question: str, file_path) -> str:

        result = self.md_converter.convert(file_path)

        if ".zip" in file_path:
            return result.text_content

        messages = [
            {
                "role": "user",
                "content": "You will have to write a short caption for this file, then answer this question:"
                + question,
            },
            {
                "role": "user",
                "content": "Here is the complete file:\n### "
                + str(result.title)
                + "\n\n"
                + result.text_content[:70000],
            },
            {
                "role": "user",
                "content": "Now write a short caption for the file, then answer this question:"
                + question,
            },
        ]
        return websurfer_llm_engine(messages)


ti_tool_search = TextInspectorTool()
ti_tool_search.description = """
Call this tool to read a downloaded file as markdown text and ask questions about it.
This tool handles the following file extensions: [".html", ".htm", ".xlsx", ".pptx", ".wav", ".mp3", ".flac", ".pdf", ".docx"], and all other types of text files. IT DOES NOT HANDLE IMAGES."""

WEB_TOOLS.append(ti_tool_search)


surfer_agent = ReactJsonAgent(
    llm_engine=websurfer_llm_engine,
    tools=WEB_TOOLS,
    max_iterations=12,
    verbose=1,
    system_prompt=DEFAULT_REACT_JSON_SYSTEM_PROMPT + "\nAdditionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information.",
)


params = {
    "engine": "bing",
    "gl": "us",
    "hl": "en",
}


class SearchTool(Tool):
    name = "ask_search_agent"
    description = "A search agent that will browse the internet to answer your question. Ask him for all your web-search related questions, but he's unable to do problem-solving. Provide him as much context as possible, in particular if you need to search on a specific timeframe!"

    inputs = {
        "query": {
            "description": "Your question, as a natural language sentence with a verb! You are talking to an human, so provide them with as much context as possible! DO NOT ASK a google-like query like 'paper about fish species 2011': instead ask a real sentence like: 'What appears on the last figure of a paper about fish species published in 2011?'",
            "type": "text",
        }
    }
    output_type = "text"

    def forward(self, query: str) -> str:
        return surfer_agent.run(f"You've been submitted this request by your manager: {query}\nYou're solving the task for your manager: so even if your search is unsuccessful, please return as much context as possible, so they can act upon this feedback. Also, if the answer to the task is on an image or pdf file, you can download it to inspect it. If you do not succeed to inspect it, you can return the path where the file was downloaded, and your manager will handle it from there.")


class ZipInspectorTool(Tool):
    name = "extract_inspect_zip_folder"
    description = "Use this to extract and inspect the contents of a zip folder."
    inputs = {
        "folder": {
            "description": "The path to the zip folder you want to inspect.",
            "type": "text",
        }
    }
    output_type = "text"

    def forward(self, folder: str) -> str:
        folder_name = folder.replace(".zip", "")
        os.makedirs(folder_name, exist_ok=True)
        shutil.unpack_archive(folder, folder_name)

        # Convert the extracted files
        result = "We extracted all files from the zip into a directory: find the extracted files at the following paths:\n"
        for root, dirs, files in os.walk(folder_name):
            for file in files:
                result += f"- {os.path.join(root, file)}\n"

        return result

ti_tool = TextInspectorTool()

TASK_SOLVING_TOOLBOX = [
    SearchTool(),
    VisualQAGPT4Tool(),  # VisualQATool(),
    ti_tool,
    ZipInspectorTool(),
]


if USE_JSON_AGENT:
    TASK_SOLVING_TOOLBOX.append(PythonInterpreterTool())

hf_llm_engine = HfEngine(model=url_qwen2)

llm_engine = hf_llm_engine if USE_OS_MODELS else oai_llm_engine

if USE_JSON_AGENT:
    react_agent = ReactJsonAgent(
        llm_engine=llm_engine,
        tools=TASK_SOLVING_TOOLBOX,
        max_iterations=10,
        verbose=0,
        memory_verbose=True,
        system_prompt=DEFAULT_REACT_JSON_SYSTEM_PROMPT,
    )
else:
    react_agent = ReactCodeAgent(
        llm_engine=llm_engine,
        tools=TASK_SOLVING_TOOLBOX,
        max_iterations=10,
        verbose=0,
        memory_verbose=True,
        system_prompt=DEFAULT_REACT_CODE_SYSTEM_PROMPT,
        additional_authorized_imports=["requests", "zipfile", "os", "pandas", "numpy", "json", "bs4"],
    )

### EVALUATE

from scripts.reformulator import prepare_response

async def call_transformers(agent, question: str, **kwargs) -> str:
    result = agent.run(question, **kwargs)
    agent_memory = agent.write_inner_memory_from_logs()[1:]
    try:
        final_result = prepare_response(question, agent_memory, llm_engine)
    except Exception as e:
        final_result = result
    return {
        "output": str(final_result),
        "intermediate_steps": [
            {key: value for key, value in log.items() if key != "agent_memory"}
            for log in agent.logs
        ],
    }

food_file = "data/gaia/validation/9b54f9d9-35ee-4a14-b62f-d130ea00317f/food_duplicates.xls"
# surfer_agent.run("What was the revenue of Carrefour in 2013?", run_planning_step=True)
# surfer_agent

results = asyncio.run(answer_questions(
    eval_ds,
    react_agent,
    "react_code_gpt4o_12-june_planning3_initial-file-inspection",
    output_folder=OUTPUT_DIR,
    agent_call_function=call_transformers,
    visual_inspection_tool = VisualQAGPT4Tool(),
    text_inspector_tool = ti_tool,
))