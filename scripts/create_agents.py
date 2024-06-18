from typing import List, Optional, Dict
import numexpr
import math

from langchain.agents.output_parsers import (
    ReActJsonSingleInputOutputParser,
    OpenAIFunctionsAgentOutputParser,
)
from langchain.llms import HuggingFaceEndpoint
from langchain.chat_models import ChatOpenAI
from langchain.tools.render import (
    render_text_description_and_args,
    format_tool_to_openai_function,
)
from langchain.agents.format_scratchpad import (
    format_to_openai_function_messages,
    format_log_to_str,
)
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.agents import AgentExecutor, load_tools
from langchain.schema import HumanMessage
from langchain.chat_models.base import BaseChatModel
from langchain_community.chat_models.huggingface import ChatHuggingFace
from transformers.agents import Tool

from scripts.prompts import HUMAN_PROMPT, SYSTEM_PROMPT, SCRATCHPAD_PROMPT

class CalculatorTool(Tool):
    name = "calculator"
    description = "This is a tool that calculates. It can be used to perform simple arithmetic operations."

    inputs = {
        "expression": {
            "type": "text",
            "description": "The expression to be evaluated.The variables used CANNOT be placeholders like 'x' or 'mike's age', they must be numbers",
        }
    }
    output_type = "text"

    def forward(self, expression):
        if isinstance(expression, Dict):
            expression = expression["expression"]
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip().replace("^", "**"),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return output

def init_tools_with_llm(llm: BaseChatModel):
    tools = load_tools(["serpapi", "llm-math"], llm=llm)
    # Rename tools in the same format used by other tools
    tools[0].name = "search"
    # llm_math_tool = Tool(
    #     name="Calculator",
    #     description="Useful for when you need to answer questions about math.",
    #     func=LLMMathChain.from_llm(llm=llm).run,
    #     coroutine=LLMMathChain.from_llm(llm=llm).arun,
    # )
    # tools.append(llm_math_tool)
    tools[1].name = "calculator"
    return tools


def build_openai_agent_with_tools(model_id: Optional[str] = "gpt-4-1106-preview") -> AgentExecutor:
    llm = ChatOpenAI(model=model_id, temperature=0.1)
    tools = init_tools_with_llm(llm)


    llm_with_tools = llm.bind(
        functions=[format_tool_to_openai_function(t) for t in tools],
        stop=["Observation:", "<|eot_id|>"]
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Answer the following question:"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | OpenAIFunctionsAgentOutputParser()
    )
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_iterations=7,
    )


def build_hf_agent_with_tools(hf_endpoint_url: Optional[str] = None, repo_id: Optional[str] = None) -> AgentExecutor:
    """
    Build a zero-shot ReAct chat agent from HF endpoint.

    Args:
        hf_endpoint_url (str): The endpoint URL for the Hugging Face model.

    Returns:
        AgentExecutor: An agent executor object that can be used to run the agent.

    """
    assert hf_endpoint_url or repo_id, "hf_endpoint_url or repo_id must be provided."
    assert not (hf_endpoint_url and repo_id), "Only one of hf_endpoint_url or repo_id can be provided."

    # instantiate LLM and chat model
    if hf_endpoint_url:
        llm = HuggingFaceEndpoint(
            endpoint_url=hf_endpoint_url,
            task="text-generation",
            max_new_tokens= 512,
            do_sample= False,
            repetition_penalty= 1.03,
        )
    else:
        llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            task="text-generation",
            max_new_tokens= 512,
            do_sample= False,
            repetition_penalty= 1.03,
        )

    chat_model = ChatHuggingFace(llm=llm)
    tools = init_tools_with_llm(llm)

    # # TODO: remove
    # tools = [tools[1]] # only use calculator for now


    # define the prompt depending on whether the chat model supports system prompts
    system_prompt_supported = check_supports_system_prompt(chat_model)

    if system_prompt_supported:
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
                HumanMessagePromptTemplate.from_template(HUMAN_PROMPT),
                SystemMessagePromptTemplate.from_template(SCRATCHPAD_PROMPT),
            ]
        )
    else:
        prompt = ChatPromptTemplate.from_messages(
            [
                HumanMessagePromptTemplate.from_template(
                    SYSTEM_PROMPT + "\nSo, here is my question:" + HUMAN_PROMPT
                ),
                AIMessagePromptTemplate.from_template(SCRATCHPAD_PROMPT),
                HumanMessage(content="Now give your next thoughts: "),
            ]
        )

    prompt = prompt.partial(
        tool_description_with_args=render_text_description_and_args(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    # define the agent
    chat_model_with_stop = chat_model.bind(stop=["Observation:", "<|eot_id|>"])
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        }
        | prompt
        | chat_model_with_stop
        | ReActJsonSingleInputOutputParser()
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_iterations=7,
    )


def check_supports_system_prompt(chat_model):
    """
    Checks if the given chat model supports system prompts.

    Args:
        chat_model: The chat model to be checked.

    Returns:
        True if the chat model supports system prompts, False otherwise.
    """
    messages = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template(HUMAN_PROMPT),
            SystemMessagePromptTemplate.from_template(SCRATCHPAD_PROMPT),
        ]
    )
    try:
        chat_model._to_chat_prompt(messages)
        print("System prompt supported")
        return True
    except Exception as e:
        print(e)
        print("System prompt not supported")
        return False
