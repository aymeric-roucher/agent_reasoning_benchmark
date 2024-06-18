import logging
import re
import json

from transformers.agents import Agent
from transformers.agents.agents import (
    AgentError,
    AgentParsingError,
    AgentExecutionError,
    AgentGenerationError,
    AgentMaxIterationsError,
    DEFAULT_TOOL_DESCRIPTION_TEMPLATE
)
from transformers.agents.llm_engine import MessageRole
from transformers.agents.default_tools import FinalAnswerTool, evaluate_python_code, BASE_PYTHON_TOOLS, LIST_SAFE_MODULES
from prompts import (
    FACTS_PROMPT_USER,
    FACTS_PROMPT_SYSTEM,
    PLAN_PROMPT_USER,
    PLAN_PROMPT_SYSTEM,
    NEXT_ACTION_PROMPT_USER,
    NEXT_ACTION_PROMPT_SYSTEM,
    NEXT_ACTION_CODE_PROMPT_SYSTEM
)

from pygments import highlight
from pygments.formatters import Terminal256Formatter
from pygments.lexers import PythonLexer

class Manager(Agent):

    def __init__(
        self,
        llm_engine,
        tools,
        **kwargs
    ):
        super().__init__(
            tools=tools,
            llm_engine=llm_engine,
            **kwargs
        )
        self.logger.setLevel(logging.INFO)
        if "final_answer" not in self._toolbox.tools:
            self._toolbox.add_tool(FinalAnswerTool())

        tool_descriptions = self.toolbox.show_tool_descriptions(DEFAULT_TOOL_DESCRIPTION_TEMPLATE)
        tool_names = [f"'{tool_name}'" for tool_name in self.toolbox.tools.keys()]
        self.system_prompts = {
            "facts": FACTS_PROMPT_SYSTEM,
            "plan": PLAN_PROMPT_SYSTEM.replace("<<tool_descriptions>>", tool_descriptions),
            # "critique_plan": CRITIC_PROMPT_SYSTEM.replace("<<tool_descriptions>>", tool_descriptions),
            # "refine_plan": REFINE_PLAN_PROMPT_SYSTEM.replace("<<tool_descriptions>>", tool_descriptions),
            "action": NEXT_ACTION_PROMPT_SYSTEM.replace("<<tool_names>>", ", ".join(tool_names)).replace("<<tool_descriptions>>", tool_descriptions)
        }

    def initialize_for_run(self, task: str, **kwargs):
        self.task = task
        if len(kwargs) > 0:
            self.task += f"\nYou have been provided with these initial arguments: {str(kwargs)}."
        self.state = kwargs.copy()
        self.logger.warn("======== New task ========")
        self.logger.log(33, self.task)
        self.state["facts"] = "None"
        self.state["plan"] = "None"
        self.logs = []

    def write_inner_memory_from_logs(self):
        prev_step, new_step = "", ""
        for i, step_log in enumerate(self.logs):
            if "error" in step_log:
                message_content = (
                    "Error: "
                    + str(step_log["error"])
                    + "\nNow let's retry: take care not to repeat previous errors! Try to adopt different approaches.\n"
                )
            else:
                message_content = f"Observation: {step_log['observation']}. Used {step_log['tool_name']} with arguments {step_log['arguments']}\n"
            if i < len(self.logs) - 1:
                prev_step += message_content
            else:
                new_step = message_content

        return prev_step, new_step

    def run(self, task: str, **kwargs):
        self.initialize_for_run(task, **kwargs)
        final_answer = None
        iteration = 0
        while final_answer is None and iteration < self.max_iterations:
            try:
                final_answer = self.step()
            except AgentError as e:
                self.logger.error(e, exc_info=1)
                self.logs[-1]["error"] = e
            finally:
                iteration += 1

        if final_answer is None and iteration == self.max_iterations:
            error_message = "Reached max iterations."
            self.logs.append({"error": AgentMaxIterationsError(error_message)})
            self.logger.error(error_message, exc_info=1)

            prompt = [
                {
                    "role": MessageRole.SYSTEM,
                    "content": """An agent tried to answer a user query but it failed to do so. You are tasked with providing an answer instead. Your answer should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
ADDITIONALLY, your answer MUST adhere to any formatting instructions specified in the task (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, and don't include units such as $ or percent signs unless specified otherwise.
If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
If you are asked for a comma separated list, apply the above rules depending on whether the elements are numbers or strings. This is the final try - do your best to come up with an answer.""",
                },
                {
                    "role": MessageRole.USER,
                    "content": """This is what we know about the problem so far:
                    {facts}
                    
                    The history of previous actions and observations:
                    {prev_steps}
                    
                    Based on the above, give an answer to the task:
                    {task}
                    """.format(
                        facts=self.state["facts"],
                        prev_steps="\n".join(self.write_inner_memory_from_logs()),
                        task=self.task
                    )
                }
            ]
            try:
                final_answer = self.llm_engine(prompt)
            except Exception as e:
                final_answer = f"Error in generating final llm output: {e}."

        return final_answer

    def summarize_facts(self, prev_steps, new_step):
        facts_prompt = [
            {
                "role": MessageRole.SYSTEM,
                "content": self.system_prompts["facts"]
            },
            {
                "role": MessageRole.USER,
                "content": FACTS_PROMPT_USER.format(task=self.task, prev_steps=prev_steps, facts=self.state["facts"], new_step=new_step)
            }
        ]
        
        self.state["facts"] = self.llm_engine(facts_prompt)

    def plan(self, prev_steps):
        plan_prompt = [
            {
                "role": MessageRole.SYSTEM,
                "content": self.system_prompts["plan"]
            },
            {
                "role": MessageRole.USER,
                "content": PLAN_PROMPT_USER.format(task=self.task, facts=self.state["facts"], prev_steps=prev_steps) #, plan=self.state["plan"])
            }
        ]
        self.state["plan"] = self.llm_engine(plan_prompt)
    

    def critique_plan(self, prev_steps):
        critique_prompt = [
            {
                "role": MessageRole.SYSTEM,
                "content": self.system_prompts["critique_plan"]
            },
            {
                "role": MessageRole.USER,
                "content": CRITIC_PROMPT_USER.format(task=self.task, facts=self.state["facts"], prev_steps=prev_steps, plan=self.state["plan"])
            }
        ]
        return self.llm_engine(critique_prompt)
    
    def refine_plan(self, prev_steps, critique):
        refine_prompt = [
            {
                "role": MessageRole.SYSTEM,
                "content": self.system_prompts["refine_plan"]
            },
            {
                "role": MessageRole.USER,
                "content": REFINE_PLAN_PROMPT_USER.format(task=self.task, facts=self.state["facts"], prev_steps=prev_steps, plan=self.state["plan"], critique=critique)
            }
        ]
        self.state["plan"] = self.llm_engine(refine_prompt)

    def action(self, prev_steps):
        action_prompt = [
            {
                "role": MessageRole.SYSTEM,
                "content": self.system_prompts["action"]
            },
            {
                "role": MessageRole.USER,
                "content": NEXT_ACTION_PROMPT_USER.format(
                    task=self.task, facts=self.state["facts"], prev_steps=prev_steps, plan=self.state["plan"]
                )
            }
        ]
        return self.llm_engine(action_prompt, stop_sequences=["<end_action>"])

    def step(self):
        prev_steps, new_step = self.write_inner_memory_from_logs()
        self.logs.append({})

        try:
            self.summarize_facts(prev_steps, new_step)
        except Exception as e:
            self.logger.error(f"Error in generating llm output: {e}.")
            raise AgentGenerationError(f"Error in generating llm output: {e}.")
        self.logger.debug(f"Facts: {self.state['facts']}")
        self.logs[-1]["facts"] = self.state["facts"]

        prev_steps += new_step
        
        try:
            self.plan(prev_steps)
        except Exception as e:
            self.logger.error(f"Error in generating llm output: {e}.")
            raise AgentGenerationError(f"Error in generating llm output: {e}.")
        self.logger.debug(f"New plan: {self.state['plan']}")
        self.logs[-1]["plan"] = self.state["plan"]

        # try:
        #     critique = self.critique_plan(prev_steps)
        #     self.refine_plan(prev_steps, critique)
        # except Exception as e:
        #     self.logger.error(f"Error in generating llm output: {e}.")
        
        # self.logger.debug(f"Refined plan: {self.state['plan']}")
        # self.logs[-1]["refined_plan"] = self.state["plan"]

        try:
            action = self.action(prev_steps)
        except Exception as e:
            self.logger.error(f"Error in generating llm output: {e}.")
            raise AgentGenerationError(f"Error in generating llm output: {e}.")

        rationale, action = self.extract_action(llm_output=action, split_token="Action:")
        try:
            tool_name, arguments = self.tool_parser(action)
        except Exception as e:
            self.logger.error(f"Could not parse the given action: {e}.")
            raise AgentParsingError(f"Could not parse the given action: {e}.")
        self.logs[-1]["tool_name"] = tool_name
        self.logs[-1]["arguments"] = arguments
        self.logger.warning(f"Calling tool: '{tool_name}' with arguments: {arguments}")
        if tool_name == "final_answer":
            if isinstance(arguments, dict):
                answer = arguments["answer"]
            else:
                answer = arguments
            if answer in self.state:  # if the answer is a state variable, return the value
                answer = self.state[answer]
            return answer

        try:
            observation = self.execute_tool_call(tool_name, arguments)
            self.logger.info(f"Observation: {observation}")
        except Exception as e:
            self.logger.error(f"Could not parse the given action: {e}.")
            raise AgentExecutionError(f"Could not parse the given action: {e}.")
        self.logs[-1]["observation"] = observation

        return None


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    bold_yellow = "\x1b[33;1m"
    red = "\x1b[31;20m"
    green = "\x1b[32;20m"
    bold_red = "\x1b[31;1m"
    bold_white = "\x1b[37;1m"
    reset = "\x1b[0m"
    format = "%(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: format,
        logging.WARNING: bold_yellow + format + reset,
        31: reset + format + reset,
        32: green + format + reset,
        33: bold_white + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def parse_code_blob(code_blob: str) -> str:
    try:
        pattern = r"```(?:py|python)?\n(.*?)```"
        match = re.search(pattern, code_blob, re.DOTALL)
        return match.group(1).strip()
    except Exception as e:
        raise ValueError(
            f"The code blob you used is invalid: due to the following error: {e}. This means that the regex pattern {pattern} was not respected. Make sure to correct its formatting. Code blob was: {code_blob}"
        )


class CodeManager(Manager):
    def __init__(self, llm_engine, tools, additional_authorized_imports, **kwargs):
        super().__init__(llm_engine, tools, **kwargs)
        self.python_evaluator = evaluate_python_code
        self.additional_authorized_imports = additional_authorized_imports
        self.authorized_imports = list(set(LIST_SAFE_MODULES) | set(self.additional_authorized_imports))
        tool_descriptions = self.toolbox.show_tool_descriptions(DEFAULT_TOOL_DESCRIPTION_TEMPLATE)
        tool_names = [f"'{tool_name}'" for tool_name in self.toolbox.tools.keys()]
        self.system_prompts["action"] = NEXT_ACTION_CODE_PROMPT_SYSTEM.replace("<<tool_names>>", ", ".join(tool_names)).replace("<<tool_descriptions>>", tool_descriptions).replace("<<authorized_imports>>", str(self.authorized_imports))

    def action(self, prev_steps):
        action_prompt = [
            {
                "role": MessageRole.SYSTEM,
                "content": self.system_prompts["action"]
            },
            {
                "role": MessageRole.USER,
                "content": NEXT_ACTION_PROMPT_USER.format(
                    task=self.task, facts=self.state["facts"], prev_steps=prev_steps, plan=self.state["plan"]
                )
            }
        ]
        return self.llm_engine(action_prompt, stop_sequences="<end_code>")

    def log_code_action(self, code_action: str) -> None:
        self.logger.warning("==== Agent is executing the code below:")
        self.logger.log(
            31, highlight(code_action, PythonLexer(ensurenl=False), Terminal256Formatter(style="nord"))
        )

    def step(self):
        prev_steps, new_step = self.write_inner_memory_from_logs()
        self.logs.append({}) 
        try:
            self.summarize_facts(prev_steps, new_step)
        except Exception as e:
            raise AgentGenerationError(f"Error in generating llm output: {e}.")
        self.logger.debug(f"Facts: {self.state['facts']}")
        self.logs[-1]["facts"] = self.state["facts"]

        prev_steps += new_step
        try:
            self.plan(prev_steps)
        except Exception as e:
            raise AgentGenerationError(f"Error in generating llm output: {e}.")
        self.logger.debug(f"New plan: {self.state['plan']}")
        self.logs[-1]["plan"] = self.state["plan"]

        try:
            action = self.action(prev_steps)
        except Exception as e:
            raise AgentGenerationError(f"Error in generating llm output: {e}.")
        
        rationale, raw_code_action = self.extract_action(llm_output=action, split_token="Code:")

        try:
            code_action = parse_code_blob(raw_code_action)
        except Exception as e:
            raise AgentParsingError(f"Error in code parsing: {e}. Make sure to provide correct code")
        self.logs[-1]["tool_name"] = "code interpreter"
        self.logs[-1]["arguments"] = code_action
       
        self.log_code_action(code_action)
        try:
            available_tools = {**BASE_PYTHON_TOOLS.copy(), **self.toolbox.tools}
            result = self.python_evaluator(code_action, available_tools, state=self.state)
            information = self.state["print_outputs"]
            self.logger.warning("Print outputs:")
            self.logger.log(32, information)
            self.logs[-1]["observation"] = information
        except Exception as e:
            error_msg = f"Failed while trying to execute the code below:\n{CustomFormatter.reset + code_action + CustomFormatter.reset}\nThis failed due to the following error:\n{str(e)}"
            if "'dict' object has no attribute 'read'" in str(e):
                error_msg += "\nYou get this error because you passed a dict as input for one of the arguments instead of a string."
            raise AgentExecutionError(error_msg)
        for line in code_action.split("\n"):
            if line[: len("final_answer")] == "final_answer":
                self.logger.warning(">>> Final answer:")
                self.logger.log(32, result)
                return result
        return None

