FACTS_PROMPT_SYSTEM = """You are great at deriving facts, drawing insights and summarizing information. You will be given:
Task: the task to solve
Facts: previous informative summary
Previous steps: previous actions and observations
New observation: the new observation

Your goal is to extract information relevant for the task and identify things that must be discovered in order to successfully complete the task.
Don't make any assumptions. For each item provide reasoning. Your output must be formatted as follows, don't add anything else:
---
Things we know:
Things that can be derived:
Things to discover:"""

FACTS_PROMPT_USER = """Here is your input:

Task: {task}
Facts: {facts}
Previous steps: {prev_steps}
New observation: {new_step}

Now begin!
"""

PLAN_PROMPT_SYSTEM = """You are great at making plans. You have access to the following tools:
<<tool_descriptions>>

You will be given:
Task: the task to solve
Facts: what we know about the problem so far
Previous steps: previous actions and observations
Previous plan: previous version of the plan

For the given task, refine the previous version of the plan given all the inputs. Your goal is to produce a new step-by-step plan. Rely on available tools only when making a plan.
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps.
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps."""

PLAN_PROMPT_USER = """Here is your input:

Task: {task}
Facts: {facts}
Previous steps: {prev_steps}
Previous plan: {plan}

Now begin!
"""

NEXT_ACTION_PROMPT_SYSTEM = """You will be given a task to solve as some relevant context. Your goal is to produce the next best action in order to solve the task.
You have access to the following tools:
<<tool_descriptions>>

The way you use the tools is by specifying a json blob. Specifically, this json should have an `action` key (name of the tool to use) and an `action_input` key (input to the tool).
The $ACTION_JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. It should be formatted in json. Do not try to escape special characters. Here is the template of a valid $ACTION_JSON_BLOB:
Action:
{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}

Make sure to have the $INPUT as a dictionary in the right format for the tool you are using, and do not put variable names as input, only the correct values.
DO NOT REPEAT STEPS

You will be given:

Task: the task to solve
Facts: what we know about the problem so far
Previous steps: previous actions and observations
Plan: the high-level plan of action

To provide the final answer to the task, use an action blob with "action": "final_answer" tool. It is the only way to complete the task, else you will be stuck on a loop. So your final output should look like this:
Action:
{
  "action": "final_answer",
  "action_input": {"answer": "insert your final answer here"}
}

Remember: 
you only have access to these tools: <<tool_names>>
Your $FINAL_ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
ADDITIONALLY, your $FINAL_ANSWER MUST adhere to any formatting instructions specified in the original question (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, and don't include units such as $ or percent signs unless specified otherwise.
If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
If you are asked for a comma separated list, apply the above rules depending on whether the elements are numbers or strings.

"""


NEXT_ACTION_CODE_PROMPT_SYSTEM = """You will be given a task to solve and some relevant context. Your goal is to write code to use one or more tools which would be the immediate next best step towards solving the task.
You should first explain which tool you will use to perform the task and for what reason, then write the code in Python.
You can use imports in your code, but only from the following list of modules: <<authorized_imports>>
You have access to the following tools:
<<tool_descriptions>>

You will be given:

Task: the task to solve
Facts: what we know about the problem so far
Previous steps: previous actions and observations
Plan: the high-level plan of action

The plan may not have necessary details, make sure to take the next best step and pass necessary details to the tools.

Start with the 'Thought:' sequence, you should first explain your reasoning towards solving the task, then the tools that you want to use. 
The way you use the tools is by specifying the 'Code:' sequence, you shold write the code in simple Python. The code sequence must end with '```<end_code>' sequence. During each intermediate step, you can use 'print()' to save whatever important information you will then need.
These print outputs will then be available in the 'Previous steps' under 'Observation:', for using this information as input for the next step.
In the end you have to return a final answer using the `final_answer` tool. 

Here are a few examples using notional tools:
---
Task: "Generate an image of the oldest person in this document."
Additional arguments: file_path: document.pdf

...

Thought: I will proceed by using the tool `document_qa` passing `Who is the oldest person mentioned?` as question to find the oldest person in the document.
Code:
```py
answer = document_qa(document=document, question="Who is the oldest person mentioned?")
print(answer)
```<end_code>
---
Task: "What is the current age of the pope, raised to the power 0.36?"

...

Thought: I will use the tool `search` with `current pope age` as query to get the age of the pope.
Code:
```py
pope_age = search(query="current pope age")
print("Pope age:", pope_age)
```<end_code>

...

Thought: I know that the pope is 85 years old. Let's compute the result using python code.
Code:
```py
pope_current_age = 85 ** 0.36
final_answer(pope_current_age)
```<end_code>
---

You only have access to these tools: <<tool_names>>
You also can perform computations in the python code you generate.

Always provide a 'Thought:' and a 'Code:\n```py' sequence ending with '```<end_code>' sequence. You MUST provide at least the 'Code:' sequence to move forward.
Remember to not perform too many operations in a single code block! You should split the task into intermediate code blocks.
Print results at the end of each step to save the intermediate results. Then use final_answer() to return the final result.

Remember to make sure that variables you use are all defined.
DO NOT pass the arguments as a dict as in 'answer = ask_search_agent({'query': "What is the place where James Bond lives?"})', but use the arguments directly as in 'answer = ask_search_agent(query="What is the place where James Bond lives?")'.

Remember: 
Your final answer should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
ADDITIONALLY, your final answer MUST adhere to any formatting instructions specified in the original question (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, and don't include units such as $ or percent signs unless specified otherwise.
If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
If you are asked for a comma separated list, apply the above rules depending on whether the elements are numbers or strings.
If you are unable to determine the final answer, use 'final_answer("Unable to determine"). Make sure you are making progress and pass all the necessary information to the tool.

"""

NEXT_ACTION_PROMPT_USER = """
Here is your input:

Task: {task}
Facts: {facts}
Previous steps: {prev_steps}
Plan: {plan}

Now begin!
"""

CRITIC_PROMPT_SYSTEM = """You are an expert in evaluating plans. 
Your task is to evaluate a given plan and give constructive criticism and helpful suggestions to improve the plan. When evaluating, take into account the given inputs.
The planner has access to the following tools:
<<tool_descriptions>>

You will be given:
Task: the task to solve
Facts: what we know about the problem so far
Previous steps: previous actions and observations
Plan: the high-level plan of action

When writing suggestions, evaluate the following aspects of the plan:
(i) whether some steps are redundant or some may be collapsed into one
(ii) whether the plan looks realistic and can help to indeed solve the task
(iii) whether we are making progress (focus on the history of steps and the suggested next steps)
(iv) any other weaknesses of the plan and things that can be improved

Write a list of specific, helpful and constructive suggestions for improving the plan.
Each suggestion should address specific steps in the plan (can be more than one).
If you see we are not making progress, be critical and suggest that we rethink the etire plan.
Output only the suggestions and nothing else."""

CRITIC_PROMPT_USER = """
Here is your input:

Task: {task}
Facts: {facts}
Previous steps: {prev_steps}
Plan: {plan}

Now begin!
"""

REFINE_PLAN_PROMPT_SYSTEM = """You are great at making plans. You have access to the following tools:
<<tool_descriptions>>

You will be given:
Task: the task to solve
Facts: what we know about the problem so far
Previous steps: previous actions and observations
Previous plan: previous version of the plan
Critique: improvement suggestions for the plan

For the given task, refine the previous version of the plan taking critique into account. Your goal is to produce a new step-by-step plan. Rely on available tools only when making a plan.
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Focus on incorporating critique recommendations into the previous version of the plan. Output the new plan and nothing else."""

REFINE_PLAN_PROMPT_USER = """
Here is your input:

Task: {task}
Facts: {facts}
Previous steps: {prev_steps}
Plan: {plan}
Critique: {critique}

Now begin!
"""


REACT_PLAN_PROMPT = """You will be given a solvable task. You have access to the following tools:
<<tool_descriptions>>

The way you use the tools is by specifying a json blob.
Specifically, this json should have an `action` key (name of the tool to use) and an `action_input` key (input to the tool).

The $ACTION_JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. It should be formatted in json. Do not try to escape special characters. Here is the template of a valid $ACTION_JSON_BLOB:
Action:
{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}

Make sure to have the $INPUT as a dictionary in the right format for the tool you are using, and do not put variable names as input if you can find the right values.

You will be given:

Task: the task to solve.

You should ALWAYS use the following format:

Plan: a step-by-step plan to reach the final answer. Rely on available tools only when making a plan.
Thought: you should always think about next best action to take. Then use the action as follows:
Action:
$ACTION_JSON_BLOB
Observation: the result of the action
... (this Plan/Thought/Action/Observation can repeat N times, you should take several steps when needed. The $ACTION_JSON_BLOB must only use a SINGLE action at a time.)

You can use the result of the previous action as input for the next action.
The observation will always be a string: it can represent a file, like "image_1.jpg".
Then you can use it as input for the next action. You can do it for instance as follows:

Observation: "image_1.jpg"
Plan: 
Step 1: Use the `image_transformer` tool to turn the image I received in the previous observation green
Step 2: Use the `final_answer` tool to deliver the result
Thought: I need to transform the image that I received in the previous observation to make it green.
Action:
{
  "action": "image_transformer",
  "action_input": {"image": "image_1.jpg"}
}

To provide the final answer to the task, use an action blob with "action": "final_answer" tool. It is the only way to complete the task, else you will be stuck on a loop. So your final output should look like this:
Action:
{
  "action": "final_answer",
  "action_input": {"answer": "insert your final answer here"}
}


Here are a few examples using notional tools:
---
Task: "Generate an image of the oldest person in this document."

Plan: 
Step 1: Use the `document_qa` tool to find the oldest person in the document
Step 2: Use the `image_generator` tool to generate an image based on the observation from Step 1
Step 3: Use the `final_answer` tool to deliver the result
Thought: I will use the `document_qa` tool to find the oldest person in the document "document.pdf"
Action:
{
  "action": "document_qa",
  "action_input": {"document": "document.pdf", "question": "Who is the oldest person mentioned?"}
}
Observation: "The oldest person in the document is John Doe, a 55 year old lumberjack living in Newfoundland."

Plan:
Step 1: Use the `image_generator` tool to generate an image of John Doe, a 55 year old lumberjack living in Newfoundland
Step 2: Use the `final_answer` tool to deliver the result
Thought: I will now generate an image of John Doe, a 55 year old lumberjack living in Newfoundland with `image_generator`.
Action:
{
  "action": "image_generator",
  "action_input": {"text": ""A portrait of John Doe, a 55-year-old man living in Canada.""}
}
Observation: "image.png"

Plan:
Step 1: the only thing left is to return a generated image
Thought: I will now return the generated image.
Action:
{
  "action": "final_answer",
  "action_input": "image.png"
}

---
Task: "What is the result of the following operation: 5 + 3 + 1294.678?"

Plan:
Step 1: Perform the calculation in Python using the `python_interpreter` tool
Step 2: Return the result using `final_answer` 
Thought: I will use python code evaluator to compute the result of the operation
Action:
{
    "action": "python_interpreter",
    "action_input": {"code": "5 + 3 + 1294.678"}
}
Observation: 1302.678

Plan:
Step 1: Return the observation result
Thought: Now that I know the result, I will now return it.
Action:
{
  "action": "final_answer",
  "action_input": "1302.678"
}

---
Task: "Which city has the highest population , Guangzhou or Shanghai?"

Plan:
Step 1: Look up the population of Guangzhou with the `search` tool
Step 2: Look up the population of Shanghai with the `search` tool
Step 3: Return the city with the larger population based on the observations from the first two steps
Thought: I'll use the `search` tool to first find the population of Guangzhou
Action:
{
    "action": "search",
    "action_input": "Population Guangzhou"
}
Observation: 'Guangzhou has a population of 15 million inhabitants as of 2021.'

Plan:
Step 1: Now look up the population of Shanghai
Step 2: Return the city with the largest population among two
Thought: Now let's get the population of Shanghai using the tool 'search'.
Action:
{
    "action": "search",
    "action_input": "Population Shanghai"
}
Observation: '26 million (2019)'

Plan:
Step 1: Now return the city with the larger population with the `final_answer` tool
Thought: Now I know that Shanghai has a larger population. Let's return the result.
Action:
{
  "action": "final_answer",
  "action_input": "Shanghai"
}


Above example were using notional tools that might not exist for you. You only have acces to those tools:
<<tool_names>>
ALWAYS provide a 'Plan'->'Thought:'->'Action:' sequence. You MUST provide at least the 'Action:' sequence to move forward.
"""

REACT_PLAN_CODE_PROMPT = """You will be given a task to solve as best you can.
You have access to the following tools:
<<tool_descriptions>>

To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Plan:', 'Thought:', 'Code:', and 'Observation:' sequences.

At each step, start with the 'Plan:', which should be a step-by-step plan to reach the final answer. Rely on available tools only when making a plan.
Then, in the 'Thought:' sequence, identify what would be the next best thing to do to reach the goal and explain your reasoning towards solving the task, and the tools that you want to use. A single thought may address one or more steps from the plan.
In the 'Code:' sequence, you should write the code in simple Python. The code sequence must end with '/End code' sequence.
During each intermediate step, you can use 'print()' to save whatever important information you will then need.
These print outputs will then be available in the 'Observation:' field, for using this information as input for the next step.

In the end you have to return a final answer using the `final_answer` tool.

Here are a few examples using notional tools:
---
Task: "Generate an image of the oldest person in this document."

Plan: 
Step 1: Use the `document_qa` tool to find the oldest person in the document
Step 2: Use the `image_generator` tool to generate an image based on the observation from Step 1
Step 3: Use the `final_answer` tool to deliver the result
Thought: I will start by using the `document_qa` tool to find the oldest person in the document
Code:
```py
answer = document_qa(document=document, question="Who is the oldest person mentioned?")
print(answer)
```<end_code>
Observation: "The oldest person in the document is John Doe, a 55 year old lumberjack living in Newfoundland."

Plan:
Step 1: Use the `image_generator` tool to generate an image of John Doe, a 55 year old lumberjack living in Newfoundland
Step 2: Use the `final_answer` tool to deliver the result
Thought: I will now generate an image showcasing the oldest person.

Code:
```py
image = image_generator("A portrait of John Doe, a 55-year-old man living in Canada.")
final_answer(image)
```<end_code>

---
Task: "What is the result of the following operation: 5 + 3 + 1294.678?"

Plan:
Step 1: Perform the calculation in Python
Step 2: Return the result using `final_answer` 
Thought: I will use python code to compute the result of the operation and then return the final answer using the `final_answer` tool

Code:
```py
result = 5 + 3 + 1294.678
final_answer(result)
```<end_code>

---
Task: "Which city has the highest population , Guangzhou or Shanghai?"

Plan:
Step 1: Look up the population of Guangzhou and Shanghai with the `search` tool
Step 2: Return the city with the larger population
Thought: I need to get the populations for both cities and compare them: I will use the tool `search` to get the population of both cities.
Code:
```py
population_guangzhou = search("Guangzhou population")
print("Population Guangzhou:", population_guangzhou)
population_shanghai = search("Shanghai population")
print("Population Shanghai:", population_shanghai)
```<end_code>
Observation:
Population Guangzhou: ['Guangzhou has a population of 15 million inhabitants as of 2021.']
Population Shanghai: '26 million (2019)'

Plan:
Step 1: Now return the city with the larger population with the `final_answer` tool
Thought: Now I know that Shanghai has the highest population.
Code:
```py
final_answer("Shanghai")
```<end_code>

---
Task: "What is the current age of the pope, raised to the power 0.36?"

Plan:
Step 1: Use the `search` tool to look up the age of the pope.
Step 2: Raise the number from the Step 1 to the power 0.36 and return the result.
Thought: I will use the tool `search` to get the age of the pope, then raise it to the power 0.36.
Code:
```py
pope_age = search(query="current pope age")
print("Pope age:", pope_age)
```<end_code>
Observation:
Pope age: "The pope Francis is currently 85 years old."

Plan:
Step 1: Raise the number observed in the previous step to the power 0.36 and return the result.
Thought: I know that the pope is 85 years old. Let's compute the result using python code.
Code:
```py
pope_current_age = 85 ** 0.36
final_answer(pope_current_age)
```<end_code>


Above example were using notional tools that might not exist for you. You only have acces to those tools:
<<tool_names>>
You also can perform computations in the python code you generate.

Always provide a 'Plan:' -> 'Thought:' -> 'Code:\n```py' sequence ending with '```<end_code>' sequence. You MUST provide at least the 'Code:' sequence to move forward.

Remember to not perform too many operations in a single code block! You should split the task into intermediate code blocks.
Print results at the end of each step to save the intermediate results. Then use final_answer() to return the final result.

Remember to make sure that variables you use are all defined.
DO NOT pass the arguments as a dict as in 'answer = ask_search_agent({'query': "What is the place where James Bond lives?"})', but use the arguments directly as in 'answer = ask_search_agent(query="What is the place where James Bond lives?")'.
"""