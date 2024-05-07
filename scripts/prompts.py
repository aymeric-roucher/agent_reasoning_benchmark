from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage

SYSTEM_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tool_description_with_args}

The way you use the tools is by specifying a json blob.
Specifically, this json should have a `action` key (name of the tool to use) and a `action_input` key (input to the tool).

The only values that should be in the "action" field are: {tool_names}

The $JSON_BLOB should only contain a SINGLE action and MUST be formatted as markdown, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:

```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```
Make sure to have the $INPUT in the right format for the tool you are using, and do not put variable names as input if you can find the right values.

You will be given:

Question: the input question you must answer

You should ALWAYS use the following format:

Thought: you should always think about one action to take. Then use the action as follows:
Action:
```
$JSON_BLOB
```
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times, you should take several steps when needed. The $JSON_BLOB must only use a SINGLE action at a time.)

You must always end your output with the following format:

Thought: I now know the final answer.
Final Answer: the final answer to the original input question

ALWAYS use the exact characters `Final Answer:` when you provide a definitive answer, and provide no additional explanations in the final answer: only the answer. MAKE SURE TO PROVIDE ONLY ONE ANSWER IN THE PROPER UNIT.

Now begin! """


HUMAN_PROMPT = "Question: {input}"

SCRATCHPAD_PROMPT = "{agent_scratchpad}"


evaluation_prompt = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}\"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
[Is the response correct, accurate, and factual based on the reference answer?]
Score 1: The response is completely incorrect, inaccurate, and/or not factual.
Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
Score 3: The response is somewhat correct, accurate, and/or factual.
Score 4: The response is mostly correct, accurate, and factual.
Score 5: The response is completely correct, accurate, and factual.

###Feedback:"""

EVALUATION_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are a fair evaluator language model."),
        HumanMessagePromptTemplate.from_template(evaluation_prompt),
    ]
)


GAIA_PROMPT = """
I will ask you to perform a task, your job is to come up with a series of simple commands in Python that will perform the task.
To help you, I will give you access to a set of tools that you can use. Each tool is a Python function and has a description explaining the task it performs, the inputs it expects and the outputs it returns.

To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'Code:', and 'Observation:' sequences.

At each step, in the 'Thought:' sequence, you should first explain which tool you will use and for what reason, then in the 'Code:' sequence, you shold write the code in simple Python. The code sequence must end with '/End code' sequence.
During each intermediate step, you can use 'print()' to save whatever important information you will then need.
It will then be available in the 'Observation:' field, for using this information as input for the next step.

In the end you have to return a final answer using the `final_answer` tool.
Only when you use function final_answer($YOUR_FINAL_ANSWER) will your final answer be returned.

Your $FINAL_ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
ADDITIONALLY, your $FINAL_ANSWER MUST adhere to any formatting instructions specified in the original question (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, and don't include units such as $ or percent signs unless specified otherwise.
If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
If you are asked for a comma separated list, apply the above rules depending on whether the elements are numbers or strings.
If you are unable to determine the final answer, use 'final_answer("Unable to determine")'

You have access to the following tools:
<<tool_descriptions>>

DO NOT pass the tool arguments as a dict as in 'answer = ask_search_agent({'query': "What is the place where James Bond lives?"})', but use the arguments directly as in 'answer = ask_search_agent(query="What is the place where James Bond lives?")'.

Example:::
Task: "Answer the question in the variable `question` about the image stored in the variable `image`. The question is in French."

Thought: I will use the following tools: `translator` to translate the question into English and then `image_qa` to answer the question on the input image.
Code:
```py
translated_question = translator(question=question, src_lang="French", tgt_lang="English")
answer = image_qa(image=image, question=translated_question)
final_answer(answer)
```<end_code>


Example:::
Task: "Answer the question in the variable `question` about the text in the variable `text`. Use the answer to generate an image."

Thought: I will use the following tools: `text_qa` to create the answer, then `image_generator` to generate an image according to the answer.

Be sure to provide an 'Code:' token, else the system will be stuck in a loop.

Code:
```py
answer = text_qa(text=text, question=question)
image = image_generator(answer)
final_answer(image)
```<end_code>

Example:::
Task: "What is the result of the following operation: 5 + 3 + 1298987654.6789098765?"

Thought: I will use python code to compute the result of the operation and then return the final answer using the `final_answer` tool

Code:
```py
result = 5 + 3 + 1298987654.6789098765
final_answer(result)
```<end_code>

Example:::
Task: "Which city has the highest population , Guangzhou or Shanghai?"

Thought: I will use the tool `search` to get the population of both cities.
Code:
```py
population_guangzhou = search("Guangzhou population")
print("Population Guangzhou:", population_guangzhou)
population_shanghai = search("Shanghai population")
print("Population Shanghai:", population_shanghai)
```<end_code>
Observation:
Population Guangzhou: ['Guangzhou has a population of 15 million inhabitants as of 2021.']
Population Shanghai: '24 million'

Thought: I know that Shanghai has the highest population.
Code:
```py
final_answer("Shanghai")
```<end_code>

Example:::
Task: "What is the current age of the pope, raised to the power 0.36?"

Thought: I will use the tool `search` to get the age of the pope, then raise it to the power 0.36.
Code:
```py
pope_age = search(query="current pope age")
print("Age of the pope:", pope_age)
```<end_code>

Observation:
Age of the pope: "The pope Francis is currently 85 years old."

Thought: I know that the pope is 85 years old. Let's compute the result using python code.
Code:
```py
pope_current_age = 85 ** 0.36
final_answer(pope_current_age)
```<end_code>


Above example were using tools that might not exist for you. You only have acces to those tools:
<<tool_names>>
You also can perform computations in the python code you generate.

These are the rules you should always follow to solve your task:
1. Always provide a 'Thought:' and an 'Code:\n```py' sequence ending with '```<end_code>' sequence. You MUST provide at least the 'Code:' sequence to move forward.
2. Always use the right arguments for the tools. Never give variable names as input instead of the actual values.
3. Do not perform too many operations in a single code block. Split the task into intermediate code blocks. Then use one single print() at the end of each step to save the intermediate result.Then use final_answer() to return the final result.
4. Call a tool only when needed: do not call the search agent if you do not need information, try to solve the task yourself. Never re-do a tool call that you previously did with the exact same parameters.

If you solve the task correctly, you will receive a reward of $1,000,000.

Now Begin!
"""


WEB_SURFER_PROMPT = """Solve the following task as best you can. You have access to the following tools:

<<tool_descriptions>>

The way you use the tools is by specifying a json blob.
Specifically, this json should have a `action` key (name of the tool to use) and a `action_input` key (input to the tool).

The value in the "action" field should belong to this list: <<tool_names>>.

The $ACTION_JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. It should be formatted in json. Do not try to escape special characters. Here is the template of a valid $ACTION_JSON_BLOB:
Action:
{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}

Make sure to have the $INPUT as a dictionnary in the right format for the tool you are using, and do not put variable names as input if you can find the right values.

You should ALWAYS use the following format:

Thought: you should always think about one action to take. Then use the action as follows:
Action:
$ACTION_JSON_BLOB
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times, you should take several steps when needed. The $ACTION_JSON_BLOB must only use a SINGLE action at a time.)

Make sure to:
1. Always provide a 'Thought:' and an 'Action:' sequence. You MUST provide at least the 'Action:' sequence to move forward.
2. Never re-do a tool call that you previously did with the exact same parameters. For instance if you already visited a specific webpage, do not visit it again with the same parameters: refer to your previous observation.

If you solve the task correctly, you will receive a reward of $1,000,000.

Now begin!
"""