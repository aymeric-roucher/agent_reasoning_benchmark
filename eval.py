import pandas as pd
import json

from gaia_scorer import question_scorer


OUTPUT_DIR = "gaia_output"
answer_file_path = f"{OUTPUT_DIR}/plan_exec_gpt4o_json_165_4.jsonl"

result_df = pd.read_json(answer_file_path, lines=True)
result_df["is_correct"] = result_df.apply(
    lambda x: question_scorer(x["prediction"], x["true_answer"]), axis=1
)

print(result_df[:20].groupby(["task"])["is_correct"].mean())
print(result_df["is_correct"].mean())