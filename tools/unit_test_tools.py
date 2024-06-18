import logging
import os
from dotenv import load_dotenv

load_dotenv("../.env", override=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

from tools import (
    VisualQATool, SearchTool, TextInspectorTool, AudioQATool

)

vqa_tool = VisualQATool()
question = "What is the title of the oldest Blu-Ray recorded in this spreadsheet?"
logger.info(f"VisualQATool question: {question}\nAnswer:")
logger.info(
    vqa_tool(
        question=question,
        image_path=os.getcwd() + "/samples/32102e3e-d12a-4209-9163-7b3a104efe5d.png"
    )
)
logger.info("Ground truth: Time-Parking 2: Parallel Universe")

search_tool = SearchTool()
question = "When was Steve Jobs born?"
logger.info(f"SearchTool question: {question}\nOutput:")
logger.info(
    search_tool(question)
)
logger.info("Ground truth: February 24, 1955")

inspector_tool = TextInspectorTool()
question = "List the authors of the paper"
logger.info(f"TextInspectorTool question: {question}\nAnswer:")
logger.info(
    inspector_tool(question=question, file_path="https://arxiv.org/pdf/1706.03762")
)
logger.info("Ground truth: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, "
            "Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin")

aqa_tool = AudioQATool(device="cpu")
question = "When is the midterm?"
logger.info(f"AudioQATool question: {question}\nAnswer:")
logger.info(
    aqa_tool(
        question=question,
        audio_path=os.getcwd() + "/samples/1f975693-876d-457b-a649-393859e79bf3.mp3"
    )
)
logger.info("Ground truth: next week")
