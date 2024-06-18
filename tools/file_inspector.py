from transformers.agents.default_tools import Tool
from transformers.agents import HfEngine
from pypdf import PdfReader
from markdownify import markdownify as md
from tools.mdconvert import MarkdownConverter


def extract_text_from_pdf(pdf_path):
    pdf = PdfReader(pdf_path)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return md(text)


class FileInspectorTool(Tool):
    name = "inspect_file_as_text"
    description = """
You cannot load files yourself: instead call this tool to read a file as markdown text and ask questions about it.
This tool handles the following file extensions: [".html", ".htm", ".xlsx", ".pptx", ".pdf", ".docx"], and all other types of text files. IT DOES NOT HANDLE IMAGES."""

    inputs = {
        "question": {
            "description": "Your question, as a natural language sentence. Provide as much context as possible.",
            "type": "text",
        },
        "file_path": {
            "description": "The path to the file you want to read as text. Must be a '.something' file, like '.pdf'. If it is an image, use the visualizer tool instead!",
            "type": "text",
        },
    }
    output_type = "text"
    md_converter = MarkdownConverter()

    def __init__(
            self,
            llm_engine,
            description=None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.engine = llm_engine
        if description:
            self.description = description

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
        return self.engine(messages)