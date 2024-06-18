from transformers import AutoProcessor
from huggingface_hub import InferenceClient
from transformers.agents import Tool, HfEngine
from PIL import Image
from typing import Optional
import base64
import mimetypes
import json
import os
import requests
import uuid


def encode_image(image_path):
    if image_path.startswith("http"):
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
        request_kwargs = {
            "headers": {"User-Agent": user_agent},
            "stream": True,
        }

        # Send a HTTP request to the URL
        response = requests.get(image_path, **request_kwargs)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")

        extension = mimetypes.guess_extension(content_type)
        if extension is None:
            extension = ".download"

        fname = str(uuid.uuid4()) + extension
        download_path = os.path.abspath(os.path.join("downloads", fname))

        with open(download_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=512):
                fh.write(chunk)
        image_path = download_path

    content_type = mimetypes.guess_type(image_path)[0]
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8'), content_type


def resize_image(image_path):
    img = Image.open(image_path)
    width, height = img.size
    img = img.resize((int(width / 2), int(height / 2)))
    new_image_path = f"resized_{image_path}"
    img.save(new_image_path)
    return new_image_path


headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
}


class VisualQATool(Tool):
    name = "visual_qa"
    description = "A tool to answer questions about images."
    inputs = {
        "question": {"description": "the question to answer", "type": "text"},
        "image_path": {
            "description": "the path to the image",
            "type": "text",
        },
    }
    output_type = "text"

    def __init__(
            self,
            model_name: str = "HuggingFaceM4/idefics2-8b-chatty",
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.client = InferenceClient(model_name)

    def forward(self, question: str, image_path: str) -> str:
        try:
            return self.process_images_and_text(image_path, question)
        except Exception as e:
            if "Payload Too Large" in str(e):
                new_image_path = resize_image(image_path)
                return self.process_images_and_text(new_image_path, question)
            raise RuntimeError(str(e))

    def process_images_and_text(self, image_path, query):
        messages = [
            {
                "role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": query},
                ]
            },
        ]

        prompt_with_template = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        content_type = mimetypes.guess_type(image_path)[0]

        with open(image_path, "rb") as i:
            image_string = base64.b64encode(i.read()).decode()

        image_string = f"data:{content_type};base64,{image_string}"
        prompt_with_images = prompt_with_template.replace("<image>", "![]({})").format(image_string)

        payload = {
            "inputs": prompt_with_images,
            "parameters": {
                "return_full_text": False,
                "max_new_tokens": 200,
            }
        }

        return json.loads(self.client.post(json=payload).decode())[0]


class VisualQAGPT4Tool(Tool):
    name = "visual_qa"
    description = "A tool to answer questions about images."
    inputs = {
        "image_path": {
            "description": "The path to the image",
            "type": "text",
        },
        "question": {"description": "the question to answer", "type": "text"},
    }
    output_type = "text"

    def forward(self, image_path: str, question: Optional[str] = None) -> str:
        add_note = False
        if not question:
            add_note = True
            question = "Please write a detailed caption for this image."

        base64_image, content_type = encode_image(image_path)

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{content_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 500
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        output = response.json()['choices'][0]['message']['content']

        if add_note:
            output = f"You did not provide a question, so here is a detailed caption for the image: {output}"

        return output


