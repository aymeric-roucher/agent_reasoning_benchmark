from PIL import Image
import base64
from io import BytesIO
import json
import os
import requests

from huggingface_hub import InferenceClient
from transformers import AutoProcessor, Tool

idefics_processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")

def process_images_and_text(image_path, query, client):
    messages = [
        {
            "role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": query},
            ]
        },
    ]

    prompt_with_template = idefics_processor.apply_chat_template(messages, add_generation_prompt=True)

    # load images from local directory

    # encode images to strings which can be sent to the endpoint
    def encode_local_image(image_path):
        # load image
        image = Image.open(image_path).convert('RGB')

        # Convert the image to a base64 string
        buffer = BytesIO()
        image.save(buffer, format="JPEG")  # Use the appropriate format (e.g., JPEG, PNG)
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # add string formatting required by the endpoint
        image_string = f"data:image/jpeg;base64,{base64_image}"

        return image_string
    

    image_string = encode_local_image(image_path)
    prompt_with_images = prompt_with_template.replace("<image>", "![]({}) ").format(image_string)


    payload = {
        "inputs": prompt_with_images,
        "parameters": {
            "return_full_text": False,
            "max_new_tokens": 200,
        }
    }

    return json.loads(client.post(json=payload).decode())[0]


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
}


def resize_image(image_path):
    img = Image.open(image_path)
    width, height = img.size
    img = img.resize((int(width / 2), int(height / 2)))
    new_image_path = f"resized_{image_path}"
    img.save(new_image_path)
    return new_image_path


class VisualQATool(Tool):
    name = "visualizer"
    description = "A tool that can answer questions about attached images."
    inputs = {
        "question": {"description": "the question to answer", "type": "text"},
        "image_path": {
            "description": "The path to the image on which to answer the question",
            "type": "text",
        },
    }
    output_type = "text"

    client = InferenceClient("HuggingFaceM4/idefics2-8b-chatty")

    def forward(self, question: str, image_path: str) -> str:
        try:
            return process_images_and_text(image_path, question, self.client)
        except Exception as e:
            print(e)
            if "Payload Too Large" in str(e):
                new_image_path = resize_image(image_path)
                return process_images_and_text(new_image_path, question, self.client)


class VisualQAGPT4Tool(Tool):
    name = "visualizer"
    description = "A tool that can answer questions about attached images."
    inputs = {
        "question": {"description": "the question to answer", "type": "text"},
        "image_path": {
            "description": "The path to the image on which to answer the question",
            "type": "text",
        },
    }
    output_type = "text"

    def forward(self, question: str, image_path: str) -> str:
        base64_image = encode_image(image_path)

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
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                    }
                ]
                }
            ],
            "max_tokens": 300
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        return response.json()['choices'][0]['message']['content']


