from huggingface_hub import InferenceClient
from transformers import AutoProcessor
from PIL import Image
import base64
from io import BytesIO
import json


processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")

def process_images_and_text(image_path, query, client):
    messages = [
        {
            "role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": query},
            ]
        },
    ]

    prompt_with_template = processor.apply_chat_template(messages, add_generation_prompt=True)

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
    
    print(image_path)

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