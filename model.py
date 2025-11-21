
import json
from PIL import Image
from transformers import AutoProcessor, CLIPVisionModelWithProjection # hugging face extension to simplify workflow
import requests
import torch

#-------------------------------------------------------------------------#

model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch16") # loads the model that converts image to vector
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16") # handling the data to feed into the model?


def clip_embed(model, processor):
    with open('yalies.json', 'r') as f:
        yalies = json.load(f)

    image_urls = [yalie.get('image') for yalie in yalies]

    # loading in images
    images = [Image.open(requests.get(url, stream= True).raw) for url in image_urls ] # may cause memory issues down the line
    image_tensors = processor(images=images, return_tensors="pt") # converting image to format model can process

    # running model and retrieving results
    with torch.inference_mode():
        outputs = model(**image_tensors)
    image_embeds = outputs.image_embeds

    # combine with original json to build payload
    payloads = []
    for i in range(0, len(yalies)):
        yalies[i]['embedding'] = image_embeds[i].tolist()
        payloads.append(yalies[i])

    with open('yalie_embedding.json', 'w') as f:
        json.dump(payloads, f, indent= 4)

clip_embed(model, processor)
        
