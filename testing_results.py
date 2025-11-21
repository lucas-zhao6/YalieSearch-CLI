from transformers import AutoTokenizer, CLIPTextModelWithProjection
import torch
import pymongo
from PIL import Image
from io import BytesIO
import requests


# Establishing conneciton to the db
client = pymongo.MongoClient('mongodb+srv://michaelleeml3267_db_user:VH2iXNRMQRrpLA9p@yalie-search.czvtllb.mongodb.net/')
db = client['yalies'] 
clip_embedding = db['clip_embedding'] # establishing connection to this collection

# Initiating model and tokenizer
model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

inputs = tokenizer("person with light skin and dark hair", padding=True, return_tensors="pt") # CHANGE THIS LINE FOR

with torch.inference_mode():
    outputs = model(**inputs)
text_embeds = outputs.text_embeds
query_vector = text_embeds.squeeze().tolist()


k_neighbors = 3  # Number of nearest neighbors to retrieve

pipeline = [
    {
        "$vectorSearch": {
            "index": "default",
            "path": "embedding", # index fields on mongodb, found under vector search page
            "queryVector": query_vector,
            "numCandidates": 100,  # Number of candidates to consider for KNN
            "limit": k_neighbors
        }
    }
]

results = clip_embedding.aggregate(pipeline)
for yalie in results:
    url = yalie.get('image')
    response = requests.get(url)
    response.raise_for_status()

    img = Image.open(BytesIO(response.content))
    img.show()