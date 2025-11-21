import pandas as pd
import numpy as np
import time
import requests
import json
import os

# The point of this script is to retreive all images from yalies and prepare to convert them into vector embeddings

# retrieving info from all yale people >:)
# https://github.com/YaleComputerSociety/Yalies/wiki/API-Documentation

def scrape_yalies(n_pages= 1):
    all_people = []
    for i in range(1, n_pages + 1):
        headers = {
            "Authorization": "Bearer d8szIfAZepJlvKUB8wWfyfLRN9uEi7QoVGk5RHMhMwmPXEx1ctyWcA"
        }
        body = {
            "query": "", 
            'page_size': 100,
            'page': i
        }
        request = requests.post("https://api.yalies.io/v2/people", headers=headers, json=body)
        request_data = request.json()
        request_data = [yalie for yalie in request_data if yalie.get('image') is not None]
        #person_data = [{'netid': person.get('netid'), 'image': person.get('image')} for person in request_data]
    
        all_people.extend(request_data)

    with open('yalies.json', 'w') as f:
        json.dump(all_people, f, indent= 4)

    return(all_people)

scrape_yalies()




