import pymongo 
import json
from pymongo import UpdateOne 


# at highest level we have dbs
# under dbs we have collections
# collections is where all our individual entries are stored

# Establishing conneciton to the db
client = pymongo.MongoClient('mongodb+srv://michaelleeml3267_db_user:VH2iXNRMQRrpLA9p@yalie-search.czvtllb.mongodb.net/')
db = client['yalies'] 

clip_embedding = db['clip_embedding'] # establishing connection to this collection


def upload_data():
    # Loading in data 
    with open('yalie_embedding.json', 'r') as f:
        json_data = json.load(f)

    # 1. Create a list of UpdateOne requests
    requests = []

    for document in json_data:
        # Use a unique identifier (like '_id') for the filter
        filter_spec = {'id': document['id']} 
        
        # Use the $set operator to replace or add all fields in the document
        update_spec = {'$set': document} 
        
        # Create the UpdateOne object and set upsert=True
        requests.append(
            UpdateOne(
                filter_spec,
                update_spec,
                upsert=True  # THIS is where you enable upserting for each document
            )
        )

    # Uploading json to collection
    result = clip_embedding.bulk_write(requests, ordered= False)
    print(f"Total documents inserted (upserted): {result.upserted_count}")
    print(f"Total documents modified: {result.modified_count}")

def clear_data():
    pass

upload_data()

