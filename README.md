# Yalie Search

Ever wanted to find fine shyt on campus without having to talk to them irl? See shawty walking around on campus but too scared to ask for their name and contact? 

**I gotchu**

This app aims to use machine learning to identify people's facial features through semantic queries. Prompts can be:

### Descriptive
1. "Asian woman with glasses"
2. "White man with black hair"
3. "Hispanic, wearing glasses"

### Identity-anchored
1. "Tom Brady but black"
2. "Taylor Swift if she was a man"
3. "Keanu Reeves when he was 20 years old"

### Subjective
1. "Sexiest, most attractive person"
2. "Looks like they give warm hugs"
3. "Most likely to succeed"

There will a leaderboard posting top search results for each week. Let the stalking begin... >:)



## Documentation

These scripts use CLIP Embedding and a euclidean vector search to generate top results.

Pipeline (run scripts in this order):
1. data_prep.py makes API call to yalies.io to retrieve information from each individual student and outputs json file 'yalies.json'
2. model.py reads 'yalies.json', reads the image associated with each person, embeds the photo, and creates json file with the embedding for each person attached
3. db_upload reads in 'yalie_embedding.json' and uploads it to a mongodb database
4. testing_results.py is where you can search for the top n photo results by changing the input query



References:
1. https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/clip#transformers.CLIPVisionModel
2. https://www.w3schools.com/python/python_mongodb_insert.asp