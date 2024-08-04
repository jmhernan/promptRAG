# Example Script
import os
from os import listdir
from os.path import isfile, join

import operator
import json

import pandas as pd
import numpy as np

import gensim
import gensim.downloader as gensim_api
import transformers

from transformers import AutoTokenizer, AutoModel
import torch
from datasets import Dataset

import src.text_preprocess as tp
from src.create_vdb import IndexTextEmbeddings
from src.prompt_generator import TweetPromptGenerator


this_file_path = os.path.abspath(os.getcwd()) # parse this out so that it works
# get project root location
project_root = os.path.split(os.path.split(this_file_path)[0])[0]
data_dir = os.path.join(project_root, 'data/')
tfidf_data_dir = os.path.join(project_root, 'data/tfidf/')

poynter_raw = pd.read_csv(os.path.join(data_dir, 'poynter_coded_breon_tab.csv'), encoding='utf8')

poynter_text = poynter_raw['story_copy']
text = poynter_text.apply(tp.clean_text)
poynter_raw['clean_text'] = np.array(text)

create_vector_df = IndexTextEmbeddings(model_name='sentence-transformers/multi-qa-mpnet-base-dot-v1')
dataset = create_vector_df.create_dataset(poynter_raw, 'clean_text')
dataset_with_index = create_vector_df.add_faiss_index(dataset, 'embeddings')

tweet_prompt_generator = TweetPromptGenerator(dataset_with_index, 'sentence-transformers/multi-qa-mpnet-base-dot-v1', project_root)

misinformed_tweets = [
    "Drinking lemon juice cures COVID in 24 hours.",
    "COVID-19 was created as a form of population control.",
    "CDC says masks increase your chances of getting COVID.",
    "Vaccines don’t protect you; all my vaccinated friends got sick.",
    "My neighbor got the vaccine and turned into a zombie.",
    "You can catch COVID-19 from talking on the phone with an infected person.",
    "COVID-19 was brought to Earth by aliens.",
    "Once you recover from COVID, you turn invisible to viruses.",
    "Government says COVID can be defeated by singing the national anthem."
]

# clean up first
def clean_text_list(text_list):
    return [tp.clean_text(text) for text in text_list]

misinformed_tweets = clean_text_list(misinformed_tweets)

prompts, stories, themes, similarity_scores = tweet_prompt_generator.generate_prompts_for_tweets(misinformed_tweets, clean_tweets=False)

for prompt in prompts:
    print(prompt)
