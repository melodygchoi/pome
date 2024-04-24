from openai import OpenAI
from chromadb import Documents, EmbeddingFunction, Embeddings
from multiprocessing import Pool

# from classifier.embed_poems import EmbedPoems

import pandas as pd
import numpy as np
import logging
import pickle
import chromadb
import csv
import pickle
# import chromadb.utils.embedding_functions as embedding_functions

# from utils.embeddings_utils import (
#     get_embedding,
#     distances_from_embeddings,
#     tsne_components_from_embeddings,
#     chart_from_components,
#     indices_of_nearest_neighbors_from_distances,
# )

# with open("PoetryFoundationData.csv") as csvfile:
#     reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
#     for row in reader: # each row is a list
#         results.append(row)

# openai_ef = embedding_functions.OpenAIEmbeddingFunction(
#                 api_key="YOUR_API_KEY",
#                 model_name=EMBEDDING_MODEL
#             )


# chroma_client = chromadb.Client()
logger = logging.getLogger(__name__)
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

def rid_quotes(string) -> str:
    try:
        # manual getting rid of quotations
        if string[0] == "'" or string[0] == '"':
            string = string[1:]
        if string[len(string) - 1] == "'" or string[len(string) - 1] == '"':
            string = string[:len(string) - 1]
        return string
    except Exception as err:
        raise Exception(f"Unexpected {err=} at while removing quotes")

def strip(string) -> str:
    try:
        # print(string)
        # manual lstrip
        i = 0
        while len(string) > 0:
            if string[0] == '\\' and string[1] == 'n':
                string = string[2:]
            else:
                break
        string = str.strip(string)
        # manual rstrip
        i = len(string) - 1
        while i > 0:
            if string[i] == 'n' and string[i - 1] == '\\':
                string = string[:i - 1]
                i = len(string) - 1
            else:
                break
        return string
    
    except Exception as err:
        raise Exception(f"Unexpected {err=} while stripping")

# part of clean: clean beginnings and ends of string
def clean_attr(tuple):
    row = tuple[1]

    title = repr(row.Title)
    poem = repr(row.Poem)

    try:
        # manual getting rid of quotations
        title = rid_quotes(title)
        poem = rid_quotes(poem)
        
        # manual strip
        title = strip(title)
        poem = strip(poem)

        return [title, poem, row.Poet, row.Tags]
    
    except Exception as err:
        raise err

# Clean up data in df; data is a df column
def clean(df):
    # spawn 10 threads to run this concurrently
    with Pool() as pool:
        new = []
        result = pool.map_async(clean_attr, df.iterrows(), chunksize=2000)
        new.append(result.get())
    return new

# Read the CSV file into a pandas DataFrame
def main():
    logger.debug('Starting...')

    with open('PoetryFoundationData.csv') as file:
        reader = csv.reader(file)
        logger.debug('Opened file...')

        df = pd.read_csv(file)
        data = np.array(clean(df)).flatten()
        logger.debug('File has been successfully cleaned...')

        print(data)
    file.close()

    # collection = chroma_client.create_collection(name="poems", embedding_function=EmbedPoems)
    # collection.add(documents=poems.tolist())

if __name__ == '__main__':
    main()
    # freeze_support()
