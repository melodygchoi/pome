from openai import OpenAI
from chromadb import Documents, EmbeddingFunction, Embeddings
from multiprocessing import Pool
from data_clean import *

import pandas as pd
import numpy as np
import logging
import pickle
import chromadb
import csv
import pickle

logger = logging.getLogger(__name__)
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

reclean = True

# Read the CSV file into a pandas DataFrame
def main():
    logger.debug('Starting...')

    with open(r"cleaned_data.obj", "rb") as f:
        data = pickle.load(f)

    if reclean == True or data == None:
        print("lame")
        try:
            with open('PoetryFoundationData.csv') as file:
                reader = csv.reader(file)
                logger.debug('Opened file...')

                df = pd.read_csv(file)
                data = clean(df)
                logger.debug('File has been successfully cleaned...')
            file.close()
        except Exception:
            raise Exception("Couldn't clean")

    df = pd.DataFrame(data).T
    print(df.shape)


    # collection = chroma_client.create_collection(name="poems", embedding_function=EmbedPoems)
    # collection.add(documents=poems.tolist())

if __name__ == '__main__':
    main()
    # freeze_support()
