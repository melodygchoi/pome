import chromadb
import logging
import pandas as pd
import pickle

from data_clean import *
from embed_poems import *
from multiprocessing import Pool


logger = logging.getLogger(__name__)
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
reclean = True


# Returns a clean dataframe of the poetry data.
def get_df() -> pd.DataFrame:
    # If data needs to be re-cleaned, re-clean data
    if reclean:
        clean()

    # If cleaned data exists, set data = existing clean data
    with open(r"obj/cleaned_data.obj", "rb") as f:
        data = pickle.load(f)
    f.close()
        
    return data


# Gets embeddings of poems.
def main():
    logger.debug('Starting...')

    # client = chromadb.PersistentClient(path="../chroma")
    # collection = client.create_collection(name="poems")

    # Get cleaned dataframe of poetry data.
    df = get_df()

    print(df.head())

    # Get embeddings.
    # TODO
   
    # Add dataframe to chroma collection.
    # This will create and store embeddings using our custom function.
    # collection.add(documents=df)


if __name__ == '__main__':
    main()
    # freeze_support()
