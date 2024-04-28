"""
Returns the embedding data of a poem, given a pandas.DataFrame row.

"""
import pandas as pd
import openai
import numpy as np

from openai import OpenAI
from openai.types import Embedding
from pinecone import Pinecone, ServerlessSpec
from chromadb import Embeddings
from custom_features import *
from multiprocessing import Pool

logger = logging.getLogger(__name__)
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

CLIENT = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-small"
pc = Pinecone(api_key="3468749b-2be3-4ccd-b507-cfe609ece890")
index_name = "openai"

if index_name not in pc.list_indexes().names():
    pc.create_index(
    name="openai",
    dimension=256, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
        )
    )

index = pc.Index(index_name)


# Returns a list of floats (an OpenAI embedding)
def get_openai_embeddings(tuple):
    idx = str(tuple[0])
    row = tuple[1]

    logger.debug("Querying PC index.")
    query = index.query(
        namespace="ns1", 
        top_k=1, 
        id=idx, 
        include_values=True
        )
    
    if not query.matches:
        logger.debug("Match not found; creating new embedding.")
        embedding = CLIENT.embeddings.create(input=row.Poem, 
                                            model=EMBEDDING_MODEL, 
                                            dimensions=256,
                                            encoding_format="float")
        # This may take a few minutes
        logger.debug("Upserting embedding into index.")
        index.upsert(
            vectors=[
                {"id": idx, "values": embedding.data[0].embedding}
            ],
            namespace="ns1"
        )
        return embedding.data[0].embedding
   
    logger.debug("Query match found. Returning stored embedding.")
    logger.debug(query.matches)
    return query.matches[0].get('values')


# Creates a single embedding for a single poem.
# Takes in an entire df row as input.
def embed(tuple) -> list[float]:
    logger.debug("Getting embeddings.")
    row = tuple[1]
    embedding = get_openai_embeddings(tuple)
    logger.debug("Successfully got openai embeddings.")
    # get custom features for this poem
    logger.debug("Getting custom features.")
    custom_features = get_custom_features(row)

    # # append custom features to this embedding
    logger.debug("Appending custom features.")
    for feature in custom_features:
        embedding.append(feature)

    logger.debug("Successfully created custom embedding! Returning...")
    return np.transpose(np.array(embedding)).tolist()

# Asynchronously embeds all poems.
# Stores all embeddings in a Pinecone index.
def embed_poems(df: pd.DataFrame):
    try:
        logger.debug("Staring multiprocessing.")
        em_idx = pc.Index("embeddings")
        embeddings = []
        logger.debug("yuh")

        for tuple in df.iterrows():
            logger.debug("mmmmmmmmm")
            embedding = em_idx.query(
                namespace="ns1", 
                top_k=1, 
                id=str(tuple[0]), 
                include_values=True
                )
            logger.debug("okie")
            if not embedding.matches:
                logger.debug("NAUrrrrr")
                embedding = embed(tuple)
                # This may take a few minutes
                logger.debug("Upserting embedding into index.")
                em_idx.upsert(
                    vectors=[
                        {"id": str(tuple[0]), 
                        "values": embedding}
                    ],
                    namespace="ns1"
                )
                embeddings.append(embedding)
            else:
                embeddings.append(embedding.matches[0].get('values'))

        logger.debug("Successfully got custom embeddings with multiprocessing.")
        logger.debug("Returning...")
    except Exception as err:
        raise err
    return embeddings