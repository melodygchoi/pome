"""
Returns the embedding data of a poem, given a pandas.DataFrame row.

"""

from openai import OpenAI
from chromadb import EmbeddingFunction, Embeddings, Documents
from custom_features import *

EMBEDDING_MODEL = "text-embedding-3-small"

class EmbedPoems(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        client = OpenAI()

        # create an embedding for a poem
        embedding = client.embeddings.create(input=input, model=EMBEDDING_MODEL, encoding_format=float)

        # append custom features to this embedding

        embedding.data[0].embedding.append(IJFGJDKJA)

        return embedding.data.embed
        