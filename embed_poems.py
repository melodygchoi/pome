from openai import OpenAI
from chromadb import EmbeddingFunction, Embeddings
from classifier.custom_features import *

EMBEDDING_MODEL = "text-embedding-3-small"


class EmbedPoems(EmbeddingFunction):
    def __call__(self, input: str) -> Embeddings:
        client = OpenAI()

        # create an embedding for a poem
        embedding = client.embeddings.create(input=input, model=EMBEDDING_MODEL, encoding_format=float)

        # append custom features to this embedding

        embedding.data[0].embedding.append()

        return embedding.data.embed
        
