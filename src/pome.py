import logging
import pandas as pd
import pickle
import sys
import time
import matplotlib.pyplot as plt

from data_clean import *
from embed_poems import *
from multiprocessing import Pool
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


logger = logging.getLogger(__name__)
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
reclean = False
pc = Pinecone(api_key="3468749b-2be3-4ccd-b507-cfe609ece890")


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

def cluster(embeddings, k):
    points = np.array(embeddings)
    kmeans = KMeans(n_clusters=k, 
                    init="k-means++",
                    random_state=0, 
                    n_init="auto"
                    ).fit(points)
    labels = kmeans.labels_

    tsne = TSNE(n_components=2, perplexity=15, random_state=0, init="random", learning_rate=200)
    vis_dims2 = tsne.fit_transform(points)

    x = [x for x, y in vis_dims2]
    y = [y for x, y in vis_dims2]

    for category, color in enumerate(["purple", 
                                      "green", 
                                      "red", 
                                      "blue", 
                                      "orange",
                                      "yellow",
                                      "turquoise",
                                      "lime",
                                      "saddlebrown",
                                      "hotpink",
                                    #   "darkblue",
                                    #   "darkgrey",
                                    #   "maroon",
                                    #   "skyblue",
                                    #   "darkolivegreen",
                                    #   "thistle",
                                    #   "magenta",
                                    #   "lightpink",
                                    #   "lightgreen",
                                    # "darkslategrey"
                                    ]):
        xs = np.array(x)[labels == category]
        ys = np.array(y)[labels == category]
        
        plt.scatter(xs, ys, color=color, alpha=0.3, marker=".")

        avg_x = xs.mean()
        avg_y = ys.mean()

        plt.scatter(avg_x, avg_y, marker="x", color=color, s=100)
    plt.title("Clusters identified from poems")

    plt.savefig('plot10.png', dpi=300, bbox_inches='tight')

    return kmeans, labels


# Data cleaner for input poem
def clean_input(df_row):
    title = repr(df_row.Title)
    poem = repr(df_row.Poem)

    try:
        # manual getting rid of quotations
        title = rid_quotes(title)
        poem = rid_quotes(poem)
        
        # manual strip
        title = strip(title)
        poem = strip(poem)
        df = pd.DataFrame([[title, poem, df_row.Poet, df_row.Tags]])
        return df
    
    except Exception as err:
        raise err


# Return the cluster that the poem would belong in.
# Uses k-means clustering.
def predict(df, kmeans, input_embedding):
    label = kmeans.predict(np.array(input_embedding).reshape(1, -1))
    cluster = []

    for row in df.iterrows():
        if row[1].Cluster == label:
            cluster.append(row[1])

    # cluster = df.loc[df['Cluster'] == label]
    print(f"RECOMMENDATIONS ({len(cluster)} total):")
    for poem in cluster:
        print(f'"{poem.Title}" by {poem.Poet}')


# Return the k nearest poems via Pinecone query.
# Uses cosine distance.
def knearest(df, x, input_embedding):
    try:
        index = pc.Index("embeddings")

        neighbors = index.query(
                namespace="ns1", 
                top_k=x, 
                vector=input_embedding, 
                include_values=True
                )
        
        print(f"TOP {x} CLOSEST POEMS (COSINE SIMILARITY):")
        for match in neighbors.matches:
            logger.debug(match.get("id"))
            score = match.get("score")
            poem = df.iloc[int(match.get("id")) - 1]
            print(f'- "{poem.Title}" by {poem.Poet} (Similarity: {round(score * 100, 2)}%)')

    except Exception as err:
        raise err

# Embedding function for one single df row.
def embed_one(df):
    openai_embed = CLIENT.embeddings.create(input=df.Poem, 
                                            model=EMBEDDING_MODEL, 
                                            dimensions=256,
                                            encoding_format="float")
    embedding = openai_embed.data[0].embedding
    custom_features = get_custom_features(df)
    for feature in custom_features:
        embedding.append(feature)
    return embedding


# Gets embeddings of poems.
def pome(rows, k, recommend, option, x):
    try:
        ### CLASSIFICATION ###
        start = time.time()
        logger.debug('Starting...')

        index = pc.Index("embeddings")

        # Get cleaned dataframe of poetry data.
        df = get_df().head(rows)
        
        embeddings = embed_poems(df)
        
        # Get embeddings from Pinecone index
        kmeans, labels = cluster(embeddings, k)
        df['Cluster'] = labels


        ### RECOMMENDATION ####
        if recommend: 
            test_poem = get_df().iloc[600]
            input_embedding = embed_one(test_poem)
            print("----------------------------------------------------------------------------------------------------------")
            if option == "recommend":
                print(f'Recommending poems based on input poem "{test_poem.Title}" ({test_poem.Poet})...')
                print("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *")
                predict(df, kmeans, input_embedding)
            elif option == "nearest":
                
                print(f'Finding {x} nearest neighbor poems based on input poem "{test_poem.Title}" ({test_poem.Poet})...')
                print("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *")

                knearest(df, x, input_embedding)
            print("----------------------------------------------------------------------------------------------------------")

        
        return embeddings

    except TypeError as err:
        logger.error(err)

