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
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(points)
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
                                      "hotpink"]):
        xs = np.array(x)[labels == category]
        ys = np.array(y)[labels == category]
        
        plt.scatter(xs, ys, color=color, alpha=0.3, marker=".")

        avg_x = xs.mean()
        avg_y = ys.mean()

        plt.scatter(avg_x, avg_y, marker="x", color=color, s=100)
    plt.title("Clusters identified from poems")

    plt.savefig('plot.png', dpi=300, bbox_inches='tight')

    return labels


# Gets embeddings of poems.
def main(rows):
    try:
        start = time.time()
        logger.debug('Starting...')

        index = pc.Index("embeddings")

        # Get cleaned dataframe of poetry data.
        df = get_df().head(rows)
        
        embeddings = embed_poems(df)
        
        # Get embeddings from Pinecone index
        labels = cluster(embeddings, 10)
        df['Cluster'] = labels

        df.to_csv("clusters.csv", sep=',', index=False, encoding='utf-8')

        end = time.time()
        print("It took", end-start, "seconds to get embeddings and cluster them!")

    except TypeError as err:
        logger.error(err)


if __name__ == '__main__':
    rows = int(sys.argv[1])
    main(rows)
    # freeze_support()
