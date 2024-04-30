# 🍎	Pome
**_Classifying and recommending poems using custom multi-feature embeddings_**

Melody Choi // 
Advised by Brian W. Kernighan //
Spring 2024 Independent Work


## 🖋️	Overview
Pome allows for the classification and recommendation of poems using k-means clustering and nearest neighbor search.
It is developed in Python with the OpenAI model `text-embedding-3-small`, the sentiment analysis model DistilBERT, the NLP model spaCy, vector database Pinecone, pandas, scikitlearn, and matplotlib.

Given a dataset as a csv file, Pome will:
- create embeddings of each poem
- assign each embedding to a cluster center using the k-means algorithm

Given an input poem to form recommendations off of, Pome will:
- predict which cluster the poem belongs to, and
- either:
  1. return the entire cluster as a set of recommendations OR
  2. return the x nearest neighbors of the poem.

Clusters can be visualized as well:
![](/images/plot10.png)


## 💡	Installation

To install, run the following commands:
  ```
  $ git clone https://github.com/melodygchoi/pome
  $ cd pome
  $ pip install -r requirements.txt
  ```


## 💻	Using Pome
To run Pome, call `python run.py [rows] [k] [option] [x]`.

- `int rows`: Number of rows of the dataframe the user wants to create embeddings of, i.e. the number of poems the user would like embedded and clustered.
- `int k`: Number of cluster centers desired for k-means clustering.
- `str option`: Either of “cluster”, “recommend”, or “nearest”, depending on which action the user would like for Pome to perform.
  1. cluster embeds `rows` number of poems and assigns cluster center labels to each poem.
  2. recommend embeds a poem to form recommendations off of, assigns a cluster center label to the poem, and returns the rest of the cluster as a set of recommendations.
  3. nearest embeds a poem to form recommendations off of and uses Pinecone’s vector search to find `x` number of closest embeddings using cosine similarity.
- `int x`: Number of recommended poems desired. This value only matters when option is set to nearest.

