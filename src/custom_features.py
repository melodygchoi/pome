"""
Returns a List of floats representing measured values of custom features.
This List eventually gets appended to OpenAI's embedding of the poem.

"""
import pandas as pd
import logging
import spacy

from typing import List
from transformers import pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

# Uses spaCy to transform a sentence to UPOS (universal part of speech).
def to_pos(poem):
    pos = []
    nlp = spacy.load("en_core_web_trf")
    doc = nlp(poem)

    for token in doc:
        pos.append(token.pos_)
    
    return pos


# Uses spaCy to determine the frequency of newlines "\n" coming after certain UPOS.
def enjambment(poem):
    # this list will be appended to the vector. 
    enjamb = []

    # A map with all of the UPOS that exist.
    upos = {
        "ADJ": 0,
        "ADP": 0,
        "ADV": 0,
        "AUX": 0,
        "CCONJ": 0,
        "DET": 0,
        "INTJ": 0,
        "NOUN": 0,
        "NUM": 0,
        "PART": 0,
        "PRON": 0,
        "PROPN": 0,
        "PUNCT": 0,
        "SCONJ": 0,
        "SYM": 0,
        "VERB": 0,
        "X": 0,
    }

    # get transformation of poem to UPOS
    pos = to_pos(poem)

    # count all the enjmabments that exist after certain UPOS
    count = 0
    for i in range(len(pos) - 1):
        if pos[i + 1] == 'SPACE':
            upos[pos[i]] = upos.get(pos[i]) + 1
            count += 1

    # each element in enjamb represents a UPOS.
    # each value in this list represents the percentage throughout the poem 
    # that a certain UPOS appears before a line break. 
    for key in upos:
        enjamb.append(upos[key] / count)
    
    return enjamb


# Uses DistilBERT to perform sentiment analysis on a string.
def sentiment_analysis(string):
    sa = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")
    score = sa(string)

    if score[0]['label'] == 'POSITIVE':
        score_val = score[0]['score']
    else:
        score_val = 0 - score[0]['score']
    return score_val


# Returns a list of custom features.
def get_custom_features(row) -> List[float]:
    title = row.Title
    poem = row.Poem
    poet = row.Poet
    tags = row.Tags

    # List for features (float values) to be stored in
    features = List[float]

    # Call all the functions in order
    features.append(sentiment_analysis(title))
    features.append(sentiment_analysis(poem))
    features.append(enjambment(poem))


    return features


# Testing function
def main():
    title = "Objects Used to Prop Open a Window"
    poem = "Dog bone, stapler,\n\ncribbage board, garlic press\n\n     because this window is loose—lacks\n\nsuction, lacks grip.\n\nBungee cord, bootstrap,\n\ndog leash, leather belt\n\n     because this window had sash cords.\n\nThey frayed. They broke.\n\nFeather duster, thatch of straw, empty\n\nbottle of Elmer's glue\n\n     because this window is loud—its hinges clack\n\nopen, clack shut.\n\nStuffed bear, baby blanket,\n\nsingle crib newel\n\n     because this window is split. It's dividing\n\nin two.\n\nVelvet moss, sagebrush,\n\nwillow branch, robin's wing\n\n     because this window, it's pane-less. It's only\n\na frame of air."
    poet = "Michelle Menting"
    tags = "Living,Time & Brevity,Relationships,Family & Ancestors,Nature,Landscapes & Pastorals,Seas, Rivers, & Streams,Social Commentaries,History & Politics"
    sample = [title, poem, poet, tags]

    logger.debug("Hi!")

    # row = pd.DataFrame(sample, columns=['Title', 'Poem', 'Poet', 'Tags'])

    print(enjambment(poem))

    # features = get_custom_features(row)
    # print(features)

if __name__ == '__main__':
    main()