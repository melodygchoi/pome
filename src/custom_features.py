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
nlp = spacy.load("en_core_web_trf")

# A map with all of the UPOS that exist.
UPOS_MAP = {
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


# Uses spaCy to transform a sentence to UPOS (universal part of speech).
def to_pos(poem):
    pos = []
    doc = nlp(poem)

    for token in doc:
        pos.append(token.pos_)
    
    return pos


# 

# Checks if a poem is single or double spaced.



# Uses spaCy to count the number of named entities in a poem.
def named_entities(string):
    doc = nlp(string)

    return float(len(doc.ents) / len(doc))


# Counts the length of a title, given an integer limit to determine
# what counts as a "long" title.
def is_title_long(title, lim):
    split = title.split()
    if len(split) > lim:
        return 1.0
    return 0.0


# Uses spaCy to determine frequency of certain UPOS.
def pos_freq(poem):
    freq = []
    upos = UPOS_MAP.copy()

    # get transformation of poem to UPOS
    pos = to_pos(poem)

    # count frequency of certain UPOS
    for p in pos:
        if p in upos:
            upos[p] = upos.get(p, 0) + 1
    
    # each value in list represents what percentage of the poem consists
    # of that specific upos.
    for key in upos:
        logger.debug(f"Key: {key}")
        freq.append(upos[key] / len(pos))
    
    logger.debug(f"pos_freq len: {len(freq)}")
    return freq
    

# Uses spaCy to determine the frequency of newlines "\n" coming after certain UPOS.
def enjambment(poem):
    # this list will be appended to the vector. 
    enjamb = []
    upos = UPOS_MAP.copy()

    # get transformation of poem to UPOS
    pos = to_pos(poem)

    # count all the enjmabments that exist after certain UPOS
    count = 0
    for i in range(len(pos) - 1):
        if pos[i + 1] == 'SPACE':
            if pos[i] in upos:
                upos[pos[i]] = upos.get(pos[i]) + 1
                count += 1

    # each element in enjamb represents a UPOS.
    # each value in this list represents the percentage throughout the poem 
    # that a certain UPOS appears before a line break. 
    for key in upos:
        if count > 0:
            enjamb.append(upos[key] / count)
        else:
            enjamb.append(0.0)
    logger.debug(f"enjamb len: {len(enjamb)}")
    return enjamb


# Uses DistilBERT to perform sentiment analysis on a string.
def sentiment_analysis(string):
    logger.debug("Starting sentiment analysis.")
    sa = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    parts = [string[i:i+1000] for i in range(0, len(string), 1000)]
    logger.debug(parts)
    scores = 0
    for s in parts:
        score = sa(s)
        if score[0]['label'] == 'POSITIVE':
            score_val = score[0]['score']
        else:
            score_val = 0 - score[0]['score']
        scores += score_val
    scores /= len(parts)
    return scores


# Returns a list of custom features.
def get_custom_features(row):
    title = row.Title
    poem = row.Poem
    poet = row.Poet
    tags = row.Tags

    # List for features (float values) to be stored in
    features = []

    # Call all the functions in order
    features.append(sentiment_analysis(title)) # 1 value
    features.append(sentiment_analysis(poem)) #1

    for i in enjambment(poem): #17
        features.append(i)
    for i in pos_freq(poem): #17
        features.append(i)
    features.append(is_title_long(title, 10)) #1
    features.append(named_entities(title)) #1
    features.append(named_entities(poem)) #1

    logger.debug(len(features*8))
    return features*8


# Testing function
# def main():
#     title = "Objects Used to Prop Open a Window"
#     poem = "Dog bone, stapler,\n\ncribbage board, garlic press\n\n     because this window is loose—lacks\n\nsuction, lacks grip.\n\nBungee cord, bootstrap,\n\ndog leash, leather belt\n\n     because this window had sash cords.\n\nThey frayed. They broke.\n\nFeather duster, thatch of straw, empty\n\nbottle of Elmer's glue\n\n     because this window is loud—its hinges clack\n\nopen, clack shut.\n\nStuffed bear, baby blanket,\n\nsingle crib newel\n\n     because this window is split. It's dividing\n\nin two.\n\nVelvet moss, sagebrush,\n\nwillow branch, robin's wing\n\n     because this window, it's pane-less. It's only\n\na frame of air."
#     poet = "Michelle Menting"
#     tags = "Living,Time & Brevity,Relationships,Family & Ancestors,Nature,Landscapes & Pastorals,Seas, Rivers, & Streams,Social Commentaries,History & Politics"
#     sample = [title, poem, poet, tags]

#     logger.debug("Hi!")

#     # row = pd.DataFrame(sample, columns=['Title', 'Poem', 'Poet', 'Tags'])

#     print(pos_freq(poem))

#     # features = get_custom_features(row)
#     # print(features)

# if __name__ == '__main__':
#     main()