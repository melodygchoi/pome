from multiprocessing import Pool

import pandas as pd
import numpy as np
import logging
import pickle
import pickle

LOGGER = logging.getLogger(__name__)
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)


# Gets rid of beginning, end quotes/apostrophes.
def rid_quotes(string) -> str:  
    try:
        # manual getting rid of quotations
        if string[0] == "'" or string[0] == '"':
            string = string[1:]
        if string[len(string) - 1] == "'" or string[len(string) - 1] == '"':
            string = string[:len(string) - 1]
        return string
    except Exception as err:
        raise Exception(f"Unexpected {err=} at while removing quotes")


# Strips \n from beginning, end of string.
def strip(string) -> str:
    try:
        # manual lstrip
        i = 0
        while len(string) > 0:
            if string[0] == '\\' and string[1] == 'n':
                string = string[2:]
            else:
                break
        
        string = str.strip(string)

        # manual rstrip
        i = len(string) - 1
        while i > 0:
            if string[i] == 'n' and string[i - 1] == '\\':
                string = string[:i - 1]
                i = len(string) - 1
            else:
                break
        return string
    
    except Exception as err:
        raise Exception(f"Unexpected {err=} while stripping")


# Clean beginnings and ends of string.
def clean_attr(tuple):
    row = tuple[1]

    title = repr(row.Title)
    poem = repr(row.Poem)

    try:
        # manual getting rid of quotations
        title = rid_quotes(title)
        poem = rid_quotes(poem)
        
        # manual strip
        title = strip(title)
        poem = strip(poem)

        return [title, poem, row.Poet, row.Tags]
    
    except Exception as err:
        raise err


# Asynchronously clean up data in a df.
# Uses multiprocessing with chunksize = 1000.
def clean_async(df):
    with Pool() as pool:
        result = pool.map_async(clean_attr, df.iterrows(), chunksize=1000)
        new = result.get()
        pool.close()
        pool.join()
    try:
        data = np.array(new).flatten()
        print(data)
        with open(r'cleaned_data.obj', 'wb') as f:     
            pickle.dump(data,f)
        f.close()
    except Exception as err:
        raise Exception("Pickle failed")


# Reads and cleans up the string data stored in the given CSV.
def clean():
    csv_read = True
    
    try:
        if not csv_read:
            with open('PoetryFoundationData.csv') as file:
                LOGGER.debug('Reading file...')
                csv = pd.read_csv(file)
                csv_read = True

            with open(r'csv.obj', 'wb') as f:     
                pickle.dump(csv,f)
            f.close()
        else:
            with open(r'csv.obj', 'rb') as f:     
                csv = pickle.load(f)
            f.close()

        data = clean(csv)
        LOGGER.debug('File has been successfully cleaned...')
        file.close()

    except Exception:
        raise Exception("Couldn't clean")