import pandas as pd
import pickle

def main():
    csv_read = True
    try:
        if not csv_read:
            with open('../data/PoetryFoundationData.csv') as file:
                csv = pd.read_csv(file)
            file.close()

            with open(r'obj/csv.obj', 'wb') as f:     
                pickle.dump(csv,f)
            f.close()
        else:
            with open(r'obj/csv.obj', 'rb') as f:     
                csv = pickle.load(f)
            f.close()

    except Exception:
        raise Exception("Couldn't clean")
    
    print(csv.head())
    print(csv.iloc[600].Title)

    csv.head(1).to_csv("../data/input_poem.csv", 
                        sep=',', 
                        columns=['Title', 'Poem', 'Poet', 'Tags'], 
                        index=False, 
                        encoding='utf-8')

if __name__ == '__main__':
    main()
