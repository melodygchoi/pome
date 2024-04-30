from pome import *
import sys

def main(rows, k, option, x):
    if option == "cluster":
        embeddings = pome(rows, k, recommend=False, option=option, x=None)
    elif option == "recommend" or option == "nearest":
        embeddings = pome(rows, k, recommend=True, option=option, x=x)
    else:
        print("Invalid option entered. \nPlease enter an option out of the three: \n - 'cluster', 'recommend' or 'nearest")

if __name__ == '__main__':
    rows = int(sys.argv[1])
    k = int(sys.argv[2])
    option = sys.argv[3]
    x = int(sys.argv[4]) 

    main(rows, k, option, x)

    # freeze_support()