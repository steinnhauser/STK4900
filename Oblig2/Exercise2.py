import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sklearn as skl
import sklearn.linear_model as lm


def main():
    """
    olympic.txt file indicators:
        Total1996       : Number of medals won by the nation in the previous game
        Log.population  : Logarithm of the nation's population size per 1000
        Log.athletes    : Logarithm of th enumber of athletes representing the nation
        GDP.per.cap     : The par capita Gross Domestic Product of the nation
    """

    data = pd.read_csv("data/olympic.txt", "\s+")

    print(data)


if __name__ == "__main__":
    main()