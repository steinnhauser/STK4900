import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sklearn as skl
import sklearn.linear_model as lm


def main():
    """
    Crabs.txt file indicators:
        y       : Indicator for one or more satellites (0 = no, 1 = yes)
        width   : Width of carapace of female crab (in cm)
        weight  : Weight of female crab (in kg)
        color   : Color of female crab (1 = Medium light, 2 = Medium, 3 = Medium dark, 4 = Dark)
        spine   : Conditions of spine (1 = Both good, 2 = One worn or broke, 3 = Both broken)
    """
    data = pd.read_csv("data/crabs.txt", "\s+")

    """
    a)  Choose a suitable regression model for studying how the probability
        of presence of satellites depends on the explanatory variable width.
        Give the reasons of your choice of regression model.
    """

    data.corr()["y"].plot.bar()
    plt.ylabel("Correlation")
    plt.title("Correlation of features in relation to the output y.")
    plt.grid()
    plt.show()

    plt.scatter(data["width"], data["y"])
    plt.title("Scatterplot of the width versus the outcome y.")
    plt.xlabel("Width of carapace of female crab [cm]")
    plt.ylabel("Indicator for one or more satellites (0 = no, 1 = yes)")
    plt.show()

    X = data['width'].values.reshape(-1,1)
    y = data["y"].values

    classifier = lm.LogisticRegression().fit(X, y)
    print(f"Classifier coefficient: {classifier.coef_[0][0]:.2f}")
    print(f"Classifier intercept: {classifier.intercept_[0]:.2f}")


    """
    b)  In particular find the odds ratio of presences of satellites between
        crabs that differ one cm in width, and explain what this odds ratio
        means. Can the odds ratio be considered as an approximation to a
        relative risk in this situation? Also find a confidence interval for 
        the odds ratio and determine whether width influences presence of 
        satellites significantly.
    """
    print("-----------------------------------")
    print(f"Odds ratio of was found to be: {np.exp(classifier.coef_[0][0]):.2f}")


    """
    c)  Then consider the other explanatory variables weight, color and spine,
        one at a time. Discuss whether these covariates should be included as
        categorical or numerical. Determine which variables has a significant
        influence on the presence of satellites.
    """
    print("-----------------------------------")
    print("Weight logistic regression results:")
    X = data['weight'].values.reshape(-1,1)
    y = data["y"].values

    classifier = lm.LogisticRegression().fit(X, y)
    print(f"\tClassifier coefficient: {classifier.coef_[0][0]:.2f}")
    print(f"\tClassifier intercept: {classifier.intercept_[0]:.2f}")

    print("-----------------------------------")
    print("Color logistic regression results:")
    X = data['color'].values.reshape(-1,1)
    y = data["y"].values

    classifier = lm.LogisticRegression().fit(X, y)
    print(f"\tClassifier coefficient: {classifier.coef_[0][0]:.2f}")
    print(f"\tClassifier intercept: {classifier.intercept_[0]:.2f}")

    print("-----------------------------------")
    print("Spine logistic regression results:")
    X = data['spine'].values.reshape(-1,1)
    y = data["y"].values

    classifier = lm.LogisticRegression().fit(X, y)
    print(f"\tClassifier coefficient: {classifier.coef_[0][0]:.2f}")
    print(f"\tClassifier intercept: {classifier.intercept_[0]:.2f}")


    """
    d)  Next use all variables in the regression (as main effects), and describe
        your findings. Try to simplify the model only using the significant
        covariates. In particular discuss the covariates weight and width.
    """


    """
    e)  Finally investigate whether there are interactions between covariates.
    """
    data.corr()["weight"].plot.bar()
    plt.ylabel("Correlation")
    plt.title("Correlation of features in relation to variable weight.")
    plt.grid()
    plt.show()

    data.corr()["width"].plot.bar()
    plt.ylabel("Correlation")
    plt.title("Correlation of features in relation to variable width.")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()