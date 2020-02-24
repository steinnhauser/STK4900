# import rpy2 as rp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn import linear_model
from sklearn.metrics import r2_score


def main():

    data = pd.read_csv("no2.txt", "\t")

    """ a)
    Make a scatterplot with log.cars on
    the x-axis and log.no2 on the y-axis
    """

    data.describe()

    plt.plot(data["log.cars"], data["log.no2"], "r.")
    plt.grid()
    plt.xlabel("Logarithm of the number of cars per hour.")
    plt.ylabel("Logarithm of the no2 concentration.")
    plt.show()

    """ b)
    Fit a simple linear model where the log [no2] is
    explained by the amount of traffic, measured by log.cars
    """

    # Simple linear model of degree 1.
    coefs = np.polyfit(data["log.cars"], data["log.no2"], 1)

    # Generate the prediction line.
    xrange = np.linspace(
        data["log.cars"].min(),
        data["log.cars"].max(),
        data["log.cars"].index.size
    )
    pred = coefs[1]+coefs[0]*xrange

    # Calculate the R2 score.
    target = data["log.no2"].values.copy()
    target = target[np.argsort(target)]
    r2_linfit = r2_score(target, pred)
    print(f"R2 Score of the linear fit:\t{r2_linfit:.4f}.")

    # Generate a new plot which now includes the linear fit.
    plt.plot(data["log.cars"], data["log.no2"], "r.")
    plt.plot(xrange,pred)
    plt.grid()
    plt.xlabel("Logarithm of the number of cars per hour.")
    plt.ylabel("Logarithm of the no2 concentration.")
    plt.show()

    """ c)
    Check various residual plots to judge if
    the model assumptions for the model are reasonable.
    """

    sbn.residplot(
        data["log.cars"],
        data["log.no2"],
        lowess=True,
        color="r",
        order=1
    )
    plt.grid()
    plt.show()

    sbn.lmplot(x="log.cars", y="log.no2", data=data)
    plt.grid()
    plt.show()

    """ d)
    Use multiple regression now.
    Check if some variables should be transformed.
    """

    # Multiple Regression of several linear predictors
    multiple_regression(data)

    # Multidimensional Regression checks of the number of cars variable
    c2, c3, c4, c5 = multidimensional_regressions(data)

    """ e)
    For the model chosen in d), write an interpretation
    of the model coefficients and check if the model
    assumptions seem reasonable through various plots.
    """

    # Checking if the model assumptions seem reasonable
    data = pd.read_csv("no2.txt", "\t")
    data["hour.of.day"] = data["hour.of.day"].apply(lambda x: 1./np.exp(x))

    # # Multiple Regression of several linear predictors, having now transformed some predictors
    predictors = data.loc[:, ["log.cars", "hour.of.day", "wind.speed"]]
    pd_headers = predictors.columns.values.tolist()
    target = data["log.no2"]

    model = linear_model.LinearRegression()
    model.fit(predictors, target)

    coefs = dict(zip(pd_headers, model.coef_))

    print(f"Intercept of the multiple regression model:\t {model.intercept_}")
    print(f"Coefficients of the multiple regression model:{coefs}")

    return 1

def multiple_regression(data):
    # Transformation using the logarithmic function
    # data["temp"] = data["temp"].apply(lambda x: np.log(x))  # Negative values makes this invalid
    # data["wind.speed"] = data["wind.speed"].apply(lambda x: np.log(x)) # NaN values encountered here as well
    # data["hour.of.day"] = data["hour.of.day"].apply(lambda x: np.log(x)) # NaN values once again

    # Transformation using the exponential function;
    data["temp"] = data["temp"].apply(lambda x: np.exp(x))
    data["wind.speed"] = data["wind.speed"].apply(lambda x: np.exp(x))
    data["hour.of.day"] = data["hour.of.day"].apply(lambda x: np.exp(x))

    # # Multiple Regression of several linear predictors, having now transformed some predictors
    predictors = data.loc[:, data.columns!="log.no2"]
    pd_headers = data.columns.values.tolist()[1:]
    target = data["log.no2"]

    model = linear_model.LinearRegression()
    model.fit(predictors, target)

    coefs = dict(zip(pd_headers, model.coef_))

    print(f"Intercept of the multiple regression model:\t {model.intercept_}")
    print(f"Coefficients of the multiple regression model: \n{coefs}")


def multidimensional_regressions(data):
    """ Function to conduct several degrees of regression on the data and
    illustrate them nicely in the same matplotlib.pyplot image. """

    target = data["log.no2"].values.copy()
    target = target[np.argsort(target)]

    xrange = np.linspace(
        data["log.cars"].min(),
        data["log.cars"].max(),
        data["log.cars"].index.size
    )

    # Conduct polynomial fits of degrees higher than 1
    coefs2 = np.polyfit(data["log.cars"], data["log.no2"], 2)
    coefs3 = np.polyfit(data["log.cars"], data["log.no2"], 3)
    coefs4 = np.polyfit(data["log.cars"], data["log.no2"], 4)
    coefs5 = np.polyfit(data["log.cars"], data["log.no2"], 5)

    pred2 = coefs2[2] + \
            coefs2[1]*xrange + \
            coefs2[0]*xrange*xrange

    pred3 = coefs3[3] + \
            coefs3[2]*xrange + \
            coefs3[1]*xrange*xrange + \
            coefs3[0]*xrange*xrange*xrange

    pred4 = coefs4[4] + \
            coefs4[3]*xrange + \
            coefs4[2]*xrange*xrange + \
            coefs4[1]*xrange*xrange*xrange + \
            coefs4[0]*xrange*xrange*xrange*xrange

    pred5 = coefs5[5] + \
            coefs5[4]*xrange + \
            coefs5[3]*xrange*xrange + \
            coefs5[2]*xrange*xrange*xrange + \
            coefs5[1]*xrange*xrange*xrange*xrange + \
            coefs5[0]*xrange*xrange*xrange*xrange*xrange

    # Calculate the R2 scores of the higher order polynomial fits.
    r2_linfit2 = r2_score(target, pred2)
    r2_linfit3 = r2_score(target, pred3)
    r2_linfit4 = r2_score(target, pred4)
    r2_linfit5 = r2_score(target, pred5)

    print(f"R2 Score of the 2nd degree fit:\t{r2_linfit2:.4f}.")
    print(f"R2 Score of the 3rd degree fit:\t{r2_linfit3:.4f}.")
    print(f"R2 Score of the 4th degree fit:\t{r2_linfit4:.4f}.")
    print(f"R2 Score of the 5th degree fit:\t{r2_linfit5:.4f}.")

    # Generate a new plot which now includes the 2nd degree fit.
    plt.subplot(221)
    plt.title("2nd degree regression plot")
    plt.plot(data["log.cars"], data["log.no2"], "r.")
    plt.plot(xrange,pred2)
    plt.grid()
    plt.xlabel("Logarithm of the number of cars per hour.")
    plt.ylabel("Logarithm of the no2 concentration.")

    plt.subplot(222)
    plt.title("3rd degree regression plot")
    plt.plot(data["log.cars"], data["log.no2"], "r.")
    plt.plot(xrange,pred3)
    plt.grid()
    plt.xlabel("Logarithm of the number of cars per hour.")
    plt.ylabel("Logarithm of the no2 concentration.")

    plt.subplot(223)
    plt.title("4th degree regression plot")
    plt.plot(data["log.cars"], data["log.no2"], "r.")
    plt.plot(xrange,pred4)
    plt.grid()
    plt.xlabel("Logarithm of the number of cars per hour.")
    plt.ylabel("Logarithm of the no2 concentration.")

    plt.subplot(224)
    plt.title("5th degree regression plot")
    plt.plot(data["log.cars"], data["log.no2"], "r.")
    plt.plot(xrange,pred5)
    plt.grid()
    plt.xlabel("Logarithm of the number of cars per hour.")
    plt.ylabel("Logarithm of the no2 concentration.")

    plt.tight_layout()
    plt.show()

    return coefs2, coefs3, coefs4, coefs5

if __name__ == '__main__':
    main()
