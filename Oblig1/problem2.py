import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols


def main():
    df = pd.read_csv("blood.txt", " ")

    """ a)
    Describe the data using boxplots
    and numerical summeries
    """

    df.boxplot(by="age")
    plt.ylabel("Blood pressure")
    plt.xlabel("Age group no.")
    plt.show()

    df.describe()

    """ b)
    Use one-way ANOVA to answer the question above.
    Specify assumptions and the hypotheses you are testing.
    Write a summary of your findings.
    """

    mod = ols('Bloodpr ~ age', data=df).fit()
    aov_table = sm.stats.anova_lm(mod)
    print(aov_table)

    # null hyp is that age has no effect on blood pressure.
    # p<0.05, so we reject our null hypo, such that age does
    # in fact have an impact on the null hypothesis
    """ c)
    Formulate this problem using a regression model with
    age group as categorical predictor variable.
    Use treatment contrast and the youngest group
    as reference. Run the analysis, interpret the results and
    write a conclusion. Compare with b).
    """
    mod = ols("Bloodpr ~ age", data=df)
    res = mod.fit()
    print(res.summary())

    return 1


if __name__ == '__main__':
    main()
