import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


def main():
    """
    cirrhosis.txt file indicators:
        status  : Indicator for dead/censoring (1=dead; 0=censored)
        time    : Time in days from start of treatment to death/censoring
        treat   : Treadment (0=prednisone; 1=placebo)
        sex     : Gender (0=female; 1=male)
        asc     : Ascites at start of treatment (0=none; 1=slight; 2=marked)
        age     : Age in years at start of treatment 
        agegr   : Age group (1=<50; 2=50-65; 3=>65)
    """

    data = pd.read_csv("data/cirrhosis.txt", "\s+")

    """
    a)  Make Kaplan-Meier plots for the survival function for each level of the
        covariates treatment, sex, ascites, and group age (four plots in total).
        Discuss what the plots tell you.
    """

    obj = KaplanMeierFitter()

    ax1 = plt.subplot(221)
    grp0 = data[data["treat"] == 0]
    grp1 = data[data["treat"] == 1]
    obj.fit(grp0["time"], grp0["status"], label = "Prednisone").plot(ax=ax1)
    obj.fit(grp1["time"], grp1["status"], label = "Placebo").plot(ax=ax1)

    ax2 = plt.subplot(222)
    grp0 = data[data["sex"] == 0]
    grp1 = data[data["sex"] == 1]
    obj.fit(grp0["time"], grp0["status"], label = "Female").plot(ax=ax2)
    obj.fit(grp1["time"], grp1["status"], label = "Male").plot(ax=ax2)

    ax3 = plt.subplot(223)
    grp0 = data[data["asc"] == 0]
    grp1 = data[data["asc"] == 1]
    grp2 = data[data["asc"] == 2]
    obj.fit(grp0["time"], grp0["status"], label = "None").plot(ax=ax3)
    obj.fit(grp1["time"], grp1["status"], label = "Slight").plot(ax=ax3)
    obj.fit(grp2["time"], grp2["status"], label = "Marked").plot(ax=ax3)

    ax4 = plt.subplot(224)
    grp0 = data[data["agegr"] == 1]
    grp1 = data[data["agegr"] == 2]
    grp2 = data[data["agegr"] == 3]
    obj.fit(grp0["time"], grp0["status"], label = "<50").plot(ax=ax4)
    obj.fit(grp1["time"], grp1["status"], label = "50-65").plot(ax=ax4)
    obj.fit(grp2["time"], grp2["status"], label = ">65").plot(ax=ax4)

    plt.show()

    """
    b)  For each of the covariates, use the logrank test to investigate if the 
        covariate has a significant effect on survival.
    """

    lrt = logrank_test()

    """
    c)  Then do multiple Cox regression where the effects of all the covariates
        are studied simultaneously. Use age in years (not grouped). Summarize
        (and interpret) your findings. For this 'full' model with all covariates
        as main effects, find a 95% confidence interval for the hazard ratio for 
        men versus women when all other covariates are constant. Write a
        conclusion about the effect of prednisone in this trial.
    """


if __name__ == "__main__":
    main()