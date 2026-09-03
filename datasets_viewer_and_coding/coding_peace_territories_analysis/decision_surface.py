import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

df_home = pd.read_csv(
    "df_vector_hogar_160726_publica.csv",
    #sep= ";",
    #dtype=dtypes_home,
    dayfirst=True,
    encoding="latin1",
    low_memory=False
)

##################################################################################################################
#Selecting just the indexes
classification_df = pd.DataFrame({

    "criticality":
        df_home["porcentaje_ponderado_Household_criticality_index"],

    "intersectionality":
        df_home["porcentaje_Intersectionality_index"]

})
#########################################################################################################
def vulnerability_class(x):

    if x <= 0.317:
        return 0     # LOW

    elif x <= 0.500:
        return 1     # MEDIUM

    elif x <= 0.683:
        return 2     # HIGH

    else:
        return 3     # CRITICAL

classification_df["category"] = (
    classification_df["criticality"]
    .apply(vulnerability_class)
)

#####################################################################################
#Features

X = classification_df[
    [
        "criticality",
        "intersectionality"
    ]
].values

y = classification_df["category"].values

#################################################################################
#standar scaling


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

#############################################################################
from sklearn.neighbors import KNeighborsClassifier

knc = KNeighborsClassifier(
    n_neighbors=7
)

knc.fit(
    X_scaled,
    y
)

#######################################################################################



def plot_decision_surface(
    classifier,
    X,
    labels
):

    colors = [
        "#2ca25f",   # LOW
        "#99d8c9",   # MEDIUM
        "#fdae6b",   # HIGH
        "#de2d26"    # CRITICAL
    ]

    cmap = ListedColormap(colors)

    x_min, x_max = (
        X[:,0].min() - 0.5,
        X[:,0].max() + 0.5
    )

    y_min, y_max = (
        X[:,1].min() - 0.5,
        X[:,1].max() + 0.5
    )

    xx, yy = np.meshgrid(

        np.linspace(x_min, x_max, 300),

        np.linspace(y_min, y_max, 300)

    )

    grid = np.c_[

        xx.ravel(),

        yy.ravel()

    ]

    Z = classifier.predict(grid)

    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10,8))

    plt.contourf(
        xx,
        yy,
        Z,
        alpha=0.30,
        cmap=cmap
    )

    scatter = plt.scatter(
        X[:,0],
        X[:,1],
        c=labels,
        cmap=cmap,
        edgecolor="black",
        s=30
    )

    plt.xlabel(
        "Household Criticality Index"
    )

    plt.ylabel(
        "Intersectionality Index"
    )

    plt.title(
        "K-Nearest Neighbors Classification of Household Vulnerability"
    )

    cbar = plt.colorbar(scatter)

    cbar.set_ticks([0,1,2,3])

    cbar.set_ticklabels([
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL"
    ])

    plt.show()

################################################################################################
#ploting the figure

    
plot_decision_surface(
    knc,
    X_scaled,
    y
)