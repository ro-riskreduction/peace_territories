import csv
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##############################################################################
# Load data
##############################################################################

df_home = pd.read_csv(
    "df_vector_hogar_160726_publica.csv",
    dayfirst=True,
    encoding="latin1",
    low_memory=False
)

##############################################################################
# Select indices
##############################################################################

classification_df = pd.DataFrame({

    "criticality_cat":
        df_home["categoria_Household_criticality_index"],

    "intersectionality_cat":
        df_home["categorizacion_Intersectionality_index"]

}).dropna()

classification_df["criticality_cat"] = (
    classification_df["criticality_cat"]
    .str.strip()
    .str.upper()
)

classification_df["intersectionality_cat"] = (
    classification_df["intersectionality_cat"]
    .str.strip()
    .str.upper()
)

############################################################################################

classification_df["intersectionality_cat"] = (
    classification_df["intersectionality_cat"]
    .replace({
        "CRÃTICO": "CRITICO"
    })
)

classification_df["criticality_cat"] = (
    classification_df["criticality_cat"]
    .replace({
        "CRÍTICO": "CRITICO"
    })
)


##############################################################################


##############################################################################
# Convert categories to numerical coordinates
##############################################################################

category_map = {

    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4

}

classification_df["criticality_num"] = (
    classification_df["criticality_cat"]
    .map(category_map)
)

classification_df["intersectionality_num"] = (
    classification_df["intersectionality_cat"]
    .map(category_map)
)

##############################################################################
# Quadrants
##############################################################################

def assign_quadrant(row):

    low_crit = row["criticality_cat"] in [
        "BAJO",
        "MEDIO"
    ]

    low_int = row["intersectionality_cat"] in [
        "BAJO",
        "MEDIO"
    ]

    if low_crit and low_int:
        return "A"

    elif low_crit and not low_int:
        return "B"

    elif not low_crit and low_int:
        return "C"

    else:
        return "D"


classification_df["quadrant"] = (
    classification_df
    .apply(assign_quadrant, axis=1)
)

quadrant_counts = (
    classification_df["quadrant"]
    .value_counts()
    .sort_index()
)

quadrant_percentages = (
    quadrant_counts
    / len(classification_df)
    * 100
)


#print(quadrant_counts)


pd.crosstab(
    classification_df["criticality_cat"],
    classification_df["intersectionality_cat"]
)


###########################################################################################################
##############################################################################
# Risk Matrix
##############################################################################

import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

risk_matrix = pd.crosstab(

    classification_df["intersectionality_cat"],
    classification_df["criticality_cat"]

)

order = [
    "BAJO",
    "MEDIO",
    "ALTO",
    "CRITICO"
]

risk_matrix = risk_matrix.reindex(
    index=order,
    columns=order,
    fill_value=0
)

##############################################################################
# Percentages
##############################################################################

risk_matrix_pct = (
    risk_matrix
    / risk_matrix.values.sum()
    * 100
)

##############################################################################
# Text labels
##############################################################################

annotations = risk_matrix.copy().astype(str)

for row in risk_matrix.index:

    for col in risk_matrix.columns:

        annotations.loc[row, col] = (
            f"{risk_matrix_pct.loc[row,col]:.1f}%\n"
            f"(n={risk_matrix.loc[row,col]:,})"
        )

##############################################################################
# Custom risk color palette
##############################################################################

risk_colors = LinearSegmentedColormap.from_list(

    "risk",

    [

        "#1a9850",   # green
        "#91cf60",   # light green
        "#ffffbf",   # yellow
        "#fdae61",   # orange
        "#d73027",   # red
        "#7f0000"    # dark red

    ]

)

##############################################################################
# Plot
##############################################################################

risk_scores = pd.DataFrame(

    [
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7]
    ],

    index=["LOW", "MEDIUM", "HIGH", "CRITIC"],
    columns=["LOW", "MEDIUM", "HIGH", "CRITIC"]

)

#######################################################
plt.figure(figsize=(12,9))

ax = sns.heatmap(

    risk_scores,

    annot=annotations,      # keeps your percentages and n

    fmt="",

    cmap="RdYlGn_r",

    vmin=1,
    vmax=7,

    linewidths=2,
    linecolor="white",

    square=True,

    annot_kws={
        "fontsize":14,
        "fontweight":"bold"
    },

    cbar=False

)


ax.add_patch(

    plt.Rectangle(
        (3,3),
        1,
        1,
        fill=False,
        edgecolor="black",
        linewidth=5
    )

)
##############################################################################
# Labels
##############################################################################

plt.xlabel(
    "Household Criticality Index",
    fontsize=14,
    fontweight="bold"
)

plt.ylabel(
    "Intersectionality Index",
    fontsize=14,
    fontweight="bold"
)

plt.title(
    "Integrated Household Vulnerability Risk Matrix",
    fontsize=16,
    fontweight="bold"
)

##############################################################################
# Highlight priority cell
##############################################################################

ax.add_patch(

    plt.Rectangle(
        (3,3),
        1,
        1,
        fill=False,
        edgecolor="black",
        lw=5
    )

)

##############################################################################
# Quadrant interpretation
##############################################################################

plt.figtext(
    0.5,
    0.94,
    (
        "Green = Lower Vulnerability   |   "
        "Yellow = Moderate Vulnerability   |   "
        "Orange = High Vulnerability   |   "
        "Dark Red = Priority Group"
    ),
    ha="center",
    fontsize=11,
    fontweight="bold"
)

plt.tight_layout(rect=[0,0,1,0.92])

plt.show()