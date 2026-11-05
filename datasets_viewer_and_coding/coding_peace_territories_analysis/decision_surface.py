import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

# retrieve the dataframes

dtypes_home = {
    # ✅ identifiers
    "id_general": "Int64",

    # ✅ categorical / text (due to mixed values like PNC, N/A, ###)
    "nombre_alcaldia": "string",
    "nombre_territorio_paz": "string",
    "nombre_poligono_paz": "string",
    "tipo_vivienda": "string",
    "tipo_tenencia": "string",
    "animal_compania": "string",
    "menores_sin_estudiar": "string",
    "motivo_sin_estudiar": "string",
    "otro_motivo_sin_estudiar": "string",
    "menores_trabajando": "string",
    "mayores65_trabajando": "string",
    "PFE_vulnerabilidad_geologica": "string",
    "PFE_vulnerabilidad_hidrometeorologica": "string",
    "agua_acceso": "string",
    "agua_frecuencia": "string",
    "agua_metodo_acceso": "string",
    "agua_servicios_percepcion": "string",
    "electricidad_acceso": "string",
    "electricidad_tipo": "string",
    "electricidad_percepcion": "string",
    "gas_acceso": "string",
    "gas_tipo_toma": "string",
    "gas_percepcion": "string",
    "drenaje_acceso": "string",
    "drenaje_frecuencia_fallas": "string",
    "basura_frecuencia": "string",
    "existen_botones_panico": "string",
    "botones_panico_usado": "string",
    "deportivos_nivelseguridad": "string",
    "deportivos_serviciosadecuados": "string",
    "tipo_albergue": "string",
    "estado_de_instalaciones_albergues": "string",
    "albergues_nivelseguridad": "string",
    "serviciosadecuados_albergues": "string",
    "tiempotraslado_albergues": "string",
    "gimnasios_nivelseguridad": "string",
    "gimnasios_serviciosadecuados": "string",
    "festividad_nombre": "string",
    "festividades_temporalidad": "string",
    "riesgo_comunidadindigena": "string",
    "indigena_seguridad_violencia": "string",
    "indigena_seguridad_zonas_riesgo": "string",
    "jovenes_trabajo_estudio": "string",
    "zonas_riesgo_para_jovenes": "string",
    "jovenes_seguridad_violencia": "string",
    "jefa_de_hogar": "string",
    "zonas_riesgo_para_mujeres": "string",
    "mujeres_seguridad_violencia": "string",
    "riesgo_precipitacion": "string",
    "riesgo_Inundacion": "string",
    "riesgo_tormenta_electrica": "string",
    "riesgo_granizo": "string",
    "riesgo_tem_max": "string",
    "riesgo_tem_min": "string",
    "riesgo_nevada": "string",
    "riesgo_susceptibilidad_ladera": "string",
    "vul_social": "string",
    "vulnerabilidad_social_FR": "string",
    "riesgo_sismo": "string",
    "riesgo_integrado": "string",
    "Intensidad_recurrencia_encharcamiento": "string",

    # numeric but safe 
    "anios_residencia": "string",   # ⚠ contains numeric but keep as string (safe)
    "cantidad_personas": "string",      # ⚠ sometimes ###
    "hacinamiento": "string",  
    "hacinamiento_cat": "string", #decimals + ###
    "cant_dormi": "string",   # ⚠ ####
    "cantidad_menores_trabajando": "string",
    "cantidad_mayores65_trabajando": "string"
}

df_home = pd.read_csv(
    "df_vector_hogar_160726_publica.csv",
    sep=";",
    decimal=",",
    dtype=dtypes_home,
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
####################################################################################################
#changing to numeric
#temp = pd.to_numeric(
#    classification_df["criticality"],
#    errors="coerce"
#)

#print(
#    "Criticality NaNs:",
#    temp.isna().sum()
#)

#classification_df["criticality"] = temp



#temp = pd.to_numeric(
#    classification_df["intersectionality"],
#    errors="coerce"
#)

#print(
#    "intersectionality NaNs:",
#    temp.isna().sum()
#)

#classification_df["intersectionality"] = temp

#########################################################################################
#removing na
classification_df = classification_df.dropna()



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
########################################################################

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