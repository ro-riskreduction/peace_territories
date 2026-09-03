import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#retrieve the dataframes with parsed dates

dtypes = {
    "id_ind": "Int64",
    "id_general": "Int64",
    "nombre_alcaldia": "string",
    "nombre_territorio_paz": "string",
    "nombre_poligono_paz": "string",
    "edad": "string",
    "grupo_etario": "string",
    "sexo": "string",
    "nivel_estudios": "string",
    "ocupacion_categoria": "string",
    "estado_civil": "string",
    "pertenece_comunidad_originaria": "string",
    "tipo_de_discapacidad": "string",
    "discapacidad_motriz": "string",
    "sintomas_depresion_ansiedad": "string",
    "ingreso_mensual": "string",
    "fuente_ingreso": "string",
    "tiene_empleo": "string",
    "realiza_labores_cuidados": "string",
    "frecuencia_partip_comunitaria": "string",
    "tiene_smartphone": "string",
    "transporte_publico_que_utiliza": "string",
    "hubo_act_vecinales_ultimo_anio": "string",
    "participo_act_vecinales": "string",
    "le_gustaria_participar_en_act_vecinales": "string",
    "se_siente_parte_comunidad": "string",
    "lengua_originaria_cual": "string"
}

df_ind = pd.read_csv(
    "df_vector_ind_16072026_limpia.csv",
    dtype=dtypes,
    parse_dates=["fecha_encuesta"],
    dayfirst=True,
    encoding="latin1",   
    low_memory=False     
)

df_ind.set_index("fecha_encuesta")


date_ind = df_ind["fecha_encuesta"]
date_ind = date_ind.drop(date_ind.index[[0]])


plt.figure()
plt.hist(date_ind, bins=34)
plt.xlabel("Date")
plt.ylabel("Number of surveys")
plt.title("Distribution of surveys")
plt.xticks(rotation = 90)
fig = plt.gcf()
fig.text(0.5, -0.2, 'Figure 1', ha='center')
plt.show()