import csv
import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.stats import entropy, mode

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
    sep= ";",
    dtype=dtypes_home,
    dayfirst=True,
    encoding="latin1",
    low_memory=False
)

df_home.columns = df_home.columns.str.strip()

if "fecha_encuesta" in df_home.columns:
    df_home["fecha_encuesta"] = pd.to_datetime(
        df_home["fecha_encuesta"],
        dayfirst=True,
        errors="coerce"
    )


# Define dtype per column for the ind df


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
    encoding="latin1",   # ✅ fix Unicode error
    low_memory=False     # ✅ suppress dtype warning
)

#############################################################
#Definition for normalizing
def normalize_text(series):

    return (
        series
        .astype("string")
        .str.strip()
        .replace({
            "SÃ­": "SI",
            "Sí": "SI",
            "si": "SI",
            "Si": "SI",

            "No": "NO",
            "no": "NO",

            "mucho": "MUCHO",
            "poco": "POCO",
            "nada": "NADA",

            "trabajo": "TRABAJO",
            "pension": "PENSION",
            "programas_sociales": "PROGRAMAS_SOCIALES",
            "otro": "OTRO",

            "casado": "CASADO",
            "soltero": "SOLTERO",
            "divorciado": "DIVORCIADO",
            "union": "UNION",
            "viudo": "VIUDO",

            "casa": "CASA",
            "departamento": "DEPARTAMENTO",
            "cuarto": "CUARTO",
            "asentamiento": "ASENTAMIENTO",

            "rentada": "RENTADA",
            "propia": "PROPIA",
            "prestada": "PRESTADA",
            "otra": "OTRA",

            "60 y mÃ¡s": "60 Y MAS",
            "60 y más": "60 Y MAS",
        })
    )
#######################################################
#Normalization

problem_columns_ind = [

    "tipo_de_discapacidad",
    "frecuencia_partip_comunitaria",
    "participo_act_vecinales",
    "ingreso_mensual",
    "hubo_act_vecinales_ultimo_anio",
    "grupo_etario",
    "fuente_ingreso",
    "tiene_smartphone",
    "tiene_empleo",
    "realiza_labores_cuidados",
    "se_siente_parte_comunidad",
    "le_gustaria_participar_en_act_vecinales",
    "estado_civil",
    "sexo",
    "nivel_estudios",
    "discapacidad_motriz",
    "sintomas_depresion_ansiedad"
]

for col in problem_columns_ind:

    if col in df_ind.columns:
        df_ind[col] = normalize_text(df_ind[col])


problem_columns_home = [

    "riesgo_comunidadindigena",
    "existen_botones_panico",
    "tipo_tenencia",
    "tipo_vivienda",
    "mayores65_trabajando",
    "menores_trabajando",
    "agua_acceso",
    "electricidad_acceso",
    "drenaje_acceso",
    "gas_acceso",
    "botones_panico_usado"
]

for col in problem_columns_home:

    if col in df_home.columns:
        df_home[col] = normalize_text(df_home[col])

###################################################
# We are moving to create the arrays from the relevant vectors from our datasets while encoding
#################################################


inund_array = (
    df_home["riesgo_Inundacion"]
    .map({
        "Muy Bajo": 1,
        "Bajo": 2,
        "Medio": 3,
        "Alto": 4,
        "Muy Alto": 5
    })
    .to_numpy()
)

anios_residencia_array = (
    df_home["anios_residencia"]
    .map({
        "0":0,
        "1":1,
        "2":2,
        "3":3,
        "4":4,
        "5":5,
        "6":6,
        "7":7,
        "8":8,
        "9":9,
        "10":10,
        "11":11,
        "12":12,
        "13":13,
        "14":14,
        "15":15,
        "16":16,
        "17":17,
        "18":18,
        "19":19,
        "20":20,
        "21":21,
        "22":22,
        "23":23,
        "24":24,
        "25":25,
        "26":26,
        "27":27,
        "28":28,
        "29":29,
        "30":30,
        "31":31,
        "32":32,
        "33":33,
        "34":34,
        "35":35,
        "36":36,
        "37":37,
        "38":38,
        "39":39,
        "40":40,
        "41":41,
        "42":42,
        "43":43,
        "44":44,
        "45":45,
        "46":46,
        "47":47,
        "48":48,
        "49":49,
        "50":50,
        "51":51,
        "52":52,
        "53":53,
        "54":54,
        "55":55,
        "56":56,
        "57":57,
        "58":58,
        "59":59,
        "60":60,
        "61":61,
        "62":62,
        "63":63,
        "64":64,
        "65":65,
        "66":66,
        "67":67,
        "68":68,
        "69":69,
        "70":70,
        "71":71,
        "72":72,
        "73":73,
        "74":74,
        "75":75,
        "76":76,
        "77":77,
        "78":78,
        "79":79,
        "80":80,
        "81":81,
        "82":82,
        "83":83,
        "84":84,
        "85":85,
        "86":86,
        "87":87,
        "88":88,
        "89":89,
        "90":90,
        "91":91,
        "92":92,
        "93":93,
        "94":94,
        "95":95,
        "96":96,
        "97":97,
        "98":98,
        "99":99,
        "100":100,
        "101":101,
        "104":104,
        "110":110, 
        "111":111,
        "115":115,
        "118":118,
        "119":119,
        "120":120,
        "125":125,
        "128":128,
        "129":129,
        "130":130,
        "135":135,
        "138":138,
        "139":139,
        "140":140,
        "145":145,
        "149":149,
        "150":150,            
    })

.to_numpy()
)

tipo_vivienda_array = (
    df_home["tipo_vivienda"]
    .map({
        "ASENTAMIENTO": 1,
        "CASA": 2,
        "CUARTO": 3,
        "DEPARTAMENTO": 4
    })
    .to_numpy()
)

tipo_tenencia_array = (
    df_home["tipo_tenencia"]
    .map({
        "OTRA": 1,
        "PRESTADA": 2,
        "PROPIA": 3,
        "RENTADA": 4
    })
    .to_numpy()
)

cantidad_personas_array = (
df_home["cantidad_personas"]
.map({
        "1":1,
        "2":2,
        "3":3,
        "4":4,
        "5":5,
        "6":6,
        "7":7,
        "8":8,
        "9":9,
        "10":10,
        "11":11,
        "12":12,
        "13":13,
        "14":14,
        "15":15,
        "16":16,
        "17":17,
        "18":18,
        "19":19,
        "20":20
    })
    .to_numpy()
)

hacinamiento_cat_array = (
    df_home["hacinamiento_cat"]
    .map({
        "SIN HACINAMIENTO": 0,
        "CON HACINAMIENTO": 1
    })
    .to_numpy()
)

cantidad_dormitorios_array = (
    df_home["cant_dormi"]
    .map({
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "10": 10,
        "11": 11,
        "12": 12,
        "13": 13,
        "14": 14,
        "15": 15,
        "16": 16,
        "17": 17,
        "18": 18,
        "19": 19,
        "20": 20,
        "21": 21,
        "22": 22
    })
    .to_numpy()
)

animal_compania_array = (
    df_home["animal_compania"]
    .map({
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6":6,
        "7":7,
        "8":8,
        "9":9,
        "10":10,
        "11":11,
        "12":12,
        "13":13,
        "14":14,
        "15":15,
        "16":16,
        "17":17,
        "18":18,
        "19":19,
        "20":20,
        "21":21,
        "22":22,
        "23":23,
        "24":24,
        "25":25,
        "26":26,
        "27":27,
        "28":28,
        "29":29,
        "30":30,
        "31":31,
        "33":33,
        "35":35,
        "40":40,
        "41":41,
        "43":43,
        "47":47,
        "50":50,
        "51":51,
        "54":54,
    })
    .to_numpy()
)

menores_sin_estudiar_array = (
    df_home["menores_sin_estudiar"]
    .map({
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "7": 7,
        "10": 10,
        "11": 11,
        "12": 12,
    })
    .to_numpy()
)

motivo_sin_estudiar_array = (
    df_home["motivo_sin_estudiar"]
    .map({
        "DISTANCIA": 1,
        "ECONOMIA": 2,
        "ECONOMIA, TRABAJO": 3,
        "TRABAJO": 4,
        "DISCRIMINACION": 5,
        "DISTANCIA, DISCRIMINACION": 6, 
        "DISTANCIA, TRABAJO": 7,
        "ECONOMIA, DISCRIMINACION": 8,
        "ECONOMIA, DISTANCIA": 9,
        "ECONOMIA, DISTANCIA, DISCRIMINACION": 10,
        "ECONOMIA, DISTANCIA, TRABAJO": 11,
        "ECONOMIA, DISTANCIA, TRABAJO, DISCRIMINACION": 12,
    })
    .to_numpy()
)

otro_motivo_sin_estudiar_array = (
 df_home["otro_motivo_sin_estudiar"]
    .map({
        "Actualmente estudia (otro nivel)": 1,
        "Económico / Trabajo": 2,
        "Falta de voluntad / Desinterés": 3,
        "Problemas escolares": 4,
        "Sin información específica": 5,
        "Situación familiar": 6,
        "Embarazo / Maternidad": 7,
        "Salud / Discapacidad": 8,
    })
    .to_numpy()
)

agua_servicios_percepcion_array = (
df_home["agua_servicios_percepcion"]
 .map({
    	"MUY BUENA": 5,
    	"BUENA": 4,
    	"REGULAR": 3,
    	"MALA": 2,
    	"MUY MALA": 1
    })
    .to_numpy()
)
  
electricidad_acceso_array = (
    df_home["electricidad_acceso"]
    .map({
    "SI": 1,
    "NO": 0,
    "PNC": np.nan
})
    .to_numpy()
)
 
electricidad_tipo_array = (
    df_home["electricidad_tipo"]
    .map({
    	"PRIVADA": 1,
    	"PUBLICA": 2,
        "NO": 0
    })
    .to_numpy()
)
 
electricidad_percepcion_array = (
    df_home["electricidad_percepcion"]
    .map({
    	"MUY BUENA": 5,
    	"BUENA": 4,
    	"REGULAR": 3,
    	"MALA": 2,
    	"MUY MALA": 1
    })
    .to_numpy()
)
 
gas_acceso_array = (
    df_home["gas_acceso"]
    .map({
    "SI": 1,
    "NO": 0,
    "PNC": np.nan
})
    .to_numpy()
)
 
gas_tipo_toma_array = (
    df_home["gas_tipo_toma"]
    .map({
    	"CILINDRO": 1,
    	"ESTACIONARIO": 2,
    	"NATURAL": 3,
    	"OTRO": 4,
        "NO": 0
    })
    .to_numpy()
)
 
gas_percepcion_array = (
	df_home["gas_percepcion"]
    .map({
    	"MUY BUENA": 5,
    	"BUENA": 4,
    	"REGULAR": 3,
    	"MALA": 2,
    	"MUY MALA": 1
    })
    .to_numpy()
)
 
drenaje_acceso_array = (
	df_home["drenaje_acceso"]
    .map({
    "SI": 1,
    "NO": 0,
    "PNC": np.nan
})
    .to_numpy()
)
 
drenaje_frecuencia_fallas_array = (
    df_home["drenaje_frecuencia_fallas"]
    .map({
    	"NUNCA": 1,
    	"RARA VEZ": 2,
    	"OCASIONAL": 3,
    	"FRECUENTE": 4,
    	"MUY FRECUENTE": 5,
        "NO": 0
    })
    .to_numpy()
)
 
basura_frecuencia_array = (
	df_home["basura_frecuencia"]
    .map({
    	"DIARIO": 5,
    	"2-3 VECES A LA SEMANA": 4,
    	"UNA VEZ A LA SEMANA": 3,
    	"RARA VEZ": 2,
    	"NUNCA": 1
    })
    .to_numpy()
)

existen_botones_panico_array = (
    df_home["existen_botones_panico"]
    .map({
    "SI": 1,
    "NO": 0,
    "PNC": np.nan
})
    .to_numpy()
)

botones_panico_usado_array = (
    df_home["botones_panico_usado"]
    .map({
    "SI": 1,
    "NO": 0,
    "PNC": np.nan
})
    .to_numpy()
)

deportivos_nivelseguridad_array = (
    df_home["deportivos_nivelseguridad"]
    .map({
        "BUENO": 3,
        "REGULAR": 2,
        "MALO": 1,
        "NO": 0
    })
    .to_numpy()
)

deportivos_serviciosadecuados_array = (
    df_home["deportivos_serviciosadecuados"]
    .map({
    "SI": 1,
    "NO": 0,
    "PNC": np.nan
})
    .to_numpy()
)

tipo_albergue_array = (
    df_home["tipo_albergue"]
    .map({
        "PUBLICO": 1,
        "PRIVADO": 2
    })
    .to_numpy()
)

estado_de_instalaciones_albergues_array = (
    df_home["estado_de_instalaciones_albergues"]
    .map({
        "BUENO": 3,
        "REGULAR": 2,
        "MALO": 1
    })
    .to_numpy()
)

albergues_nivelseguridad_array = (
    df_home["albergues_nivelseguridad"]
    .map({
        "BUENO": 3,
        "REGULAR": 2,
        "MALO": 1
    })
    .to_numpy()
)

serviciosadecuados_albergues_array = (
    df_home["serviciosadecuados_albergues"]
    .map({
    "SI": 1,
    "NO": 0,
    "PNC": np.nan
})
    .to_numpy()
)

tiempotraslado_albergues_array = (
    df_home["tiempotraslado_albergues"]
    .map({
        "1": 1,
        "2": 2,
        "3": 3, 
        "4": 4,
        "5": 5,
        "6": 6, 
        "7": 7,
        "8": 8,
        "9": 9, 
        "10": 10,
        "11": 11,
        "12": 12,
        "13": 13, 
        "15": 15,
        "16": 16, 
        "17": 17,
        "18": 18,
        "20": 20,
        "23": 23, 
        "24": 24,
        "25": 25,
        "30": 30,
        "40": 40,
        "45": 45,
        "50": 50
    })
    .to_numpy()
)

gimnasios_nivelseguridad_array = (
    df_home["gimnasios_nivelseguridad"]
    .map({
        "BUENO": 3,
        "REGULAR": 2,
        "MALO": 1
    })
    .to_numpy()
)

gimnasios_serviciosadecuados_array = (
    df_home["gimnasios_serviciosadecuados"]
    .map({
    "SI": 1,
    "NO": 0,
    "PNC": np.nan
})
    .to_numpy()
)

festivities_name_array = (
 df_home["festividad_nombre"]
 .map({
        "Aniversario de Mercado o Colonia": 1,
        "Dia de Muertos": 2,
        "Fechas Civicas y Conmemorativas": 3,
        "Ferias del Pueblo o Barrio": 4,
        "Ferias Gastronomicas o Tematicas": 5,
        "Festividades Decembrinas": 6,
        "Fiestas Patronales y Religiosas": 7,
        "Semana Santa y Carnaval": 8,
        "Otra / Por Clasificar": 9
    })
 .to_numpy()
 )

festivities_time_array = (
 df_home["festividades_temporalidad"]
 .map({
        "Enero": 1,
        "Febrero": 2,
        "Marzo": 3,
        "Abril": 4,
        "Mayo": 5,
        "Junio": 6,
        "Julio": 7,
        "Agosto": 8,
        "Septiembre": 9,
        "Octubre": 10,
        "Noviembre": 11,
        "Diciembre": 12,
        "Anual (Sin mes especifico)": 13,
        "Festividad movil (Semana Santa / Cuaresma)": 14
    })
 .to_numpy()
 )

riesgo_comunidadindigena_array = (
 df_home["riesgo_comunidadindigena"]
 .map({
    "SI": 1,
    "NO": 0,
    "PNC": np.nan
})
    .to_numpy()
)

indigena_seguridad_violencia_array = (
 df_home["indigena_seguridad_violencia"]
 .map({
    "SI": 1,
    "NO": 0,
    "PNC": np.nan
})
    .to_numpy()
)

indigena_seguridad_zonas_riesgo_array = (
df_home["indigena_seguridad_zonas_riesgo"]
 .map({
        "Espacio Publico y Comunitario": 1,
        "Transporte y Conectividad Territorial": 2,
        "Zonas de Comercio y comecio ilicito": 3,
        "Otro espacio descrito": 4,
        "NINGUNA": 5,
    })
 .to_numpy()
 )

jovenes_trabajo_estudio_array = (
 df_home["jovenes_trabajo_estudio"]
 .map({
    "SI": 1,
    "NO": 0,
    "PNC": np.nan
})
    .to_numpy()
)

zonas_riesgo_para_jovenes_array = (
 df_home["zonas_riesgo_para_jovenes"]
 .map({
        "Espacio Publico y Comunitario": 1,
        "Entorno Institucional / Privado": 2,
        "Transporte y Conectividad Territorial": 3,
        "Zonas de Comercio y comecio ilicito": 4,
        "Otro espacio descrito": 5,
        "NINGUNA": 6
    })
  .to_numpy()
 )

jovenes_seguridad_violencia_array = (
 df_home["jovenes_seguridad_violencia"]
 .map({
    "SI": 1,
    "NO": 0,
    "PNC": np.nan
})
    .to_numpy()
)

jefa_de_hogar_array = (
 df_home["jefa_de_hogar"]
 .map({
    "SI": 1,
    "NO": 0,
    "PNC": np.nan
})
    .to_numpy()
)

zonas_riesgo_para_mujeres_array = (
 df_home["zonas_riesgo_para_mujeres"]
 .map({
        "Espacio Publico y Comunitario": 1,
        "Entorno Institucional / Privado": 2,
        "Transporte y Conectividad Territorial": 3,
        "Zonas de comercio y comercio ilicito": 4,
        "Otro espacio descrito": 5,
        "NINGUNA": 6
     })
 .to_numpy()
 )

mujeres_seguridad_violencia_array = (
 df_home["mujeres_seguridad_violencia"]
 .map({
    "SI": 1,
    "NO": 0,
    "PNC": np.nan
})
    .to_numpy()
)

riesgo_precipitacion_array = (
 df_home["riesgo_precipitacion"]
 .map({
        "Muy Bajo": 1,
        "Bajo": 2,
        "Medio": 3,
        "Alto": 4
    })
 .to_numpy()
 )

riesgo_tormenta_electrica_array = (
 df_home["riesgo_tormenta_electrica"]
 .map({
        "Muy Bajo": 1,
        "Bajo": 2,
        "Medio": 3,
        "Alto": 4,
        "Muy Alto": 5
    })
 .to_numpy()
 )

riesgo_granizo_array = (
 df_home["riesgo_granizo"]
 .map({
        "Muy Bajo": 1,
        "Bajo": 2,
        "Medio": 3,
        "Alto": 4,
        "Muy Alto": 5
    })
 .to_numpy()
 )

riesgo_tem_max_array = (
 df_home["riesgo_tem_max"]
 .map({
        "Bajo": 1,
        "Medio": 2,
        "Alto": 3
    })
 .to_numpy()
 )

riesgo_tem_min_array = (
 df_home["riesgo_tem_min"]
 .map({
        "Muy Bajo": 1,
        "Bajo": 2,
        "Medio": 3,
        "Alto": 4,
        "Muy Alto": 5
    })
 .to_numpy()
 )

riesgo_nevada_array = (
 df_home["riesgo_nevada"]
 .map({
        "Muy Bajo": 1,
        "Bajo": 2,
        "Medio": 3,
        "Alto": 4
    })
 .to_numpy()
 )

riesgo_susceptibilidad_ladera_array = (
 df_home["riesgo_susceptibilidad_ladera"]
 .map({
        "Muy Bajo": 1,
        "Bajo": 2,
        "Medio": 3,
        "Alto": 4,
        "Muy Alto": 5
    })
 .to_numpy()
 )

vul_social_array = (
 df_home["vul_social"]
 .map({
       "Muy Bajo": 1,
       "Bajo": 2,
       "Medio": 3,
       "Alto": 4,
       "Muy Alto": 5
    })
 .to_numpy()
 )

vulnerabilidad_social_FR_array = (
 df_home["vulnerabilidad_social_FR"]
 .map({
        "Muy Bajo": 1,
        "Bajo": 2,
        "Medio": 3,
        "Alto": 4,
        "Muy Alto": 5
    })
 .to_numpy()
 )

riesgo_sismo_array = (
 df_home["riesgo_sismo"]
 .map({
        "Medio": 1,
        "Alto": 2,
        "Muy Alto": 3
    })
 .to_numpy()
 )

riesgo_integrado_array = (
 df_home["riesgo_integrado"]
 .map({
        "Muy Bajo": 1,
        "Bajo": 2,
        "Medio": 3,
        "Alto": 4,
        "Muy Alto": 5
    })
 .to_numpy()
 )

Intensidad_recurrencia_encharcamiento_array = (
 df_home["Intensidad_recurrencia_encharcamiento"]
 .map({
        "Sin registro de recurrencia": 1,
        "Muy Bajo": 2,
        "Bajo": 3,
        "Medio": 4,
        "Alto": 5
    })
 .to_numpy()
 )

menores_trabajando_array = (
	df_home["menores_trabajando"]
	.map({
    "SI": 1,
    "NO": 0,
    "PNC": np.nan
})
    .to_numpy()
)

cantidad_menores_trabajando_array = (
	df_home["cantidad_menores_trabajando"]
	.map({
    	"0": 0,
    	"1": 1,
    	"2": 2,
    	"3": 3,
    	"4": 4,
    	"5": 5,
    	"11": 11,
    	"15": 15,
        "12": 12
	})
	.to_numpy()
)

mayores65_trabajando_array = (
	df_home["mayores65_trabajando"]
	.map({
    "SI": 1,
    "NO": 0,
    "PNC": np.nan
})
    .to_numpy()
)

cantidad_mayores65_trabajando_array = (
	df_home["cantidad_mayores65_trabajando"]
	.map({
    	"1": 1,
    	"2": 2,
    	"3": 3,
    	"4": 4,
    	"5": 5,
    	"6": 6,
    	"7": 7,
    	"8": 8,
    	"11": 11,
    	"12": 12,
        "0": 0
 	})
	.to_numpy()
)

PFE_vulnerabilidad_geologica_array = (
	df_home["PFE_vulnerabilidad_geologica"]
	.map({
    	"Alta": 3,
    	"Baja": 1,
    	"Media": 2,
        "No determinado": 0   
 	})
	.to_numpy()
)

PFE_vulnerabilidad_hidrometeorologica_array = (
	df_home["PFE_vulnerabilidad_hidrometeorologica"]
	.map({
    	"Bajo": 1,
    	"Media": 2,
    	"Alta": 3,
        "No determinado": 0
  	})
	.to_numpy()
)
 
agua_acceso_array = (
	df_home["agua_acceso"]
	.map({
    "SI": 1,
    "NO": 0,
    "PNC": np.nan
})
    .to_numpy()
)
 
agua_frecuencia_array = (
	df_home["agua_frecuencia"]
	.map({
    	"DIARIO": 4,
    	"UNA VEZ A LA SEMANA": 2,
    	"UNA VEZ AL MES": 1,
    	"VARIAS VECES POR SEMANA": 3,
    	"NO": 0
	})
	.to_numpy()
)

agua_metodo_acceso_array = (
    df_home["agua_metodo_acceso"]
    .map({
        "ACARREO": 3,
        "PIPA GRATUITA": 1,
        "PIPA PRIVADA": 2,
        "RED": 4,
        "OTRO": 5
    })
    .to_numpy()
)

agua_servicios_percepcion_array = (
	df_home["agua_servicios_percepcion"]
	.map({
    	"BUENA": 4,
    	"MUY BUENA": 5,
    	"MALA": 2,
    	"MUY MALA": 1,
    	"REGULAR": 3
	})
	.to_numpy()
)

####################################################################################################################################################################################
###ind_array###

edad_array= (
    df_ind["edad"]
        .map({
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "10": 10,
        "11": 11,
        "12": 12,
        "13": 13,
        "14": 14,
        "15": 15,
        "16": 16,
        "17": 17,
        "18": 18,
        "19": 19,
        "20": 20,
        "21": 21,
        "22": 22,
        "23": 23,
        "24": 24,
        "25": 25,
        "26": 26,
        "27": 27,
        "28": 28,
        "29": 29,
        "30": 30,
        "31": 31,
        "32": 32,
        "33": 33,
        "34": 34,
        "35": 35,
        "36": 36,
        "37": 37,
        "38": 38,
        "39": 39,
        "40": 40,
        "41": 41,
        "42": 42,
        "43": 43,
        "44": 44,
        "45": 45,
        "46": 46,
        "47": 47,
        "48": 48,
        "49": 49,
        "50": 50,
        "51": 51,
        "52": 52,
        "53": 53,
        "54": 54,
        "55": 55,
        "56": 56,
        "57": 57,
        "58": 58,
        "59": 59,
        "60": 60,
        "61": 61,
        "62": 62,
        "63": 63,
        "64": 64,
        "65": 65,
        "66": 66,
        "67": 67,
        "68": 68,
        "69": 69,
        "70": 70,
        "71": 71,
        "72": 72,
        "73": 73,
        "74": 74,
        "75": 75,
        "76": 76,
        "77": 77,
        "78": 78,
        "79": 79,
        "80": 80,
        "81": 81,
        "82": 82,
        "83": 83,
        "84": 84,
        "85": 85,
        "86": 86,
        "87": 87,
        "88": 88,
        "89": 89,
        "90": 90,
        "91": 91,
        "92": 92,
        "93": 93,
        "94": 94,
        "95": 95,
        "96": 96,
        "97": 97,
        "98": 98,
        "99": 99,
        "100": 100,
        "101": 101,
        "102": 102,
        "103": 103,
        "104": 104,
        "105": 105
    })
.to_numpy()
)

grupo_etario_array = (
    df_ind["grupo_etario"]
    .map({
        "0-5 PRIMERA INFANCIA": 1,
        "12-17 ADOLESCENCIA": 2,
        "18-29 JUVENTUD": 3,
        "30-59 ADULTEZ": 4,
        "6-11 INFANCIA": 5,
        "60 Y MAS": 6
    })
    .to_numpy()
)

sexo_array = (
    df_ind["sexo"]
    .map({
        "M": 1,
        "H": 2
    })
    .to_numpy()
)

nivel_estudios_array = (
    df_ind["nivel_estudios"]
    .map({
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6
    })
    .to_numpy()
)

occupation_array= (
    df_ind["ocupacion_categoria"]
    .map({
        "Administrativos y Oficinistas": 1,
        "Agricultura y Campo": 2,
        "Ama de casa / Labores del hogar": 3,
        "Atención al Cliente y Call center": 4,
        "Belleza y Estética": 5,
        "Comercio y Ventas": 6,
        "Construcción y Albañilería": 7,
        "Contabilidad y Finanzas": 8,
        "Desempleados o Sin Ocupación": 9,
        "Diseño, Arte y Fotografía": 10,
        "Educación y Academia":11,
        "Estudiantes": 12,
        "Gerencia y Dirección": 13,
        "Ingenierías (General)": 14,
        "Jubilados y Pensionados": 15,
        "Leyes y Derecho": 16,
        "Limpieza y Mantenimiento": 17,
        "Logística y Almacén": 18,
        "Manufactura y Operarios": 19,
        "Marketing y Publicidad": 20,
        "Preparación de Alimentos y Cocina": 21,
        "Recursos Humanos": 22,
        "Sector Salud y Medicina": 23,
        "Seguridad y Vigilancia": 24,
        "Servicios Sociales y Religión": 25,
        "Servicios Técnicos (Electricidad/Plomería)": 26,
        "Servicios Técnicos (Mecánica)": 27,
        "Sistemas y Tecnología": 28,
        "Transporte y Conductores": 29,
        "Turismo y Hotelería": 30,
        "Otros / No clasificado": 31
    })
    .to_numpy()
)

estado_civil_array = (
    df_ind["estado_civil"]
    .map({
        "SOLTERO": 1,
        "CASADO": 2,
        "DIVORCIADO": 3,
        "UNION": 4,
        "VIUDO": 5
    })
    .to_numpy()
)

pertenece_comunidad_originaria_array = (
    df_ind["pertenece_comunidad_originaria"]
    .map({
    "SI": 1,
    "NO": 0,
    "PNC": np.nan
})
    .to_numpy()
)

tipo_de_discapacidad_array = (
    df_ind["tipo_de_discapacidad"]
    .map({
        "Auditiva": 1,
        "Del habla o lenguaje": 2,
        "Intelectual o del aprendizaje": 3,
        "Mental o psicosocial": 4,
        "Motriz": 5,
        "Visual": 6,
        "Otra": 7,
        "No especificada": 8
    })
    .to_numpy()
)

discapacidad_motriz_array = (
    df_ind["discapacidad_motriz"]
    .map({
        "SI": 1,
        "NO": 0,
        "SIN DATO": 0
    })
    .to_numpy()
)

#######################################################################################

#discapacidad_motriz_array = df_ind["discapacidad_motriz"].dropna()

########################################################################################

sintomas_depresion_ansiedad_array = (    
    df_ind["sintomas_depresion_ansiedad"]
    .map({
    "SI": 1,
    "NO": 0
    })
    .to_numpy()
)

ingreso_mensual_array = (
    df_ind["ingreso_mensual"]
    .map({
        "< 5000": 1,
        "> 25001": 5,
        "10001-15000": 3,
        "15001-25000": 4,
        "5001-10000": 2
    })
    .to_numpy()
)

tiene_empleo_array = (    
    df_ind["tiene_empleo"]
    .map({
    "SI": 1,
    "NO": 0,
    "PNC": np.nan
})
    .to_numpy()
)

fuente_ingreso_array = (    
    df_ind["fuente_ingreso"]
    .map({
        "TRABAJO": 1,
        "PROGRAMAS_SOCIALES": 2,
        "PENSION": 3,
        "OTRA": 4
    })
    .to_numpy()
)

realiza_labores_cuidados_array = (
    df_ind["realiza_labores_cuidados"]
    .map({
        "NO": 0,
        "SI": 1,
        "PNC": np.nan
    })
    .to_numpy()
)

frecuencia_partip_comunitaria_array = (
    df_ind["frecuencia_partip_comunitaria"]
    .map({
        "MUCHO": 2,
        "POCO": 1,
        "NADA": 0,
    })
    .to_numpy()
)

tiene_smartphone_array = (
    df_ind["tiene_smartphone"]
    .map({
    "SI": 1,
    "NO": 0,
    "PNC": np.nan
})
    .to_numpy()
)

transporte_publico_que_utiliza_array = (
    df_ind["transporte_publico_que_utiliza"]
    .map({
        "CABLEBUS": 1,
        "ECOBICI": 2,
        "METRO": 3,
        "METROBÚS": 4,
        "RTP": 5,
        "TREN LIGERO": 6,
        "TROLEBUS": 7,
        "NINGUNO": 8
    })
    .to_numpy()
)

hubo_act_vecinales_ultimo_anio_array = (
    df_ind["hubo_act_vecinales_ultimo_anio"]
    .map({
    "SI": 1,
    "NO": 0,
    "PNC": np.nan
})
    .to_numpy()
)

participo_act_vecinales_array = (
    df_ind["participo_act_vecinales"]
    .map({
    "SI": 1,
    "NO": 0,
    "PNC": np.nan
})
    .to_numpy()
)

le_gustaria_participar_en_act_vecinales_array = (
    df_ind["le_gustaria_participar_en_act_vecinales"]
    .map({
    "SI": 1,
    "NO": 0,
    "PNC": np.nan
})
    .to_numpy()
)

se_siente_parte_comunidad_array = (
    df_ind["se_siente_parte_comunidad"]
    .map({
    "SI": 1,
    "NO": 0,
    "PNC": np.nan
})
    .to_numpy()
)

lengua_originaria_cual_array = (
    df_ind["lengua_originaria_cual"]
    .map({
        "CETZEN": 1,
        "INZINI": 2,
        "MAZAHUA": 3,
        "MAZATECO": 4,
        "MEXICA": 5,
        "MIXTECO": 6,
        "NAHUATL Y OTOMI": 8,
        "NAHUATL": 7,
        "NAHUATL Y ZAPOTECO": 9,
        "NAJIAC": 10,
        "OTOMI": 11,
        "PUREPECHA": 12,
        "TLAPANECO": 13,
        "TOTONACA": 14,
        "TSOLTIZ": 15,
        "ZAPOTECO": 16,
        "ZATECO": 17,
        "ZELTHA": 18,
        "SIN ESPECIFICAR": 19,
        "NO": 20
    })
    .to_numpy()
)

###############################################################################
# We are then creating the clusters of categories acoording to principal vectors

grupo_prioritario_mujeres =  np.vstack (( jefa_de_hogar_array, zonas_riesgo_para_mujeres_array, mujeres_seguridad_violencia_array ) ).T

grupo_prioritario_jovenes=  np.vstack (( jovenes_trabajo_estudio_array, zonas_riesgo_para_jovenes_array, jovenes_seguridad_violencia_array ) ).T

grupo_prioritario_indigena=  np.vstack (( riesgo_comunidadindigena_array, indigena_seguridad_violencia_array, indigena_seguridad_zonas_riesgo_array ) ).T

equipamiento_de_servicios=  np.vstack (( botones_panico_usado_array, tipo_albergue_array, estado_de_instalaciones_albergues_array, serviciosadecuados_albergues_array, tiempotraslado_albergues_array, deportivos_serviciosadecuados_array, gimnasios_serviciosadecuados_array) ).T

seguridad=  np.vstack (( deportivos_nivelseguridad_array, gimnasios_nivelseguridad_array, albergues_nivelseguridad_array) ).T

cultura=  np.vstack (( festivities_name_array, festivities_time_array) ).T

composicion_del_hogar=  np.vstack (( cantidad_personas_array, animal_compania_array, menores_trabajando_array, cantidad_menores_trabajando_array, mayores65_trabajando_array) ).T

vivienda=  np.vstack (( anios_residencia_array, tipo_vivienda_array, tipo_tenencia_array, hacinamiento_cat_array, cantidad_dormitorios_array) ).T

educacion_menores=  np.vstack (( menores_sin_estudiar_array, motivo_sin_estudiar_array, otro_motivo_sin_estudiar_array) ).T

servicios=  np.vstack (( agua_acceso_array, agua_frecuencia_array, agua_metodo_acceso_array, agua_servicios_percepcion_array, electricidad_acceso_array, electricidad_tipo_array, electricidad_percepcion_array, gas_acceso_array, gas_tipo_toma_array, gas_percepcion_array, drenaje_acceso_array, drenaje_frecuencia_fallas_array, basura_frecuencia_array) ).T

riesgo=  np.vstack (( PFE_vulnerabilidad_geologica_array, PFE_vulnerabilidad_hidrometeorologica_array, riesgo_precipitacion_array, inund_array, Intensidad_recurrencia_encharcamiento_array, riesgo_tormenta_electrica_array, riesgo_granizo_array, riesgo_tem_max_array, riesgo_tem_min_array, riesgo_nevada_array, riesgo_susceptibilidad_ladera_array, vul_social_array, vulnerabilidad_social_FR_array, riesgo_sismo_array, riesgo_integrado_array) ).T





### df_ind ###

demographics=  np.vstack (( edad_array, grupo_etario_array, sexo_array, nivel_estudios_array, occupation_array, estado_civil_array) ).T

social_identity=  np.vstack (( pertenece_comunidad_originaria_array, lengua_originaria_cual_array) ).T

vulnerability=  np.vstack (( tipo_de_discapacidad_array, discapacidad_motriz_array, sintomas_depresion_ansiedad_array) ).T

socioeconomic=  np.vstack (( ingreso_mensual_array, fuente_ingreso_array, tiene_empleo_array) ).T

caregivin = realiza_labores_cuidados_array.reshape(-1, 1)

community=  np.vstack (( frecuencia_partip_comunitaria_array, hubo_act_vecinales_ultimo_anio_array, participo_act_vecinales_array, le_gustaria_participar_en_act_vecinales_array, se_siente_parte_comunidad_array) ).T

technology= tiene_smartphone_array.reshape(-1, 1)

transportation= transporte_publico_que_utiliza_array.reshape(-1, 1)



###############################################################################################################


def dispersion_stats(n):

    return {
        "Minimum": np.nanmin(n),
        "Maximum": np.nanmax(n),
        "Range": np.nanmax(n) - np.nanmin(n),
        "P25": np.nanpercentile(n, 25),
        "P75": np.nanpercentile(n, 75),
        "IQR": np.nanpercentile(n, 75) - np.nanpercentile(n, 25),
        "Variance": np.nanvar(n),
        "Std Dev": np.nanstd(n)
    }

datasets = {

    "women_priority_group": {
        "data": grupo_prioritario_mujeres,
        "variables": [
            "female_head_of_household",
            "areas_at_risk_for_women",
            "women_security_violence"
        ]
    },

    "youth_priority_group": {
        "data": grupo_prioritario_jovenes,
        "variables": [
            "youth_work_study",
            "areas_at_risk_for_youth",
            "youth_security_violence"
        ]
    },

    "indigenous_priority_group": {
        "data": grupo_prioritario_indigena,
        "variables": [
            "indigenous_community_risk",
            "indigenous_security_violence",
            "indigenous_security_risk_zones"
        ]
    },

    "security": {
        "data": seguridad,
        "variables": [
            "sports_center_sacurity_level",
            "gyms_security_level",
            "shelter_security_level"
        ]
    },

    "dwelling": {
        "data": vivienda,
        "variables": [
            "years_of_residence",
            "dwelling_type",
            "tenure_type",
            "overcrowding",
            "number_of_bedrooms"
        ]
    },

    "demographics": {
        "data": demographics,
        "variables": [
            "age",
            "age_group",
            "sex",
            "education_level",
            "occupation_category",
            "marital_status"
        ]
    },

"service_equipment": {
        "data": equipamiento_de_servicios,
        "variables": [
            "panic_buttons_used",
            "shelter_type",
            "shelter_facility_condition",
            "shelter_adequate_services",
            "shelter_travel_time",
            "sports_center_adequate_services",
            "gyms_adequate_services"
        ]
    },

    "culture": {
        "data": cultura,
        "variables": [
            "festivities_name",
            "festivities_time"
        ]
    },

    "household_composition": {
        "data": composicion_del_hogar,
        "variables": [
            "number_of_people",
            "companion_animal",
            "working_minors",
            "number_of_working_minors",
            "working_people_over_65"
        ]
    },

    "education_for_minors": {
        "data": educacion_menores,
        "variables": [
            "minors_not_studying",
            "reason_for_not_studying",
            "other_reason_not_studying"
        ]
    },

    "services": {
        "data": servicios,
        "variables": [
            "water_access",
            "water_frequency",
            "water_access_method",
            "water_service_perception",
            "electricity_access",
            "electricity_type",
            "electricity_perception",
            "gas_access",
            "gas_connection_type",
            "gas_perception",
            "drainage_access",
            "drainage_failure_frequency",
            "waste_collection_frequency"
        ]
    },

    "risk": {
        "data": riesgo,
        "variables": [
            "main_source_of_employment_and_geological_vulnerability",
            "main_source_of_employment_and_hydrometeorological_vulnerability",
            "precipitation_risk",
            "flood_risk",
            "intensity_recurrence_waterlogging",
            "thunderstorm_risk",
            "hail_risk",
            "maximum_temperature_risk",
            "minimum_temperature_risk",
            "snowfall_risk",
            "slope_susceptibility_risk",
            "social_vulnerability",
            "social_vulnerability_to_fracturing",
            "seismic_risk",
            "hazard_risk_index"
        ]
    },

    "social_identity": {
        "data": social_identity,
        "variables": [
            "belongs_to_native_community",
            "which_native_language"
        ]
    },

    "vulnerability": {
        "data": vulnerability,
        "variables": [
            "type_of_disability",
            "motor_disability",
            "depression_and_anxiety_symptoms"
        ]
    },

    "socioeconomic": {
        "data": socioeconomic,
        "variables": [
            "monthly_income",
            "source_income",
            "has_a_job"
        ]
    },

    "caregivin": {
        "data": caregivin,
        "variables": [
            "performs_caregiving_duties"
        ]
    },

    "community": {
        "data": community,
        "variables": [
            "frequency_of_community_participation",
            "there_were_neighborhood_activities_over_the_last_year",
            "participated_in_neighborhood_activities",
            "would_you_like_to_participate_in_neighborhood_activities",
            "feels_part_of_the_community"
        ]
    },

    "technology": {
        "data": technology,
        "variables": [
            "has_smartphone"
        ]
    },

    "transportation_means": {
        "data": transportation,
        "variables": [
            "public_transport_used"
        ]
    }
}

####################################################################
variable_types = {

    # =====================
    # CONTINUOUS
    # =====================

    "age": "continuous",
    "years_of_residence": "continuous",
    "number_of_people": "continuous",
    "number_of_bedrooms": "continuous",
    "companion_animal": "continuous",
    "number_of_working_minors": "continuous",
    "shelter_travel_time": "continuous",

    # =====================
    # BINARY
    # =====================

    "female_head_of_household": "binary",
    "women_security_violence": "binary",

    "youth_work_study": "binary",
    "youth_security_violence": "binary",

    "indigenous_community_risk": "binary",
    "indigenous_security_violence": "binary",

    "panic_buttons_used": "binary",

    "shelter_adequate_services": "binary",
    "sports_center_adequate_services": "binary",
    "gyms_adequate_services": "binary",

    "working_minors": "binary",
    "working_people_over_65": "binary",

    "water_access": "binary",
    "electricity_access": "binary",
    "gas_access": "binary",
    "drainage_access": "binary",

    "belongs_to_native_community": "binary",

    "depression_and_anxiety_symptoms": "binary",

    "has_a_job": "binary",

    "performs_caregiving_duties": "binary",

    "there_were_neighborhood_activities_over_the_last_year": "binary",
    "participated_in_neighborhood_activities": "binary",
    "would_you_like_to_participate_in_neighborhood_activities": "binary",
    "feels_part_of_the_community": "binary",

    "has_smartphone": "binary",

    # =====================
    # ORDINAL
    # =====================

    "age_group": "ordinal",

    "education_level": "ordinal",

    "overcrowding": "ordinal",

    "sports_center_sacurity_level": "ordinal",
    "gyms_security_level": "ordinal",
    "shelter_security_level": "ordinal",

    "water_frequency": "ordinal",
    "water_service_perception": "ordinal",

    "electricity_perception": "ordinal",

    "gas_perception": "ordinal",

    "drainage_failure_frequency": "ordinal",

    "waste_collection_frequency": "ordinal",

    "precipitation_risk": "ordinal",
    "flood_risk": "ordinal",
    "intensity_recurrence_waterlogging": "ordinal",

    "thunderstorm_risk": "ordinal",
    "hail_risk": "ordinal",

    "maximum_temperature_risk": "ordinal",
    "minimum_temperature_risk": "ordinal",

    "snowfall_risk": "ordinal",

    "slope_susceptibility_risk": "ordinal",

    "social_vulnerability": "ordinal",
    "social_vulnerability_to_fracturing": "ordinal",

    "seismic_risk": "ordinal",

    "hazard_risk_index": "ordinal",

    "monthly_income": "ordinal",

    "frequency_of_community_participation": "ordinal",

    # =====================
    # NOMINAL
    # =====================

    "areas_at_risk_for_women": "nominal",
    "areas_at_risk_for_youth": "nominal",
    "indigenous_security_risk_zones": "nominal",

    "dwelling_type": "nominal",
    "tenure_type": "nominal",

    "shelter_type": "nominal",

    "festivities_name": "nominal",
    "festivities_time": "nominal",

    "water_access_method": "nominal",
    "electricity_type": "nominal",
    "gas_connection_type": "nominal",

    "main_source_of_employment_and_geological_vulnerability": "nominal",
    "main_source_of_employment_and_hydrometeorological_vulnerability": "nominal",

    "sex": "nominal",
    "occupation_category": "nominal",
    "marital_status": "nominal",

    "which_native_language": "nominal",

    "type_of_disability": "nominal",

    "source_income": "nominal",

    "public_transport_used": "nominal",

    "reason_for_not_studying": "nominal",
    "other_reason_not_studying": "nominal"
}
###################################################################
def continuous_stats(x):

    valid = x[~np.isnan(x)]

    return {

        "Type": "Continuous",

        "N": len(valid),

        "Mean": round(np.mean(valid),2),

        "Median": round(np.median(valid),2),

        "Min": round(np.min(valid),2),

        "Max": round(np.max(valid),2),

        "Std": round(np.std(valid),2),

        "P25": round(np.percentile(valid,25),2),

        "P75": round(np.percentile(valid,75),2),

        "IQR": round(
            np.percentile(valid,75)
            -
            np.percentile(valid,25),
            2
        )
    }

###################################################################
def binary_stats(x):

    valid = x[~np.isnan(x)]

    yes = np.sum(valid == 1)

    no = np.sum(valid == 0)

    total = len(valid)

    return {

        "Type": "Binary",

        "N": total,

        "Yes_%": round(
            100 * yes / total,
            2
        ),

        "No_%": round(
            100 * no / total,
            2
        )
    }

#########################################################################
def ordinal_stats(x):

    valid = x[~np.isnan(x)]

    return {

        "Type": "Ordinal",

        "N": len(valid),

        "Median": round(
            np.median(valid),
            2
        ),

        "Mode": int(
            mode(
                valid,
                keepdims=False
            ).mode
        ),

        "P25": round(
            np.percentile(valid,25),
            2
        ),

        "P75": round(
            np.percentile(valid,75),
            2
        ),

        "IQR": round(
            np.percentile(valid,75)
            -
            np.percentile(valid,25),
            2
        )
    }

###################################################################
def nominal_stats(x):

    valid = x[~np.isnan(x)]

    vals, counts = np.unique(
        valid,
        return_counts=True
    )

    idx = np.argmax(counts)

    return {

        "Type": "Nominal",

        "N": len(valid),

        "Categories": len(vals),

        "Mode": vals[idx],

        "Mode_%": round(
            100 *
            counts[idx] /
            counts.sum(),
            2
        ),

        "Entropy": round(
            entropy(counts),
            3
        )
    }
################################################################



######################################################################
continuous_results = []
binary_results = []
ordinal_results = []
nominal_results = []

for dataset_name, dataset_info in datasets.items():

    data = dataset_info["data"]
    variables = dataset_info["variables"]

    for col_idx, variable in enumerate(variables):

        x = data[:, col_idx].astype(float)

        vartype = variable_types.get(
            variable,
            "nominal"
        )

        if vartype == "continuous":

            stats = continuous_stats(x)

            stats["Dataset"] = dataset_name
            stats["Variable"] = variable

            continuous_results.append(stats)

        elif vartype == "binary":

            stats = binary_stats(x)

            stats["Dataset"] = dataset_name
            stats["Variable"] = variable

            binary_results.append(stats)

        elif vartype == "ordinal":

            stats = ordinal_stats(x)

            stats["Dataset"] = dataset_name
            stats["Variable"] = variable

            ordinal_results.append(stats)

        else:

            stats = nominal_stats(x)

            stats["Dataset"] = dataset_name
            stats["Variable"] = variable

            nominal_results.append(stats)

#################################################################
#Tablas de markdown

continuous_df = pd.DataFrame(
    continuous_results
)

print(
    continuous_df.to_markdown(
        index=False
    )
)

binary_df = pd.DataFrame(
    binary_results
)

print(
    binary_df.to_markdown(
        index=False
    )
)

ordinal_df = pd.DataFrame(
    ordinal_results
)

print(
    ordinal_df.to_markdown(
        index=False
    )
)

nominal_df = pd.DataFrame(
    nominal_results
)

print(
    nominal_df.to_markdown(
        index=False
    )
)


