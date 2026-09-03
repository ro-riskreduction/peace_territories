import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import csv
from sklearn import preprocessing
import seaborn as sns
from adjustText import adjust_text

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
    "cantidad_dormitorios": "string",   # ⚠ ###
    "cantidad_menores_trabajando": "string",
    "cantidad_mayores65_trabajando": "string"
}

df_home = pd.read_csv(
    "df_vector_hogar_160726_publica.csv",
    #sep= ";",
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
    df_home["cantidad_dormitorios"]
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
        "14": 14,
        "15": 15,
        "16": 16,
        "17": 17,
        "18": 18,
        "19": 19,
        "20": 20,
        "21": 21,
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
    	"NO": 0
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
    	"NO": 0
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
        "NO": 0
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
        "NO": 0
    })
    .to_numpy()
)

botones_panico_usado_array = (
    df_home["botones_panico_usado"]
    .map({
        "SI": 1,
        "NO": 0
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
        "NO": 0
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
        "NO": 0
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
        "NO": 0
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
        "NO": 0
    })
 .to_numpy()
 )

indigena_seguridad_violencia_array = (
 df_home["indigena_seguridad_violencia"]
 .map({
        "SI": 1,
        "NO": 0
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
        "NO": 0
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
        "NO": 0
    })
 .to_numpy()
 )

jefa_de_hogar_array = (
 df_home["jefa_de_hogar"]
 .map({
        "SI": 1,
        "NO": 0
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
        "NO": 0
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
    	"NO": 0 
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
    	"NO": 0
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
    	"NO": 0    	
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
        "0-5 primera infancia": 1,
        "12-17 adolescencia": 2,
        "18-29 juventud": 3,
        "30-59 adultez": 4,
        "6-11 infancia": 5,
        "60 y más": 6
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
        "NO": 0
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
        "Sí": 1,
        "No": 0
    })
    .to_numpy()
)

sintomas_depresion_ansiedad_array = (    df_ind["sintomas_depresion_ansiedad"]
    .map({
       "si": 1,
       "no": 0
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
        "NO": 0
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
        "SI": 1,
        "NO": 0
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
        "NO": 0 
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
        "NO": 0
    })
    .to_numpy()
)

participo_act_vecinales_array = (
    df_ind["participo_act_vecinales"]
    .map({
        "SI": 1,
        "NO": 0
    })
    .to_numpy()
)

le_gustaria_participar_en_act_vecinales_array = (
    df_ind["le_gustaria_participar_en_act_vecinales"]
    .map({
        "SI": 1,
        "NO": 0
    })
    .to_numpy()
)

se_siente_parte_comunidad_array = (
    df_ind["se_siente_parte_comunidad"]
    .map({
        "SI": 1,
        "NO": 0,
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

#######################################################################################################################################################



risk_cols = [
    "PFE_vulnerabilidad_geologica",
    "PFE_vulnerabilidad_hidrometeorologica",
    "riesgo_precipitacion",
    "riesgo_Inundacion",
    "riesgo_tormenta_electrica",
    "riesgo_granizo",
    "riesgo_tem_max",
    "riesgo_tem_min",
    "riesgo_nevada",
    "riesgo_susceptibilidad_ladera",
    "vul_social",
    "vulnerabilidad_social_FR",
    "riesgo_sismo",
    "riesgo_integrado"
]

categories = [
    "Muy Bajo",
    "Bajo",
    "Medio",
    "Alto",
    "Muy Alto"
]




######################################################################################################################################
#Harmonized version for visualization

risk_df = df_home[risk_cols].copy()

risk_df["PFE_vulnerabilidad_geologica"] = (
    risk_df["PFE_vulnerabilidad_geologica"]
    .replace({
        "Baja": "Bajo",
        "Media": "Medio",
        "Alta": "Alto",
        "No determinado": "N/D"
    })
)

risk_df["PFE_vulnerabilidad_hidrometeorologica"] = (
    risk_df["PFE_vulnerabilidad_hidrometeorologica"]
    .replace({
        "Baja": "Bajo",
        "Media": "Medio",
        "Alta": "Alto",
        "No determinado": "N/D"
    })
)

risk_df = risk_df.replace({
    "N/D": np.nan,
    "No determinado": np.nan
})

#####################################################################################################
risk_levels = {
    "Muy Bajo": 1,
    "Bajo": 2,
    "Medio": 3,
    "Alto": 4,
    "Muy Alto": 5
}

risk_scores = []

for col in risk_cols:

    vc = (
        df_home[col]
        .replace({
            "Baja": "Bajo",
            "Media": "Medio",
            "Alta": "Alto",
            "N/D": np.nan,
            "No determinado": np.nan
        })
        .value_counts(normalize=True)
    )

    score = 0

    for category, weight in risk_levels.items():

        score += vc.get(category, 0) * weight

    risk_scores.append({
        "Variable": col,
        "Weighted Risk Score": score
    })

risk_score_df = pd.DataFrame(risk_scores)

risk_score_df = risk_score_df.sort_values(
    "Weighted Risk Score",
    ascending=False
)



######################################################################################################
high_risk_results = []

for col in risk_cols:

    vc = (
        df_home[col]
        .replace({
            "Baja": "Bajo",
            "Media": "Medio",
            "Alta": "Alto",
            "N/D": np.nan,
            "No determinado": np.nan
        })
        .value_counts(normalize=True)
        * 100
    )

    high_risk = (
        vc.get("Alto",0)
        + vc.get("Muy Alto",0)
    )

    high_risk_results.append({
        "Variable": col,
        "High Risk (%)": high_risk
    })

high_risk_df = pd.DataFrame(
    high_risk_results
)

high_risk_df = (
    high_risk_df
    .sort_values(
        "High Risk (%)",
        ascending=False
    )
)
#################################################################################################################
#Severity and prevalence matrix using weighted risk score
risk_matrix = (
    risk_score_df
    .merge(
        high_risk_df,
        on="Variable"
    )
)




###########################################################################################################################



# Short labels for the plot
risk_labels = {
    "PFE_vulnerabilidad_geologica": "Geological",
    "PFE_vulnerabilidad_hidrometeorologica": "Hydromet",
    "riesgo_precipitacion": "Precipitation",
    "riesgo_Inundacion": "Flood",
    "riesgo_tormenta_electrica": "Storm",
    "riesgo_granizo": "Hail",
    "riesgo_tem_max": "Heat",
    "riesgo_tem_min": "Cold",
    "riesgo_nevada": "Snow",
    "riesgo_susceptibilidad_ladera": "Slope",
    "vul_social": "Social",
    "vulnerabilidad_social_FR": "Social FR",
    "riesgo_sismo": "Earthquake",
    "riesgo_integrado": "Integrated"
}

# Calculate quadrant reference lines
x_mean = risk_matrix["High Risk (%)"].mean()
y_mean = risk_matrix["Weighted Risk Score"].mean()

plt.figure(figsize=(12, 8))

# Scatter
plt.scatter(
    risk_matrix["High Risk (%)"],
    risk_matrix["Weighted Risk Score"],
    s=150,
    color="darkred",
    edgecolor="black",
    alpha=0.8
)

# Labels
texts = []

for _, row in risk_matrix.iterrows():

    texts.append(
        plt.text(
            row["High Risk (%)"],
            row["Weighted Risk Score"],
            risk_labels.get(
                row["Variable"],
                row["Variable"]
            ),
            fontsize=10
        )
    )

# Automatically move labels
adjust_text(
    texts,
    arrowprops=dict(
        arrowstyle="-",
        color="gray",
        lw=0.8,
        alpha=0.6
    )
)

# Quadrants
plt.axvline(
    x_mean,
    color="black",
    linestyle="--",
    alpha=0.6
)

plt.axhline(
    y_mean,
    color="black",
    linestyle="--",
    alpha=0.6
)

# Quadrant labels

plt.text(
    x_mean + 2,
    y_mean + 0.45,
    "Priority Risks",
    fontsize=15,
    fontweight="bold",
    color="darkred"
)

plt.text(
    x_mean - 15,
    y_mean + 0.45,
    "Severe but\nLess Common",
    fontsize=13,
    fontweight="bold",
    color="royalblue"
)

plt.text(
    x_mean + 2,
    y_mean - 0.45,
    "Common but\nModerate",
    fontsize=13,
    fontweight="bold",
    color="royalblue"
)

plt.text(
    x_mean - 15,
    y_mean - 0.45,
    "Lower\nPriority",
    fontsize=13,
    fontweight="bold",
    color="royalblue"
)

label_dict = {
    "Geological": "PFE_vulnerabilidad_geologica",
    "Hydromet": "PFE_vulnerabilidad_hidrometeorologica",
    "Precipitation": "riesgo_precipitacion",
    "Flood": "riesgo_Inundacion",
    "Storm": "riesgo_tormenta_electrica",
    "Hail": "riesgo_granizo",
    "Heat": "riesgo_tem_max",
    "Cold": "riesgo_tem_min",
    "Snow": "riesgo_nevada",
    "Slope": "riesgo_susceptibilidad_ladera",
    "Social": "vul_social",
    "Social FR": "vulnerabilidad_social_FR",
    "Earthquake": "riesgo_sismo",
    "Integrated": "riesgo_integrado"
}

legend_text = "\n".join(
    [f"{k}: {v}" for k, v in label_dict.items()]
)

plt.figtext(
    1.02,        # move outside figure
    0.5,
    legend_text,
    fontsize=8,
    va="center",
    ha="left",
    bbox=dict(
        facecolor="white",
        edgecolor="gray",
        alpha=0.9
    )
)

plt.tight_layout(rect=[0, 0, 0.85, 1])

plt.axvline(
    x_mean,
    color="navy",
    linestyle="--",
    linewidth=1.5,
    alpha=0.5
)

plt.axhline(
    y_mean,
    color="navy",
    linestyle="--",
    linewidth=1.5,
    alpha=0.5
)

plt.xlabel(
    "% Households in High or Very High Risk",
    fontsize=12
)

plt.ylabel(
    "Weighted Risk Score",
    fontsize=12
)

plt.title(
    "Risk Severity–Prevalence Matrix",
    fontsize=15,
    fontweight="bold"
)

plt.grid(alpha=0.25)

plt.tight_layout()
plt.show()

#########################################################################################################

