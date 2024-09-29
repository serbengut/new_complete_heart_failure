import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
#from streamlit_echarts import st_echarts
#from codigo_ejecucion_heart_failure_menos_variables_graficos import *
import numpy as np
#import cloudpickle
#import pickle
from janitor import clean_names
from sklearn.metrics import confusion_matrix
#from sklearn.preprocessing import OrdinalEncoder
#from sklearn.preprocessing import KBinsDiscretizer
#from sklearn.preprocessing import Binarizer
#from sklearn.preprocessing import QuantileTransformer
#from sklearn.preprocessing import StandardScaler
####################################################
#  me funciona en local, pero no en la web, meto esto:
#from category_encoders import TargetEncoder
#from sklearn.preprocessing import TargetEncoder
#####################################################
#####################################################
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
#from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
import scikitplot as skplt
#from yellowbrick.classifier import discrimination_threshold
#####################################################
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.ensemble import HistGradientBoostingClassifier
#from xgboost import XGBClassifier
#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import FunctionTransformer
#from sklearn.compose import make_column_transformer
#from sklearn.pipeline import make_pipeline
#Desactivar los warnings
import warnings
warnings.filterwarnings("ignore")
############################################################################################


########################################################################################################################################
########################################### INICIO CARGA FUNCIONES #####################################################################
########################################################################################################################################

def cargar_dataset():
   
    dataset = pd.read_csv('heart.csv')

    return dataset

########################################################################################

def cargar_dataset_completo():
   
    dataset_completo = pd.read_csv('dataframe_completo',sep=';',decimal='.')

    return dataset_completo

########################################################################################

def cargar_predict_proba():
   
    predict_proba = np.loadtxt('mod_best_estimator_pred_proba.txt')

    return predict_proba

########################################################################################
def carga_x_y():
    #pred = np.loadtxt('pred_final.txt')
    scoring = np.loadtxt('scoring_modelo.txt')

    #val_y_final = np.loadtxt('val_y_final_.txt')
    heart_dis_realidad = np.loadtxt('heart_dis_realidad.txt')
    return scoring, heart_dis_realidad

########################################################################################
def max_roi_5(real,scoring, salida = 'grafico'):

    #DEFINIMOS LA FUNCION DEL VALOR ESPERADO
    def valor_esperado(matriz_conf):
        TN, FP, FN, TP = conf.ravel()
        VE = (TN * ITN_usu) + (FP * IFP_usu) + (FN * IFN_usu) + (TP * ITP_usu)
        return(VE)
    
    #CREAMOS UNA LISTA PARA EL VALOR ESPERADO
    ve_list = []
    
    #ITERAMOS CADA PUNTO DE CORTE Y RECOGEMOS SU VE
    for umbral in np.arange(0,1,0.05):
        predicho = np.where(scoring > umbral,1,0) 
        conf = confusion_matrix(real,predicho)
        ve_temp = valor_esperado(conf)
        ve_list.append(tuple([umbral,ve_temp]))
        
    #DEVUELVE EL RESULTADO COMO GRAFICO O COMO EL UMBRAL ÓPTIMO
    df_temp = pd.DataFrame(ve_list, columns = ['umbral', 'valor_esperado'])
    if salida == 'grafico':
    #    fig, ax =plt.subplots()
    #    ax.plot(umbral, valor_esperado, data = df_temp)
    #    st.pyplot(fig)
         solo_ve_positivo = df_temp[df_temp.valor_esperado > 0]
         plt.figure(figsize = (12,6))
         sns.lineplot(data = solo_ve_positivo, x = 'umbral', y = 'valor_esperado')
         plt.xticks(solo_ve_positivo.umbral, fontsize = 9, rotation=90)
         plt.yticks(solo_ve_positivo.valor_esperado, fontsize = 6);
        
#        plt.figure(figsize = (12,6))
#        sns.lineplot(data = df_temp, x = 'umbral', y = 'valor_esperado')
#        plt.xticks(df_temp.umbral, fontsize = 9, rotation=90)
#        plt.yticks(df_temp.valor_esperado, fontsize = 9)
#        plt.show();
    else:    
        return(df_temp.iloc[df_temp.valor_esperado.idxmax(),0])

############################################################################################################################################
########################################### FIN CARGA FUNCIONES ############################################################################
############################################################################################################################################





############################################################################################################################################
############################################    INICIO APP    ##############################################################################
############################################################################################################################################

st.set_page_config(
     page_title = 'Heart Failure (Insuficiencia Cardiaca)',
     page_icon = 'heart.jpg',
     layout = 'wide')

st.title('Heart Failure (Insuficiencia Cardiaca)')

with st.expander("Explicación del Proyecto y de la Aplicación:"):
        st.write('Se ha desarrollado un modelo de **Machine Learning** basado en diversos datos de pacientes, los cuales acabaron teniendo o no **Insufiencia Cardiaca**.')
        st.write('Estos datos han pasado por una serie de **trabajos previos**, como han sido:')
        st.markdown('- **Calidad de datos**: Analisis y gestión de **datos nulos** y **atípicos**.')
        st.markdown('- **Analisis exploratorio de datos** (EDA) para encontrar patrones y anomalías.')
        st.markdown('- **Transformación de variables**, tanto categóricas como numéricas. En este caso concreto, se ha realizado un **Target Encoding** a las variables categóricas. En cuanto a las variables numéricas, se han normalizado con **Quantile Transformer**.')
        st.markdown('- **Reescalado de variables**: para este proyecto se ha realizado el escalado con Min-Max Scaler.')
        st.markdown('- **Preselección de variables**: se han aplicado diversos métodos para encontrar qué variables eran más predictoras. En este caso, se ha decidido usar las variables que eran más predictoras de acuerdo al método **Recursive Feature Elimination**.')
        st.markdown('- Estudio de **Correlación** entre variables seleccionadas. Se ha analizado la correlación existente y se ha visto que no existe una correlación alta entre ninguna variable, por lo que no se ha tenido que prescindir de ninguna.')
        st.markdown('- **Balanceo**: el conjunto de datos con el que contamos no está al 100% balanceado, como por otro lado es normal en este tipo de proyectos. A pesar de que en este caso el número de casos positivos es mayor que el de negativos, se ha estudiado si esta diferencia puede provocar problemas al modelo. Se ha aplicado una regresión logística sobre los datos sin balancear para que nos sirva de base de comparación cuando apliquemos el mismo modelo al conjunto de datos que se ha balanceado con Undersampling, Oversampling y SMOTE-Tomek. Se ha observado que el scoring obtenido por los tres casos con el dataset balanceado apenas mejora el scoring con el dataset original. Por tanto, no ha hecho falta hacer balanceo.')
        st.markdown('- **Modelización**: se ha realizado una Validación Cruzada, reservando al principio del proyecto un 30% de los datos como Validación. En el momento de modelizar, se han separado los datos restantes en "train" y "test". Posteriormente se ha creado un pipe con varios algoritmos y varios parámetros para, con Randon Search, escoger el mejor modelo. Tras realizar esta operación, el mejor modelo ha sido un XGBClassifier. Cabría recordar que el Random Search también nos ha dado los párametros óptimos para este modelo. Tras esto, se ha procedido a validar con el dataset de "test" y con el dataset de "validación" reservado al principio del proyecto. Se ha visto que el modelo no está sobreajustado y se ha procedido a aceptarse como modelo válido.')
        st.markdown('- Creación del **Pipeline**: finalmente, se han metido en un Pipeline todos los pasos llevados a cabo hasta ahora. Se han creado funciones que recogen la calidad de datos realizada, se han metido también las transformaciones realizadas y, por último, se ha incluido el modelo elegido. Este Pipe se ha guardado y es el que se está usando en esta aplicación.')


st.divider()

st.subheader('ENFOQUE PÚRAMENTE :orange[ECONÓMICO]:')

col1, col2 =st.columns(2)
col1.write('**Esta aplicación nos indica las :orange[probabilidades] que tiene un paciente de sufrir :orange[Insuficiencia Cardiaca] de acuerdo al modelo de machine learning desarrollado.**')
col1.write('**Además, nos permite fijar el :orange[Coste del Tratamiento Preventivo] y el :orange[Coste del Tratamiento Correctivo]**.')

#col3, col4 =st.columns(2)
col2.write('**El modelo nos da también el :orange[porcentaje umbral] a partir del cual es más rentable desde el punto de vista económico aplicar el Tratamiento Preventivo.**')
col2.write('**A su vez, nos da el :orange[retorno de la inversión], esto es, la cantidad de dinero ahorrado por paciente al aplicar esta política.**')

st.divider()

st.subheader('ENFOQUE PÚRAMENTE :green[MÉDICO]:')

col3, col4 =st.columns(2)
col3.write('**Desde el punto de vista Médico, se busca encontrar al :green[máximo numero de pacientes] que acabarán teniendo Insuficiencia Cardiaca para aplicar el :green[Tratamiento Preventivo] sobre ellos.**')
col3.write('**Para ello, debemos aumentar en la medida de lo posible el :green[RECALL]. Es decir, atendiendo a la :green[matriz de confusión], debemos aumentar todo lo posible los :green[Verdaderos Positivos] y reducir lo máximo posible los :green[Falsos Negativos].**')
col4.write('****')
col4.write('****')

st.divider()

st.subheader("Introducción de :blue[datos del paciente]:")#, divider='grey')  #Please choose from: blue, green, orange, red, violet, gray, grey, rainbow


col5, col6 = st.columns(2)
chest_pain_type = col5.selectbox('Chest Pain Type: Tipo de dolor de pecho (ATA: Angina Atípica; NAP: Dolor no anginal  ; ASY: Asintomático  ; TA: Angina Típica)', ['ATA', 'NAP', 'ASY','TA'])
oldpeak = col6.number_input(label='OldPeak: depresión del ST inducida por el ejercicio relativo al descanso', min_value=-3.0, max_value=6.0,step=0.1,format="%.2f")

col7, col8 = st.columns(2)
sex = col7.selectbox('Sexo', ['M', 'F'])
st_slope = col8.selectbox('ST Slope: pendiente del segmento ST del ejercicio máximo: (Up: Pendiente ascendente; Flat: plano; Down: Pendiente descendente)', ['Up', 'Flat','Down']) 


registro = pd.DataFrame(
                        {'chest_pain_type':chest_pain_type,                                           
                         'sex':sex,
                         'oldpeak':oldpeak,
                         'st_slope':st_slope
                         }
                        ,index=[0])
datos_paciente = pd.DataFrame(
                        {'Tipo dolor de pecho':chest_pain_type,                                             
                         'Depresión del ST':oldpeak,
                         'Sexo':sex,
                         'Pendiente del segmento ST':st_slope
                         }
                        ,index=[0])

st.subheader('Estos son los :blue[datos introducidos] del paciente:')#, divider='grey')
datos_paciente

#st.divider()
st.subheader('Introduzca los costes del :blue[tto. preventivo] y del :blue[tto. correctivo] (a largo plazo):')#,divider='grey')
col9, col10 = st.columns(2)
coste_tto_preventivo = col9.slider('Coste **TTO. PREVENTIVO**', 500, 1500)
coste_tto = col10.slider('Coste tratamiento paciente enfermo a largo plazo (**TTO. CORRECTIVO**): ', 1600, 28000)

st.divider()

ITN_usu = 0
IFP_usu = -coste_tto_preventivo
IFN_usu = -coste_tto + coste_tto_preventivo
ITP_usu = +coste_tto - coste_tto_preventivo

########################################################################################################################################
########################################################################################################################################
################################## A PARTIR DE AQUI TRATO DE NO USAR EL PIPE EJECUCION #################################################
########################################################################################################################################
########################################################################################################################################



# modelo = XGBClassifier(n_jobs = -1, 
#                        verbosity = 0,
#                        learning_rate = 0.025,
#                        max_depth = 10,
#                        reg_alpha = 0,
#                        reg_lambda = 1,
#                        n_estimators = 100,
#                        use_label_encoder=False
#                        )

dataset = cargar_dataset()
dataset = clean_names(dataset)
dataset.rename(columns = {'chestpaintype':'chest_pain_type',
                     'restingbp':'resting_bp',
                     'fastingbs':'fasting_bs',
                    'restingecg':'resting_ecg',
                    'maxhr':'max_hr',
                    'exerciseangina':'exercise_angina',
                    'heartdisease':'heart_disease'}, inplace=True)

variables_finales = [
                     'chest_pain_type',
                     'sex',
                     'oldpeak',
                     'st_slope'
                    ]

dataset = dataset[variables_finales].copy()

scoring, heart_dis_realidad = carga_x_y()
dataset_completo = cargar_dataset_completo()
#scoring
#heart_dis_realidad
#dataset_completo

##### boton calcular prob fallo cardiaco: ##########################
if st.sidebar.button('CALCULAR POSIBILIDAD DE FALLO CARDIACO y MAX ROI'):
#     fallo = ejecutar_modelo(registro)
      
    elegido = dataset_completo [ ( dataset_completo['chest_pain_type'] == chest_pain_type ) & ( dataset_completo['sex'] == sex) & ( dataset_completo['oldpeak'] == oldpeak ) & ( dataset_completo['st_slope'] == st_slope ) ]
    #elegido
    fallo = elegido.iloc[0,4]


    st.subheader(f'Probabilidad de sufrir Insuficiencia Cardiaca: :blue[{round(100*fallo,2)}%]')
    st.divider()
    st.subheader('ENFOQUE PÚRAMENTE :orange[ECONÓMICO]:')

    umbral_usu = max_roi_5(heart_dis_realidad, scoring,salida = 'automatico')
    ##### para meter grafico (abajo) #####    
    def valor_esperado(matriz_conf):
        TN, FP, FN, TP = conf.ravel()
        VE = (TN * ITN_usu) + (FP * IFP_usu) + (FN * IFN_usu) + (TP * ITP_usu)
        return(VE)
    
    ve_list = []
    for umbral in np.arange(0,1,0.025):
        predicho = np.where(scoring > umbral,1,0) 
        conf = confusion_matrix(heart_dis_realidad,predicho)
        ve_temp = valor_esperado(conf)
        ve_list.append(tuple([umbral,ve_temp]))

    st.subheader(f'El umbral para decidir si aplicar o no tratamiento preventivo de acuerdo a los costes introducidos es: :orange[{100*(round(umbral_usu,2))}%]')

    ###col11.subheader(f'Como la probabilidad de que el paciente tenga fallo cardiaco es: :red[{round(100*fallo,2)}%]')

    st.write('**Desde el punto de vista del coste del tto. preventivo versus coste del tto. normal:**')
    if fallo > umbral_usu:
        st.subheader('Dado que la probabilidad de sufrir Insuficiencia cardiaca es :orange[MAYOR] que el umbral:')
        st.write('**Recomendación del modelo:**')
        st.subheader('PACIENTE :orange[ELEGIBLE] PARA HACER TRATAMIENTO PREVENTIVO')
    else:
        st.subheader('Dado que la probabilidad de sufrir Insuficiencia cardiaca es :orange[MENOR] que el umbral:')
        st.write('**Recomendación del modelo:**')
        st.subheader('PACIENTE :orange[NO ELEGIBLE] PARA HACER TRATAMIENTO PREVENTIVO')

    

#     #col11.subheader(f'El ahorro económico por paciente sería de: :blue[{round(ganancia_por_paciente,2)}€]')
    
    col11, col12 = st.columns(2)

 ################## grafico de ganancias ##########################################################

    col11.write('**Gráfico del Retorno de la Inversión:**')

    #DEVUELVE EL RESULTADO COMO GRAFICO O COMO EL UMBRAL ÓPTIMO
    df_temp = pd.DataFrame(ve_list, columns = ['umbral', 'valor_esperado'])
    
    total_pacientes = len(scoring)
    df_temp['Ganancia por paciente']=df_temp['valor_esperado'].apply(lambda x: x/total_pacientes)
    ganancia_por_paciente=df_temp['Ganancia por paciente'].max()

    fig, ax = plt.subplots( figsize=(6,4))
    ax.plot('umbral', 'Ganancia por paciente', data = df_temp, color='orange')
    plt.xticks(df_temp['umbral'], fontsize = 7, rotation=90)
    plt.ylim(df_temp['Ganancia por paciente'].min()*1.10,df_temp['Ganancia por paciente'].max()*1.20) 
    plt.yticks(fontsize = 8)
    plt.xlabel('Umbral',fontsize = 7)
    plt.ylabel('Ahorro por paciente (€)',fontsize = 7)
    #plt.yticks([df_temp['Ganancia por paciente'].min(),df_temp['Ganancia por paciente'].max()], fontsize = 5)
    #plt.yticks(df_temp['Ganancia por paciente'], fontsize = 5)
    col11.pyplot(fig)
    col11.subheader(f'El ahorro económico por paciente sería de: :orange[{round(ganancia_por_paciente,2)}€]')

    with col11.expander("Explicación del gráfico **Retorno de la Inversión**"):
        st.write('La gráfica de arriba muestra en el **eje x** los distintos umbrales (de 0 a 100%). El **eje y** representa el ahorro o coste (si el valor es negativo) para cada umbral. El umbral varía con la introducción de los costes preventivo y correctivo.')
        st.write(f'En este caso, el umbral es del :orange[{100*(round(umbral_usu,2))}%] que es el punto más alto de la gráfica. En ese punto más alto, el **eje y** muestra el ahorro (o coste) por paciente con las mismas características si seguimos la recomendación que nos da el modelo. En nuestro caso sería: :orange[{round(ganancia_por_paciente,2)}€.]')     
        st.write('Cuanto mayor sea la diferencia entre los costes del Tto. Preventivo y del Tto. Correctivo más disminuirá el umbral, significando que si los costes son muy dispares, se tenderá a aplicar el Tto. Preventivo a todos los pacientes, por muy baja probabilidad que tengan de sufrir la dolencia estudiada.')
        st.write('Por el contrario, si el coste del Tto. Preventivo es muy similar al coste del Tto. Correctivo aumentará el umbral enórmemente, con lo que se tenderá a no aplicar a ningún paciente el Tto. Preventivo a no ser que la probabilidad de sufrir la dolencia sea altísima (por encima del umbral). ')
# ################## grafico de ganancias ##########################################################


    #ganancia_por_paciente=df_temp['Ganancia por paciente'].max()

###### para meter grafico (arriba) #####

#    grafico = max_roi_5(heart_dis_realidad, scoring,salida = 'grafico')
#    st.pyplot(grafico)



################## gain chart ##########################################################
    predict_proba=cargar_predict_proba()    
    col12.write('**Gráfico de Ganancias:**')
    fig2, ax = plt.subplots()
    #modelo = cargar_modelo()
    #skplt.metrics.plot_cumulative_gain(scoring, modelo.best_estimator_.predict_proba(scoring), ax=ax) 
    #skplt.metrics.plot_cumulative_gain(heart_dis_realidad, modelo.predict_proba(dataset), ax=ax, title = 'Gráfico de Ganancias Acumuladas') 
    skplt.metrics.plot_cumulative_gain(heart_dis_realidad, predict_proba, ax=ax, title = 'Gráfico de Ganancias Acumuladas')
    #Eliminamos la línea de los ceros y personalizamos la leyenda
    ax.lines[0].remove()                 
    plt.legend(labels = ['Modelo','Aleatoria'])
    col12.pyplot(fig2);

    with col12.expander("Explicación del gráfico **Cumulative Gains Curve** o **Gráfico de Ganancias**"):
        st.write('En el gráfico se muestran dos líneas: la :orange[**Aleatoria**], la cual representa decidir al azar si un paciente puede o no sufrir una Insuficiencia Cardiaca. Por otro lado, la línea del :orange[**Modelo**] representa los aciertos con respecto al porcentaje de datos clasificados para el modelo.')
        st.write('De acuerdo al modelo, la gráfica de ganancia muestra un fuerte aumento por encima de la línea aleatoria y, luego, un aplanamiento.')
        st.write('Cuanto mayor sea la ganancia, mejor será el modelo.')
        st.write('En este caso, aproximadamente el :orange[40%] de los datos representan aproximadamente el :orange[66%] de los verdaderos positivos. Por lo tanto, si nos enfocáramos, por ejemplo, en el :orange[30%] de la población guiada por el modelo, el porcentaje de la tasa de verdaderos positivos sería aproximadamente :orange[50%].')
        st.write('Sin el modelo, el porcentaje correspondiente sería :orange[30%] (línea "Aleatoria"). Esta diferencia es la ganancia adicional que se obtiene al utilizar el modelo')
                 #El **eje x*/ representa el percentil de la fracción de datos acumulada. Por ejemplo, un porcentaje de 0.4 (40%) de datos clasificados equivale a ')     

# ##################### gain chart #########################################################        
    st.divider()
    st.subheader('ENFOQUE PÚRAMENTE :green[MÉDICO]:')

else:
    st.write('DEFINE LOS PARÁMETROS DEL PACIENTE, LOS COSTES DE LOS TRATAMIENTOS Y HAZ CLICK EN CALCULAR POSIBILIDAD DE FALLO CARDIACO')
