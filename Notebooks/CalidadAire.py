# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

# [![UAEM](https://www.uaem.mx/fcaei/images/uaem.png)](https://www.uaem.mx/fcaei/moca.html)
# [![Pandas](https://img.shields.io/badge/Pandas-2.0+-blue.svg)](https://pandas.pydata.org/)
# [![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2+-1560BD.svg)](https://scikit-learn.org/stable/)
# [![Statsmodels](https://img.shields.io/badge/Statsmodels-0.14+-8A2BE2.svg)](https://www.statsmodels.org/stable/index.html)
# [![Python 3.9+](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/downloads/release/python-390/)
# 
# # Proyecto: Análisis Predictivo y de Series de Tiempo de Contaminantes Atmosféricos
# 
# **Autor:** [Emiliano Rodriguez Villegas](https://github.com/EmilianoRdzV)
# **Fecha:** 14 de Junio de 2025
# **Versión:** 1.0
# 
# ---
# 
# ## Introducción y Objetivos
# 
# Este notebook realiza un análisis multifacético sobre un conjunto de datos de seis contaminantes criterio, correspondientes al mes de enero de 2019. El proyecto se centra en dos áreas principales de la ciencia de datos:
# 
# 1.  **Modelado Predictivo:** Se construirán y evaluarán modelos de regresión para estimar la concentración de ciertos contaminantes (como PM2.5 y O3) utilizando otros gases como variables de entrada.
# 2.  **Análisis de Series de Tiempo:** Se analizará el comportamiento temporal de un contaminante específico para descomponer su serie, identificar su tendencia y observar los patrones subyacentes una vez eliminado el componente estacional.
# 
# ---
# 
# ### Índice del Notebook
# 
# 1.  [**Fase 1: Preparación y Análisis Exploratorio**](#fase-1)
#     * [1.1. Carga de Librerías y Datos](#1-1)
#     * [1.2. Limpieza y Verificación de Datos](#1-2)
#     * [1.3. Análisis Exploratorio y Matriz de Correlación](#1-3)
# 2.  [**Fase 2: Modelado Predictivo (Regresión)**](#fase-2)
#     * [2.1. Modelo 1: Predicción de Partículas PM10 / PM2.5](#2-1)
#     * [2.2. Modelo 2: Predicción de Ozono (O3)](#2-2)
#     * [2.3. Evaluación de Modelos (Métricas y Análisis de valor-t)](#2-3)
# 3.  [**Fase 3: Análisis de Series de Tiempo**](#fase-3)
#     * [3.1. Selección y Visualización de la Serie Temporal](#3-1)
#     * [3.2. Descomposición de la Serie](#3-2)
#     * [3.3. Observaciones de la Serie Desestacionalizada](#3-3)
#     * [3.4. Análisis de Patrones por Día de la Semana](#3-4)
#   
# 4.  [**Fase 4: Conclusiones Finales**](#fase-4)
#     * [4.1. Resumen de Hallazgos](#4-1)
#     * [4.2. Pasos Futuros](#4-2)

# ## <a id="fase-1"></a>1. Fase 1: Preparación y Análisis Exploratorio
# 
# En esta fase inicial, cargaremos todas las herramientas necesarias y nuestro conjunto de datos. Realizaremos una limpieza básica y una exploración para entender la estructura y las características principales de los datos antes de pasar al modelado.
# 
# ### <a id="1-1"></a>1.1. Carga de Librerías y Datos
# 
# Comenzamos importando todas las librerías de Python que utilizaremos a lo largo del proyecto. Posteriormente, cargaremos el conjunto de datos desde el archivo local a un DataFrame de `pandas` que llamaremos `contaminantes`.




sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 8)


rutaArchivo = '../Data/contaminantes.xlsx' 

contaminantes = pd.read_excel(rutaArchivo)  
contaminantes['DATE'] = pd.to_datetime(contaminantes['DATE'])
# 'DATE' como el índice
contaminantes.set_index('DATE', inplace=True)
display(contaminantes.head())


# ### <a id="1-2"></a>1.2. Limpieza y Verificación de Datos
# 
# Una vez cargados los datos, el siguiente paso crítico es la limpieza. Verificaremos la existencia de valores nulos (NaN) y filas duplicadas que puedan afectar la calidad de nuestro análisis y modelos.
# 
# **Estrategia de Limpieza:**
# * **Valores Nulos:** Para los valores numéricos faltantes, utilizaremos el método de **interpolación lineal**. Esta técnica es ideal para datos de series temporales (como mediciones de sensores), ya que estima los valores faltantes basándose en los puntos anterior y posterior, asumiendo una progresión lógica.
# * **Duplicados:** Se eliminará cualquier fila que sea una copia exacta de otra para evitar redundancia en los datos.



# Valores nulos/duplicados que puedan interferir en el analisiss 
totalNulos = contaminantes.isnull().sum().sum()
totalDuplicados = contaminantes.duplicated().sum()

print("\n--- Resumen final de los tipos de datos ---")
contaminantes.info()


# ### <a id="1-3"></a>1.3. Análisis Exploratorio y Matriz de Correlación
# 
# Con los datos limpios y verificados, realizamos un último análisis exploratorio. Primero, generaremos estadísticas descriptivas para entender la escala y distribución de cada contaminante.
# 
# Segundo, y más importante, crearemos una **matriz de correlación**. Esta nos mostrará numéricamente qué tan fuerte es la relación lineal entre cada par de contaminantes. La visualizaremos como un mapa de calor (heatmap) para identificar rápidamente las relaciones más significativas, lo cual será crucial para seleccionar las variables de entrada en nuestros modelos de regresión de la Fase 2.



# .describe() nos da un resumen numérico completo de cada columna.

display(contaminantes.describe())


# Si observamos la tabla generada, verremos el contexto cuantitativo a la distribución de cada contaminante. Nos permite entender tres aspectos clave de los datos:
# 
# * **Tendencia Central:** ¿Cuál es el valor promedio o típico de cada contaminante? (ver la fila `mean`).
# * **Dispersión:** ¿Qué tan variables o dispersos son los datos alrededor de ese promedio? (ver la fila `std` o desviación estándar).
# * **Rango:** ¿Cuáles son los valores mínimos y máximos registrados en este periodo? (ver las filas `min` y `max`).
# 
# #### Observaciones Clave de los Datos:
# 
# Observamos lo siguiente:
# 
# * **Consistencia de los Datos:** Todas las columnas tienen una cuenta de **742 entradas**, lo que confirma que tenemos un conjunto de datos completo para el período de tiempo analizado (Enero de 2019).
#   
# * **Diferentes Escalas:** Los contaminantes operan en escalas muy distintas. Por ejemplo, el `CO` se mueve en valores cercanos a 0.8, mientras que el `PM2.5` tiene un promedio de 19.5. Esto es importante tenerlo en cuenta para el futuro modelado.
#   
# * **Variabilidad:** El contaminante **PM2.5** muestra una desviación estándar (`std`) de **8.11**, que es considerable en relación con su media (`mean`) de **19.5**. Esto sugiere que sus niveles son bastante variables. En contraste, el `SO2` es mucho más estable, con una desviación estándar muy baja.
#   
# * **Rango de Medición:** Durante este mes, los niveles de `PM2.5` oscilaron entre un mínimo de 3.0 y un máximo de 47.0.
# 
# Ahora que entendemos las características individuales de cada contaminante, el siguiente paso natural es analizar cómo se relacionan entre sí. Para ello, generaremos la matriz de correlación.



# Matriz de Correlación y Heatmap

# Solo columnas numéricas para el cálculo 
columnasNumericas = ['CO', 'NO2', 'O3', 'SO2', 'PM2.5', 'PM10']
contaminantesNumericos = contaminantes[columnasNumericas]

# Matriz de correlacion
matrizCorrelacion = contaminantesNumericos.corr()

# Heatmap para la representacion de la matriz de correlacion    
plt.figure(figsize=(10, 8))
sns.heatmap(matrizCorrelacion, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación entre Contaminantes', fontsize=16)
plt.savefig('../Images/matrizCorrelacion.png', dpi=300, bbox_inches='tight')
plt.show()


# El mapa de calor anterior es una de las herramientas más importantes de nuestro análisis exploratorio. Su función principal en este proyecto es la de **guiar la selección de características (features)** para los modelos de regresión que construiremos en la Fase 2.
# 
# **¿Por qué?**
# * **Para predecir PM2.5 y PM10:** Podemos observar que `PM2.5` y `PM10` tienen una **correlación positiva muy fuerte (0.81)**. También vemos correlaciones positivas moderadas con `SO2`. Esto nos sugiere que estas variables serán buenos predictores.
# * **Para predecir Ozono (O3):** Notamos una **correlación negativa significativa (-0.46) entre `O3` y `NO2`**.El heatmap la confirma en nuestros datos, señalando al `NO2` como un predictor clave.
# 
# E heatmap valida nuestras hipótesis y nos da una base cuantitativa para elegir las variables de entrada más prometedoras para nuestros modelos.

# ## <a id="fase-2"></a>2. Fase 2: Modelado Predictivo (Regresión)
# 
# Ahora que vimos las relaciones entre nuestros datos, comenzaremos a construir el modelo. Primer bjetivo: predecir la concentración de partículas `PM2.5` y `PM10`.
# 
# ### <a id="2-1"></a>2.1. Modelo 1: Predicción de Partículas PM2.5
# 
# Para ilustrar el concepto, comenzaremos con el caso más simple: una **regresión lineal simple** para predecir la concentración de `PM2.5` usando únicamente la variable con la que tiene mayor correlación: `PM10`. Luego, visualizaremos el resultado.



# Regresión Lineal Simple: PM10 -> PM2.5

# Selección de Características (X) y Objetivo (y)
# Doble corchete en X para mantenerlo como un DataFrame
X = contaminantes[['PM10']] 
y = contaminantes['PM2.5']

# Entrenamiento y Prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelado
modeloLineal = LinearRegression()
modeloLineal.fit(X_train, y_train)

# Predicciones
y_pred = modeloLineal.predict(X_test)

# Evaluación del Modelo
plt.figure(figsize=(10, 6))

# Visualización de los resultados
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Datos Reales')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Línea de Regresión')

plt.title('Regresión Lineal: PM2.5 vs. PM10', fontsize=16)
plt.xlabel('PM10', fontsize=12)
plt.ylabel('PM2.5', fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig('../Images/regresionLineal.png', dpi=300, bbox_inches='tight')
plt.show()


# #### Scikit-Learn
# La imagen a continuación muestra la "firma" del método `.fit()`.
# 
# 
# ![Firma del método .fit() de Scikit-learn](../Images/parametrosRLineal.png)
# 
# Como indica la documentación, el método espera dos argumentos principales:
# * **`X`**: Un arreglo o matriz con las características de entrada (en nuestro caso, `X_train`).
# * **`y`**: Un arreglo con los valores objetivo correspondientes (en nuestro caso, `y_train`).
# 
# **Referencia:** [Documentación Oficial de LinearRegression en scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

# ### <a id="2-2"></a>2.2. Modelo 2: Predicción de Ozono (O3)
# 
# Siguiendo los objetivos del proyecto, ahora construiremos nuestro segundo modelo predictivo. En este caso, utilizaremos una **regresión lineal múltiple** para estimar la concentración de Ozono (`O3`) basándonos en las concentraciones de `NO2` y `CO` como variables de entrada.
# 
# El flujo de trabajo será el mismo: seleccionar datos, dividir en entrenamiento/prueba, entrenar el modelo y, finalmente, evaluar su rendimiento.



# Regresión Lineal Múltiple: (NO2, CO) -> O3 

# Selección de Características (X) y Objetivo (y)
# Usamos una lista de columnas para nuestras características
featuresO3 = ['NO2', 'CO']
targetO3 = 'O3'

X_o3 = contaminantes[featuresO3]
y_o3 = contaminantes[targetO3]

# Entrenamiento y Prueba
X_train_o3, X_test_o3, y_train_o3, y_test_o3 = train_test_split(
    X_o3, y_o3, test_size=0.3, random_state=42)

# Modelado R Multipe
modeloO3 = LinearRegression()
modeloO3.fit(X_train_o3, y_train_o3)


# Predicciones
y_pred_o3 = modeloO3.predict(X_test_o3)

# Métricas

# Error Cuadrático Medio (MSE): El promedio de los errores al cuadrado. *** Más bajo es mejor.
mse = mean_squared_error(y_test_o3, y_pred_o3)

# Coeficiente de Determinación (R²): Qué proporción de la varianza de O3 es explicada por el modelo. *** Más cercano a 1 es representativo.
r2 = r2_score(y_test_o3, y_pred_o3)

print(f"\n--- Métricas de Rendimiento del Modelo ---")
print(f"Error Cuadrático Medio (MSE): {mse:.4f}")
print(f"Coeficiente de Determinación (R²): {r2:.4f}")

# Valores Reales vs. Predicciones
plt.figure(figsize=(10, 6))
plt.scatter(y_test_o3, y_pred_o3, alpha=0.6, edgecolors='k')

# Añadimos una línea de referencia (y=x). Si los puntos caen sobre esta línea, la predicción es perfecta.
plt.plot([y_test_o3.min(), y_test_o3.max()], [y_test_o3.min(), y_test_o3.max()], '--', color='red', linewidth=2)
plt.title('Valores Reales vs. Valores Predichos para O3', fontsize=16)
plt.xlabel('Valores Reales de O3', fontsize=12)
plt.ylabel('Valores Predichos de O3', fontsize=12)
plt.grid(True)
plt.savefig('../Images/regresionMultiple.png', dpi=300, bbox_inches='tight')
plt.show()


# **Referencias de las Métricas:**
# * [Documentación Oficial de `mean_squared_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)
# * [Documentación Oficial de `r2_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)
# 
# ![R2 Scikit-learn](../Images/r2Info.png)

# ### <a id="2-3"></a>2.3. Evaluación de Modelos (Métricas y Análisis de valor-t)
# 
# Profundizaremos en la evaluación de los dos modelos de regresión que hemos construido. Analizaremos no solo las métricas de error, sino también la significancia estadística de nuestras variables predictoras.
# 
# #### Interpretando el Coeficiente de Determinación (R²)
# 
# En nuestro segundo modelo, obtuvimos un valor de **R² cercano a 0.30**. Es común pensar que un R² por debajo de 0.5 es "malo", pero esta interpretación es incorrecta sin considerar el contexto.
# 
# * **En sistemas controlados (como un laboratorio de física):** Se esperan valores de R² muy altos (>0.95), ya que las variables están aisladas.
# * **En sistemas complejos y caóticos (química atmosférica, nuestro xaso):** Es prácticamente imposible capturar todas las variables que influyen en un resultado.
# 
# En nuestro caso, la formación de Ozono (`O3`) es un proceso químico extremadamente complejo que no solo depende del `NO2` y `CO`, sino también de la **luz solar (radiación UV), la temperatura, la humedad y el viento.
# 
# **¿Qué significa realmente nuestro R² de ~0.30?**
# 
# Significa que nuestro modelo, utilizando **únicamente dos variables** (`NO2` y `CO`), ha logrado explicar el **30% de la variabilidad** en la concentración de Ozono. Esto es bastante significativo. Demuestra que `NO2` y `CO` son predictores **relevantes y con una relación medible**, aunque el 70% restante de la variabilidad se deba a otros factores que no hemos incluido en el modelo.
# 
# Por lo tanto, el modelo no es perfecto, pero es **informativo y mucho mejor que el azar**. Ahora, usaremos el análisis de **valor-t** para determinar cuál de nuestras dos variables (`NO2` o `CO`) tiene un impacto más significativo estadísticamente.
# 
# 



# Statsmodels requiere que añadamos una constante (el intercepto) a nuestras variables X.
X_train_o3_sm = sm.add_constant(X_train_o3)

# Modelo OLS, Use OLS para la obtencion de t value (tabla de coeficientes)
modeloSm = sm.OLS(y_train_o3, X_train_o3_sm)


resultados = modeloSm.fit()
resultados.params

print(resultados.summary())


# **Referencias t value:**
# * [Documentación Oficial de `tValue statsmodels`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html#statsmodels.regression.linear_model.OLS)
# 

# #### Interpretación de los Resultados Estadísticos
# 
# La tabla de regresión generada por `statsmodels` nos ofrece las siguientes conclusiones clave sobre nuestro modelo de predicción de Ozono (O3):
# 
# 1.  **Significancia de los Predictores:**
#     * La columna **`P>|t|`** nos muestra la significancia de cada variable. Para `NO2` y `CO`, este valor es **0.000**. Al ser un valor mucho menor que el umbral estándar de 0.05, podemos concluir con una alta confianza que **ambas variables son estadísticamente significativas** y tienen un impacto real en la concentración de O3.
#     * La constante (`const`) tiene un p-value de 0.143, lo que indica que no es significativamente diferente de cero.
# 
# 2.  **Impacto y Relación de cada Contaminante:**
#     * La columna **`coef`** revela la naturaleza de la relación:
#     * **NO2:** Tiene un coeficiente de **-1.8556**. Esto indica una fuerte **relación inversa**: por cada unidad que aumenta el `NO2`, la concentración de `O3` tiende a **disminuir** en 1.86 unidades.
#     * **CO:** Tiene un coeficiente de **0.0562**, mostrando una **relación positiva**: por cada unidad que aumenta el `CO`, el `O3` tiende a **aumentar** en 0.056 unidades.
# 
# 3.  **Rendimiento General del Modelo:**
#     * El **`R-squared` de 0.314** confirma que nuestro modelo, con `NO2` y `CO` como predictores, logra explicar el **31.4%** de la variabilidad en los niveles de Ozono.
# 
# 

# ## <a id="fase-3"></a>3. Fase 3: Análisis de Series de Tiempo
# 
# Ahora, nos enfocaremos en el comportamiento de un contaminante a lo largo del tiempo. El objetivo es descomponer su serie para identificar sus componentes principales: tendencia (movimiento a largo plazo), estacionalidad (patrones) y el residuo (aleatoridad).
# 
# ### <a id="3-1"></a>3.1. Selección y Visualización de la Serie Temporal
# Seleeccione **PM2.5** para el desarrollo de este punto dado por el Doc. Pedro Moreno: **"Una vez seleccionada, realizas la serie de tiempo con los datos originales y después de desestacionizarlos, qué observas?"**
# 
# 
# El **PM2.5** por ser uno de los contaminantes más importantes para la salud pública. Crearemos un gráfico de línea para ver cómo cambió su concentración durante el periodo analizado 



# Gráfico de línea para PM2.5

ejeX_numerico = np.arange(len(contaminantes))
ejeY_pm25 = contaminantes['PM2.5'].values


plt.figure(figsize=(17, 7))
plt.plot(ejeX_numerico, ejeY_pm25)

plt.title('Concentración Horaria de PM2.5 (Enero 2019)', fontsize=16)
plt.xlabel('Índice de Muestra Horaria (0 a 741)', fontsize=12)
plt.ylabel('Concentración de PM2.5', fontsize=12)
plt.grid(True)
plt.xlim(0, len(contaminantes)) # Aseguramos que el eje X empiece en 0

plt.savefig('../Images/sTiempoPM2.5_porMuestra.png', dpi=300, bbox_inches='tight')

plt.show()


# ### <a id="3-2"></a>3.2. Descomposición de la Serie
# 
# Buscamos identificar los patrones subyacentes que componen los datos. Utilizaremos una técnica de descomposición clásica para separar nuestra serie de `PM2.5` en tres componentes:
# 
# * **Tendencia:** Muestra la dirección general de los datos a largo plazo. ¿Está la contaminación aumentando, disminuyendo o manteniéndose estable durante el mes?
# * **Estacionalidad :** Captura los patrones que se repiten en intervalos fijos. Dado que tenemos datos horarios, esperamos ver un **ciclo diario** (24 horas) en la contaminación.
# * **Residuo :** Es el "ruido" que queda después de extraer la tendencia y la estacionalidad. Representa la variabilidad aleatoria en los datos.
# 
#  `Serie Original = Tendencia + Estacionalidad + Residuo`.



# Descomposición de la Serie de Tiempo de PM2.5 (con Eje X numérico)

descomposicion = seasonal_decompose(contaminantes['PM2.5'], model='additive', period=24)

# Creamos un eje X numérico de 0 a 741
ejeX_numerico = np.arange(len(contaminantes))


fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True) # sharex=True hace que todos compartan el mismo eje X

# Gráfico 1: Datos Observados
axes[0].plot(ejeX_numerico, descomposicion.observed)
axes[0].set_ylabel('Observado')
axes[0].grid(True)

# Gráfico 2: Tendencia
axes[1].plot(ejeX_numerico, descomposicion.trend)
axes[1].set_ylabel('Tendencia')
axes[1].grid(True)

# Gráfico 3: Estacionalidad
axes[2].plot(ejeX_numerico, descomposicion.seasonal)
axes[2].set_ylabel('Estacionalidad')
axes[2].grid(True)

# Gráfico 4: Residuo
axes[3].plot(ejeX_numerico, descomposicion.resid, marker='.', linestyle='none')
axes[3].set_ylabel('Residuo')
axes[3].set_xlabel('Índice de Muestra Horaria (0 a 741)')
axes[3].grid(True)



plt.suptitle('Descomposición de la Serie de Tiempo PM2.5', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.97]) 
plt.savefig('../Images/descomposicionPM2.5_numerico.png', dpi=300, bbox_inches='tight')
plt.show()


# **Referencias a la funcion: seasonal_decompose**
# * [Documentación Oficial de `statsmodels`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html#statsmodels.tsa.seasonal.seasonal_decompose)

# ### <a id="3-3"></a>3.3. Observaciones de la Serie Desestacionalizada
# 
# 
# La descomposición nos permite aislar la tendencia de los patrones repetitivos. Una serie **desestacionalizada** es simplemente la serie original a la que se le ha restado el componente estacional (`Serie Desestacionalizada = Tendencia + Residuo`).
# 
# El propósito de visualizarla es poder observar la tendencia subyacente de forma mucho más clara, sin la "distracción" de los ciclos diarios. A continuación, graficaremos la serie original y la desestacionalizada juntas para comparar.



# Serie Desestacionalizada

tendencia = descomposicion.trend
residuo = descomposicion.resid
estacionalidad = descomposicion.seasonal

# Calculamos la serie
serieDesestacionalizada = contaminantes['PM2.5'] - estacionalidad

# Creamos el eje X numérico
ejeX_numerico = np.arange(len(contaminantes))

plt.figure(figsize=(17, 7))
plt.plot(ejeX_numerico, contaminantes['PM2.5'], label='Serie Original', alpha=0.6)
plt.plot(ejeX_numerico, serieDesestacionalizada, label='Serie Desestacionalizada (Tendencia + Residuo)', linewidth=2)

plt.title('Comparación: Serie Original vs. Serie Desestacionalizada de PM2.5', fontsize=16)
plt.xlabel('Índice de Muestra Horaria (0 a 741)', fontsize=12)
plt.ylabel('Concentración de PM2.5', fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig('../Images/serieOriginalDesestacionalizada.png', dpi=300, bbox_inches='tight')

plt.show()


# #### Análisis de la Comparación: Original vs. Desestacionalizada
# 
# El gráfico anterior ilustra de manera contundente el éxito y la utilidad de la descomposición de la serie temporal.
# 
# * **La Serie Original (azul):** Se observa como una línea altamente volátil y "ruidosa". Los picos y valles agudos que se repiten constantemente corresponden al **patrón estacional diario**. Si bien muestra los datos en su totalidad, este ruido diario dificulta la identificación de patrones a más largo plazo.
# 
# * **La Serie Desestacionalizada (naranja):** Es notablemente más suave. Al haber restado matemáticamente el componente estacional, hemos filtrado el ciclo repetitivo de 24 horas. Lo que emerge es la **tendencia subyacente** con una claridad mucho mayor.
# 
# **Observaciones Clave:**
# 
# Gracias a la línea naranja, ahora es posible identificar fácilmente los **episodios de contaminación que duran varios días**. En lugar de ver solo picos horarios, podemos detectar las "olas" de contaminación. Por ejemplo, se observan claramente los periodos de alta contaminación que culminan cerca de los puntos 250, 350 y, de forma más pronunciada, el gran evento alrededor del punto 450 en el eje X.
# 
# 

# ### <a id="3-3"></a>3.3. Observaciones de la Serie Desestacionalizada y Frecuencia de Picos
# 
# El gráfico anterior compara la serie original (azul) con su versión desestacionalizada (naranja) a lo largo de los **742 registros horarios** que componen el mes. El eje X representa el número de cada hora consecutiva, desde el punto `0` (la primera hora del mes) hasta el `741` (la última).
# 
# **Análisis de la Comparación:**
# 
# Al eliminar el "ruido" del ciclo diario, la línea naranja (desestacionalizada) nos permite ver la tendencia subyacente con mucha más claridad. Gracias a esto, podemos analizar la frecuencia de los eventos de contaminación:
# 
# * **Frecuencia de Picos:** Observando la línea naranja, podemos identificar visualmente **aproximadamente 4 o 5 picos principales** a lo largo del mes. Esto sugiere que los episodios de alta contaminación no son eventos de un solo día, sino que duran varios días y ocurren, a grandes rasgos, una vez por semana.
# 
# Esta vista macro nos lleva a una pregunta más específica: **¿estos picos están relacionados con días particulares de la semana, como los días laborales vs. los fines de semana?** 



# Análisis por Día de la Semana 

# Nos aseguramos de que el índice sea de tipo datetime
contaminantes.index = pd.to_datetime(contaminantes.index)

# Aqui estamos convirtiedo nuestro index al dia correspondiente a la fecha 
contaminantes['DiaSemana'] = contaminantes.index.day_name()

# Promedio de contaminantes porp dia 
promedioPorDia = contaminantes.groupby('DiaSemana')[['PM2.5', 'NO2', 'CO']].mean()

# Ordenar los días
diasOrdenados = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
promedioPorDia = promedioPorDia.reindex(diasOrdenados)

print("Promedio de contaminantes por día")
display(promedioPorDia)

promedioPorDia.plot(kind='bar', 
                    subplots=True, 
                    figsize=(15, 12), 
                    legend=False,
                    title='Concentración Promedio de Contaminantes por Día de la Semana')

plt.tight_layout()
plt.savefig('../Images/averageDia.png', dpi=300, bbox_inches='tight')


plt.show()


# 
