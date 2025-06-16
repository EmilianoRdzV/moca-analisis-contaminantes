# Proyecto: Análisis Estadístico y de Series de Tiempo de Contaminantes Atmosféricos

## Contexto del Proyecto

Este repositorio contiene el desarrollo de un proyecto de ciencia de datos enfocado en el análisis de contaminantes atmosféricos.

El trabajo se centra en un conjunto de datos que registra las concentraciones de seis contaminantes criterio durante el mes de enero de 2019. La implementación se realizó en un **Jupyter Notebook**, utilizando **Python** y un conjunto de librerías especializadas para el análisis de datos, incluyendo **Pandas, Matplotlib, Scikit-learn y Statsmodels**.

---

## Objetivos

El propósito de este proyecto es demostrar un flujo de trabajo de análisis de datos de principio a fin, cubriendo los siguientes puntos clave:

* **Procesamiento y Análisis Exploratorio de Datos (EDA):** Realizar la carga, limpieza y preparación de los datos. Posteriormente, explorar las relaciones entre las variables a través de estadísticas descriptivas y visualizaciones como mapas de calor de correlación para extraer los primeros insights.

* **Modelado Predictivo (Regresión):** Construir y entrenar modelos de regresión lineal con **Scikit-learn** para predecir la concentración de un contaminante (como O3 o PM2.5) basándose en otros.

* **Inferencia Estadística:** Utilizar la librería **Statsmodels** para ir más allá de la predicción y evaluar la **significancia estadística** de las variables predictoras a través del análisis de sus coeficientes, errores estándar y valores-t.

* **Análisis de Series de Tiempo:** Aplicar técnicas de descomposición a una serie temporal para aislar y analizar sus componentes de **tendencia, estacionalidad** y residuo, permitiendo identificar patrones diarios y semanales en la contaminación.

---

## Estructura del Repositorio

* **/Notebooks/CalidadAire.ipynb**: El Jupyter Notebook principal que contiene todo el código, las visualizaciones y el análisis detallado del proyecto.
* **/Data/contaminantes.xlsx**: El conjunto de datos original utilizado para el análisis.
* **/Images/**: Carpeta que contiene todas las gráficas generadas y guardadas durante el análisis (mapas de calor, diagramas de caja, gráficos de series de tiempo, etc.).
