# TC3002B_IA

## Descripción del Proyecto

Este proyecto se enfoca en la clasificación binaria de datos mediante técnicas de aprendizaje supervisado. Para el desarrollo de las distintas fases, se utilizaron múltiples datasets provenientes de la plataforma Kaggle. Esta decisión se tomó con el objetivo de mejorar la robustez del modelo y cubrir distintos escenarios de evaluación. Aunque los datasets presentan diferencias, se emplearon estrategias de preprocesamiento y técnicas de modelado similares para asegurar la coherencia entre los experimentos. A lo largo del proceso se probaron diversas arquitecturas y algoritmos de clasificación, con el fin de identificar el modelo que ofrezca el mejor rendimiento de acuerdo con las métricas clásicas de evaluación: precisión, recall y F1-score.

## Descripción de los Datasets

Como se mencionó anteriormente, ambos datasets fueron encontrados en la plataforma Kaggle. Estos datasets fueron elegidos porque cumplían con lo necesario para ser usados en aprendizaje supervisado.

El primer dataset, que se puede encontrar como "data.csv", fue el dataset usado para el avance de las primeras dos fases. Este dataset fue construido para la predicción de si un empleado iba a pertenecer a la empresa al finalizar el año. Contaba con 10,000 registros de información de empleados. Incluía información demográfica, personal, desempeño, satisfacción con la empresa, entre muchas otras features. En total contaba con 26 features, siendo la última nuestra target feature, que era de clasificación binaria (Yes/No).
Link: https://www.kaggle.com/datasets/ziya07/employee-attrition-prediction-dataset

El segundo dataset, guardado como "alzheimers_disease_data.csv", cuenta con información médica de 2,149 pacientes. Tiene un total de 35 features, siendo la penúltima nuestra target feature llamada "Diagnosis", una clasificación binaria (0/1) dependiendo de si el paciente fue diagnosticado o no con la enfermedad.
Link: https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset

## Estado del Arte

Para desarrollar el proyecto se consultó de papers científicos de los cuales se obtuvieron ejemplos de proyectos e investigaciones anteriores relacionados al tema elegido. Estos papers de investigación fueron críticos para encontrar experimentos que se pusieron a prueba, tanto de modelos, como de configuración y estadísticas para medir qué tan robusto era mi modelo.

1) https://arxiv.org/pdf/1711.07831
Este primer artículo se usó para implementar dos modelos de experimentación que se pueden encontrar dentro de los archivos "alzheimerInitialMPL.ipynb" y "alzheimerGru.ipynb". Los resultados serán documentados a continuación.
2) https://riti.es/index.php/riti/article/view/87/107
Este segundo artículo fue crucial para encontrar el modelo usado para la entrega #3. Se discutieron el uso de dos modelos distintos, sin embargo, como no se entró en tanto detalle para la configuración del modelo, opté por seguir buscando.
3) https://arxiv.org/pdf/1911.11471
Este artículo habló sobre el uso de Random Forest para un proyecto similar al mío. Fue esencial para encontrar recomendaciones de la configuración del modelo usado para la entrega de la fase #3.

## Generación o Selección del Set de Datos

Ambos datasets fueron descargados de la plataforma de Kaggle y fueron importados haciendo uso de la librería de pandas, que me permitió estar haciendo modificaciones a los datasets de manera rápida y efectiva. Una vez importados los datos, los separé haciendo uso de la función train_test_split de la librería sklearn. Esta es la forma más rápida de hacer la separación de datos para entrenar nuestro modelo y para probarlo una vez teniendo el modelo. Estuve jugando con la configuración haciendo experimentos con una separación de 70/30%, pero obtuve mejores resultados con lo que es considerado el estándar en la industria: 80% para entrenamiento y 20% para la validación.

Un problema que encontré en el primer dataset, y por el que tuve un modelo con pésimo rendimiento, fue el desbalance de los datos, ya que el 81% de los datos contaba con una sola clasificación, por lo que mi modelo únicamente estaba prediciendo esa clase. Para evitar este desbalance hice uso de la herramienta SMOTE de la librería imblearn. SMOTE significa Synthetic Minority Oversampling Technique y es un método estadístico que genera datos de manera artificial para balancear un set de datos. Estuve jugando con otras alternativas de esta misma librería, como el undersampling, pero obtuve buenos resultados con SMOTE y es fácil de integrar. Esto me permitía aumentar un set de datos para balancearlo.

## Preprocesado de los Datos

Para el preprocesado de datos usé tres técnicas:

1) OneHotEncoder: Es una técnica usada para los datos categóricos. Es una función avanzada ya que considera posibles riesgos que aparecerían con métodos más básicos. Digo que es avanzada porque balancea features categóricos cuando no son binarios o cuando cuentan con más de dos categorías. Esto es importante ya que tenemos que convertir los datos categóricos en información que sea aceptada por nuestro modelo, pero que al mismo tiempo esté balanceada. OneHotEncoder crea matrices o features dependiendo del número de categorías. Se mostrará una imagen a continuación.
<img width="248" alt="image" src="https://github.com/user-attachments/assets/bd6eba4f-e831-46cd-b5a1-b7a659fed8b9" />

2) LabelEncoder: Es otra técnica usada para los datos categóricos, pero esta vez para casos binarios. Usé este método únicamente para la target feature, ya que en el primer dataset no eran datos numéricos.

3) StandardScaler: Finalmente usé este método para el escalamiento de datos, ya que ciertos features en ambos datasets estaban un tanto dispersos. Realizar una normalización de datos ayuda a balancearlos y a mejorar el rendimiento de nuestros modelos, por lo que la transformación de estos fue esencial para mejorar el input usado en nuestro modelo, y con ello, obtener mejores resultados.
   
## Implementación de Modelo 

Para la implementación de los modelos me apoyé en el framework de TensorFlow con Keras, ya que es increíblemente fácil de incorporar y muy flexible para mis necesidades.

Para la primera iteración usando el primer set de datos, usé un modelo Sequential básico que contaba con una capa Flatten, una capa Dense de 256 neuronas y una capa Dense con shape=1 debido a que mi clasificación es binaria. Este modelo contaba con un total de 42,245 parámetros, pero fue experimental, y para la siguiente entrega se usaron modelos más avanzados.

Usé este mismo modelo cuando cambié de dataset, únicamente para comparar los resultados obtenidos con arquitecturas y configuraciones más avanzadas.

Siguiendo el paper marcado como #2, implementé un modelo ligeramente más robusto como otro experimento. Este modelo seguía la base del primer modelo, pero usando una capa más. Estas capas eran más densas y contaban con más neuronas, ya que eran de 500.

Para el tercer modelo, inspirado en el mismo paper, realicé un modelo GRU, usando como primera capa una GRU, una segunda capa Dropout, y una última capa de tamaño uno para mi clasificación binaria. Este modelo contaba con 38,789 parámetros y usaba un loss personalizado llamado l2_svm_loss, tal como lo indicaba el artículo de investigación.

Finalmente, mi último modelo usado para la entrega, que fue inspirado por el segundo y tercer artículo de investigación, fue el Random Forest. Es un modelo avanzado que crea un número deseado de árboles usando subconjuntos aleatorios de features. Este modelo combina los resultados de los árboles para obtener un resultado más acertado. Es un modelo que destaca en la capacidad predictiva y tiende a evitar riesgos comunes como lo es el overfitting.

## Resultados

Las métricas usadas para la evaluación de los modelos fueron vistas en clase y, además, son las mismas que fueron utilizadas en los tres artículos de investigación seleccionados. Estas métricas son las más recomendadas, ya que te cuentan la historia completa de lo que está sucediendo. Esto es importante, ya que en el experimento de mi primer dataset me pasó factura no contar con estas métricas. Me dejé llevar por un accuracy de 60% y lo presenté ante el profesor, hasta que me di cuenta de que mi modelo únicamente estaba prediciendo una clase. Al incorporar métricas más avanzadas y al evaluarlo en una matriz de confusión, me di cuenta de lo que estaba pasando y que mi modelo no se estaba comportando como debía.

### Dataset 1

#### Ronda 1

##### Model: Sequential

| layer   | shape | activation |   
| ------- | ------| ---------- |
| Flatten |       |            |
| Dense   | 256   |  relu      |
| Dense   | 1     |  sigmoid   |

##### Compile

| optimizer  | loss                | metrics  |   
| ---------- | --------------------| ---------|
| adam       | binary_crossentropy | accuracy |

##### Epochs: 100

##### Metricas:

|          | label neg | label pos |   
| -------- | --------- | --------- |
| pred neg | 2005      |  520      |
| pred pos | 370       |  105      |

| metric    | value    |   
| --------- | -------- | 
| Precision | 0.221    |  
| Recall    | 0.168    |  
| F1        | 0.0955   |  

### Dataset 2

#### Ronda 1

##### Model: Sequential

| layer   | shape | activation |   
| ------- | ------| ---------- |
| Flatten |       |            |
| Dense   | 256   |  relu      |
| Dense   | 1     |  sigmoid   |

##### Compile

| optimizer  | loss                | metrics  |   
| ---------- | --------------------| ---------|
| adam       | binary_crossentropy | accuracy |

##### Epochs: 200

##### Metricas:

|          | label neg | label pos |   
| -------- | --------- | --------- |
| pred neg | 238       |  46       |
| pred pos | 39        |  107      |

| metric    | value    |   
| --------- | -------- | 
| Precision | 0.733    |  
| Recall    | 0.699    |  
| F1        | 0.716    |  

#### Ronda 2

##### Model: Sequential

| layer   | shape | activation |   
| ------- | ------| ---------- |
| GRU     | 64    |            |
| Dropout | 0.2   |            |
| Dense   | 1     |  linear    |

##### Compile

| optimizer  | loss                | metrics  |   
| ---------- | --------------------| ---------|
| adam       | l2_svm_loss         | accuracy |

##### Epochs: 200
##### batch_size=32

##### Metricas:

|          | label neg | label pos |   
| -------- | --------- | --------- |
| pred neg | 244       |  46       |
| pred pos | 33        |  107      |

| metric    | value    |   
| --------- | -------- | 
| Precision | 0.764    |  
| Recall    | 0.699    |  
| F1        | 0.730    |  

#### Ronda 3

##### Model: Random Forest

##### Configuración siguiendo artículo de investigación

| n_estimators  | max_depth | min_samples_split  | min_samples_leaf | max_features | bootstrap | criterion | class_weight | random_state | n_jobs |
| ------------- | ----------| -------------------|------------------| -------------| ----------| ----------| -------------| -------------| -------|
| 100           | 13        | 5                  | 3                | None         | True      | entropy   | balanced     | 42           | 1      |

##### Metricas:

|          | label neg | label pos |   
| -------- | --------- | --------- |
| pred neg | 270       |  13       |
| pred pos | 7         |  140      |

| metric    | value    |   
| --------- | -------- | 
| Precision | 0.952    |  
| Recall    | 0.915    |  
| F1        | 0.933    |  

###### Top 10 Most Important Features:

| feature               | importance |   
| --------------------- | ---------- | 
| FunctionalAssessment  | 0.191      |  
| MMSE                  | 0.185      |  
| ADL                   | 0.184      |  
| MemoryComplaints_1    | 0.078      |  
| BehavioralProblems_0  | 0.073      |  
| MemoryComplaints_0    | 0.066      |  
| BehavioralProblems_1  | 0.048      |  
| DietQuality           | 0.018      |  
| CholesterolHDL        | 0.016      |  
| Age                   | 0.015      |  

## Conclusiones

Hasta ahora, el modelo que mejor ha funcionado ha sido el Random Forest, usando la configuración tomada del artículo número 3. Todavía faltan varias fases por completar y hay muchos experimentos por hacer, pero este modelo ha sido el que mejor se ha ajustado a lo que busco. En las métricas se puede ver claramente que tuvo un mejor rendimiento comparado con los otros modelos probados.

