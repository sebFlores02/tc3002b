## Implementación de Modelo 

### Versión Inicial

Para la implementación de los modelos me apoyé en el framework de TensorFlow con Keras, ya que es increíblemente fácil de incorporar y muy flexible para mis necesidades.

Para la primera iteración usando el primer set de datos, usé un modelo Sequential básico que contaba con una capa Flatten, una capa Dense de 256 neuronas y una capa Dense con shape=1 debido a que mi clasificación es binaria. Este modelo contaba con un total de 42,245 parámetros, pero fue experimental, y para la siguiente entrega se usaron modelos más avanzados.

Usé este mismo modelo cuando cambié de dataset, únicamente para comparar los resultados obtenidos con arquitecturas y configuraciones más avanzadas.

Siguiendo el paper marcado como #2, implementé un modelo ligeramente más robusto como otro experimento. Este modelo seguía la base del primer modelo, pero usando una capa más. Estas capas eran más densas y contaban con más neuronas, ya que eran de 500.

Para el tercer modelo, inspirado en el mismo paper, realicé un modelo GRU, usando como primera capa una GRU, una segunda capa Dropout, y una última capa de tamaño uno para mi clasificación binaria. Este modelo contaba con 38,789 parámetros y usaba un loss personalizado llamado l2_svm_loss, tal como lo indicaba el artículo de investigación.

Finalmente, mi último modelo usado para la entrega, que fue inspirado por el segundo y tercer artículo de investigación, fue el Random Forest. Es un modelo avanzado que crea un número deseado de árboles usando subconjuntos aleatorios de features. Este modelo combina los resultados de los árboles para obtener un resultado más acertado. Es un modelo que destaca en la capacidad predictiva y tiende a evitar riesgos comunes como lo es el overfitting.

#### Evaluación inicial del modelo

Las métricas usadas para la evaluación de los modelos fueron vistas en clase y, además, son las mismas que fueron utilizadas en los tres artículos de investigación seleccionados. Estas métricas son las más recomendadas, ya que te cuentan la historia completa de lo que está sucediendo. Esto es importante, ya que en el experimento de mi primer dataset me pasó factura no contar con estas métricas. Me dejé llevar por un accuracy de 60% y lo presenté ante el profesor, hasta que me di cuenta de que mi modelo únicamente estaba prediciendo una clase. Al incorporar métricas más avanzadas y al evaluarlo en una matriz de confusión, me di cuenta de lo que estaba pasando y que mi modelo no se estaba comportando como debía.

##### Dataset 1

###### Ronda 1

###### Model: Sequential

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

##### Dataset 2

###### Ronda 1

####### Model: Sequential

| layer   | shape | activation |   
| ------- | ------| ---------- |
| Flatten |       |            |
| Dense   | 256   |  relu      |
| Dense   | 1     |  sigmoid   |

###### Compile

| optimizer  | loss                | metrics  |   
| ---------- | --------------------| ---------|
| adam       | binary_crossentropy | accuracy |

###### Epochs: 200

###### Metricas:

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
