# TC3002B_IA

## Descripción del Proyecto

Este proyecto se enfoca en la clasificación binaria de datos mediante técnicas de aprendizaje supervisado. Para el desarrollo de las distintas fases, se utilizaron múltiples datasets provenientes de la plataforma Kaggle. Esta decisión se tomó con el objetivo de mejorar la robustez del modelo y cubrir distintos escenarios de evaluación. Aunque los datasets presentan diferencias, se emplearon estrategias de preprocesamiento y técnicas de modelado similares para asegurar la coherencia entre los experimentos. A lo largo del proceso se probaron diversas arquitecturas y algoritmos de clasificación inspirados de investigaciones de estado del arte, con el fin de identificar el modelo que ofrezca el mejor rendimiento.

## Descripción de los Datasets

Como se mencionó anteriormente, ambos datasets fueron encontrados en la plataforma Kaggle. Estos datasets fueron elegidos porque cumplían con lo necesario para ser usados en aprendizaje supervisado.

El primer dataset, que se puede encontrar como "data.csv", fue el dataset usado para el avance de las primeras dos fases. Este dataset fue construido para la predicción de si un empleado iba a pertenecer a la empresa al finalizar el año. Contaba con 10,000 registros de información de empleados. Incluía información demográfica, personal, desempeño, satisfacción con la empresa, entre muchas otras features. En total contaba con 26 features, siendo la última nuestra target feature, que era de clasificación binaria (Yes/No).
Link: https://www.kaggle.com/datasets/ziya07/employee-attrition-prediction-dataset

El segundo dataset, guardado como "alzheimers_disease_data.csv", cuenta con información médica de 2,149 pacientes. Tiene un total de 35 features, siendo la penúltima nuestra target feature llamada "Diagnosis", una clasificación binaria (0/1) dependiendo de si el paciente fue diagnosticado o no con la enfermedad.
Link: https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset

## Estado del Arte

Para desarrollar el proyecto se consultó de papers científicos de los cuales se obtuvieron ejemplos de proyectos e investigaciones anteriores relacionados al tema elegido. Estos papers de investigación fueron críticos para encontrar experimentos que se pusieron a prueba, tanto de modelos, como de configuración y estadísticas para medir qué tan robusto era mi modelo.

1) https://pmc.ncbi.nlm.nih.gov/articles/PMC10801181/pdf/fbioe-11-1336255.pdf
El paper fue encontrado en una busqueda de google en la página 'National Library of Medicine'. Este paper me llamó la atención por múltiples razones pero principalmente por la confianza que me dio al velro publicado en la libreria. Ademas, este paper seguia un enfoque muy similar al dataset elegido para mi proyecto y usaba una arquitectura sumamente interesante que creaba una solución hibrida. Este paper es muy detallado, conlleva múltiples experimentos, mucho analisis, comparaciones, estadisticas. EL paper fue critico para las ultimas fases ya que use la misma infraestrucutra para replicar los experimentos con mi dataset y con mi procesado de datos, ademas para estadisticas mas avanzadas.

### Papers No Aceptados

1) https://arxiv.org/pdf/1711.07831
Este primer artículo se usó para implementar dos modelos de experimentación que se pueden encontrar dentro de los archivos "alzheimerInitialMPL.ipynb" y "alzheimerGru.ipynb". Los resultados serán documentados a continuación.
2) https://riti.es/index.php/riti/article/view/87/107
Este segundo artículo fue crucial para encontrar el modelo usado para la entrega #3. Se discutieron el uso de dos modelos distintos, sin embargo, como no se entró en tanto detalle para la configuración del modelo, opté por seguir buscando.
3) https://arxiv.org/pdf/1911.11471
Este artículo habló sobre el uso de Random Forest para un proyecto similar al mío. Fue esencial para encontrar recomendaciones de la configuración del modelo usado para la entrega de la fase #3.

## Generación o Selección del Set de Datos

Ambos datasets fueron descargados de la plataforma de Kaggle y fueron importados haciendo uso de la librería de pandas, que me permitió estar haciendo modificaciones a los datasets de manera rápida y efectiva. Una vez importados los datos, los separé haciendo uso de la función train_test_split de la librería sklearn. Esta es la forma más rápida de hacer la separación de datos para entrenar nuestro modelo y para probarlo una vez teniendo el modelo. Estuve jugando con la configuración haciendo experimentos con una separación de 70/30%, pero obtuve mejores resultados con lo que es considerado el estándar en la industria: 80% para entrenamiento y 20% para la validación.

### Data Augmentation

Un problema que encontré en el primer dataset, y por el que tuve un modelo con pésimo rendimiento, fue el desbalance de los datos, ya que el 81% de los datos contaba con una sola clasificación, por lo que mi modelo únicamente estaba prediciendo esa clase. Para evitar este desbalance hice uso de la herramienta SMOTE de la librería imblearn. SMOTE significa Synthetic Minority Oversampling Technique y es un método estadístico que genera datos de manera artificial para balancear un set de datos. Estuve jugando con otras alternativas de esta misma librería, como el undersampling, pero obtuve buenos resultados con SMOTE y es fácil de integrar. Esto me permitía aumentar un set de datos para balancearlo.

## Preprocesado de los Datos

Para el preprocesado de datos usé tres técnicas:

1) OneHotEncoder: Es una técnica usada para los datos categóricos. Es una función avanzada ya que considera posibles riesgos que aparecerían con métodos más básicos. Digo que es avanzada porque balancea features categóricos cuando no son binarios o cuando cuentan con más de dos categorías. Esto es importante ya que tenemos que convertir los datos categóricos en información que sea aceptada por nuestro modelo, pero que al mismo tiempo esté balanceada. OneHotEncoder crea matrices o features dependiendo del número de categorías. Se mostrará una imagen a continuación.
<img width="248" alt="image" src="https://github.com/user-attachments/assets/bd6eba4f-e831-46cd-b5a1-b7a659fed8b9" />

2) LabelEncoder: Es otra técnica usada para los datos categóricos, pero esta vez para casos binarios. Usé este método únicamente para la target feature, ya que en el primer dataset no eran datos numéricos.

3) StandardScaler: Finalmente usé este método para el escalamiento de datos, ya que ciertos features en ambos datasets estaban un tanto dispersos. Realizar una normalización de datos ayuda a balancearlos y a mejorar el rendimiento de nuestros modelos, por lo que la transformación de estos fue esencial para mejorar el input usado en nuestro modelo, y con ello, obtener mejores resultados.

## Implementación de Modelo 

### Versión Final

Para mi implementación final hice uso del paper del estado del arte que contaba con multiples experimentos y fases. Todos los modelos creados fueron usados con la ayuda del framework "sklearn" que permite usar estos modelos de manera rapida y flexible. Ademas, para la etapa inicial s euso la configracio2n base para todos los modelos, tal cual lo decia el estado del arte.

#### Fase Uno Experimento Uno

Inicialmente el paper hacia uso de 8 modelos todos con su configuración básica. Para estos modelos se usaba el dataset balanceado completo, sin excluir columnas. Para esta arquitectura inicial se hicieron uso de me2tricas que seran discutidas en la proxima sección. Los modelos que fueron utilizados fueron los siguientes: DecisionTreeClassifier,  GaussianNB, LogisticRegression, RandomForestClassifier, LinearDiscriminantAnalysis, AdaBoostClassifier, KNeighborsClassifier.

#### Fase Uno Experimento Dos

Esta propuesta es una arquitectura significativamente mas compleja, la propuesta es un modelo hibrido que hace uso de la combinación de modelos ademas de la selección de sets de features buscnado la optimización de la informacion ingresada al modelo.
Para esta etapa se hicieron uso de te2cnicas mas avanzadas como lo fueron:

1) SelectKBest: Esta herramienta permite elegir las características mas significativas para la predicción de resultados. Es una herramienta que se complementa con muchas otras para la optimizacioón, ya que peudes tu decidir "k" que es el numero seleccionado de columnas deseadas o se puede optimizar buscando la mejor combinación o selección de columnas posibles, que fue lo que se temrino usando con la herramienta siguiente.
2) VotingClassifier: Esta herramienta crea un modelo combiando que entrena modelos por separados pero une las predicciones de los modelos deseados para predicciones finales. Esto se hace ya que cada modelo se entrena de manera diferente, ciertos modelos pueden destacar en ciertos casos y tener desempeños bajos en otros, de esta manera podemos compensar errores de modelos individuales y podemos obtener modelos mas fuertes y estables. Existen dos configuraciones para esta herramienta 'soft' y 'hard'. Unicamente entrare en detalle con 'hard' ya que fue la que se usaba en el estado del arte seleccionado. Esta configuración cuenta la predicción de la clase mas votada dentro de todas las predicciones de los modelos y decide en esa predicción mas votada. El estado del arte hacia uso de 5 modelos para el voting classifier estos siendo: DecisionTreeClassifier, SVC, GaussianNB, LogisticRegression y RandomForestClassifier, los cuales tambien se usaron para mi implementacion.

Para este experimento buscamos conocer el numero optimo de columnas o la "k" optima para el siguiente experimento. Usamos el modelo conjunto para evaluar ciertas metricas que se discturian proximamente y se evaluaron para sets de caracteristicas. Estos sets fuero extraidos de la lectrua y son los siguientes. [2, 4, 6, 7, 8, 9, 10, 12] 

#### Fase Dos Experimento 1

Para este experimento se hizo uso de conceptos muy similares al experimento anterior, donde se hizo uso de un mismo voting classifier con la misma configuración, simplemente haciendo uso del mejor subcojunto definido por el experiment anterior. El mejor resultado fue para cuando se usaron los 6 features mas significativos para la predicción. La unica diferencia en este experimento es un concepto nuevo conocido como cross_validate.

Este concepto es mas relacionado al tema de metricas y validaciown, pero en terminos simples nos permite validar si el el modelo funciona no solo para una segmentación detallada o especifica de datos. Ademas, el paper usado usa este tecnica para asegurarse que el modelo no esta sobre ajustado antes de ser utilizado para las metricas y resultados, ya que queremoe evitar que el modelo memorize en lugar de parender. Para este experimento el paper recomenda un valor de 5.

#### Fase Dos Experimento 2

Este experimento fue mas enfocado a la evaluación de caracteristicas por lo que se explicara mas en la siguiente sección

## Evaluación inicial del modelo

Para la evaluación de los modelo hice uso de métricas vistas en clase asi como metricas mas avanzadas propuestas en el articulo del estado del arte. Devido a que las metricas se usaban de manera frecuente decidi crear dos funciones reutilizables para optimizar el codigo. La primera siendo "calcular_matriz_confusion" que regresaba las estadisticas que se peuden obtener haciendo uso  de los verdaderos positivos, negativos y falsos positivos y negativos. En esta función se calculaban las siguientes metricas: precision, accuracy, sensitivity, specificity, f1, mcc. Todas estas fueron calucladas de manera manual. La siguiente función es "evaluate_model" esta regresa todas las metricas incouyendo las de lafunción anterior, especificamente regresa auc que es una metrica discutida en el paper de investigación seleccionado.

### Resultados Fase Uno Experimento Uno

| Model    | Accuracy | Precision  | Sensitivity | Specificity | F1-Score | MCC      | AUC-ROC  |
| -------- | ---------| -----------|-------------| ------------| ---------| ---------| ---------|
| DT       | 0.891473 |  0.856115  |   0.843972  |   0.918699  | 0.850000 | 0.765034 | 0.881335 |
| SVM      | 0.839793 |  0.772414  |   0.794326  |   0.865854  | 0.783217 | 0.656368 | 0.902583 | 
| NB       | 0.692506 |  0.568750  |   0.645390  |   0.719512  | 0.604651 | 0.356600 | 0.767947 |
| LR       | 0.824289 |  0.715976  |   0.858156  |   0.804878  | 0.780645 | 0.643341 | 0.903217 |
| RF       | 0.930233 |  0.919118  |   0.886525  |   0.955285  | 0.902527 | 0.848564 | 0.944459 | 
| LDA      | 0.813953 |  0.699422  |   0.858156  |   0.788618  | 0.770701 | 0.626037 | 0.901978 |
| AdaBoost | 0.909561 |  0.848684  |   0.914894  |   0.906504  | 0.880546 | 0.809422 | 0.943868 |
| kNN      | 0.640827 |  0.504348  |   0.822695  |   0.536585  | 0.625337 | 0.352125 | 0.731433 |

#### Comparación ROC

![image](https://github.com/user-attachments/assets/6c020263-8522-479f-ad06-264d3eb4a09f)

### Resultados Fase Uno Experimento Dos

#### Top 10 características F-Score:

| Feature               | importance |   
| --------------------- | ---------- | 
| FunctionalAssessment  | 357.44     |  
| ADL                   | 305.2      |  
| MemoryComplaints_1    | 214.3      |  
| MemoryComplaints_0    | 214.38     |  
| MMSE                  | 159.40     |  
| BehavioralProblems_0  | 74.33      |  
| BehavioralProblems_1  | 74.33      |  
| Confusion_0           | 12.51      |  
| Confusion_1           | 12.51      |  
| SleepQuality          | 10.02      |  

#### Evaluación de Subconjuntos:

| Sub_Fe | Accuracy | Sensitivity | Specificity | MCC         |
| ------ | ---------| ------------|-------------| ------------| 
| 2      | 0.754522 |  0.595745   |   0.845528  |   0.457069  | 
| 4      | 0.798450 |  0.751773   |   0.825203  |   0.570629  |
| 6      | 0.919897 |  0.914894   |   0.922764  |   0.829495  | 
| 7      | 0.914729 |  0.907801   |   0.918699  |   0.818447  | 
| 8      | 0.914729 |  0.907801   |   0.918699  |   0.818447  | 
| 9      | 0.914729 |  0.907801   |   0.918699  |   0.818447  | 
| 10     | 0.912145 |  0.914894   |   0.910569  |   0.814386  | 
| 12     | 0.919897 |  0.914894   |   0.922764  |   0.829495  |

#### Mejor subconjunto:

6 características
Accuracy: 0.92%

### Resultados Fase Dos Experimento Uno

#### Cross Validation

Para esta etapa se hizo uso de la técnica de cross validation, la configuracio2n sugerida por el paper de investigación fueron k = 5.

| Accuracy    | 93.41% |
| ------ | ---------|
| Precision   | 94.10% |  
| Sensitivity | 92.63% |  
| F1-Score    | 93.34% |

#### Evaluación Modelo Final

| Accuracy    | 91.99% |
| ------ | ---------| 
| Precision   | 87.16% |  
| Sensitivity | 91.49% |  
| Specificity | 92.28% |
| F1-Score    | 89.27% |
| MCC         | 82.95% |

### Resultados Fase Dos Experimento Uno

Ademas, el paper de investigación hace uso de validaciones para la selección de características.

| Método     | Accuracy |
| ------ | ---------| 
| F-Score    | 93.407%  |  
| Chi-Square | 92.857%  |  
| MutInfo    | 87.313%  |
| RFE        | 87.513%  |
| Lasso      | 93.406%  |

## Conclusiones

Hasta ahora, el modelo que mejor ha funcionado ha sido el Random Forest, usando la configuración tomada del artículo número 3. Todavía faltan varias fases por completar y hay muchos experimentos por hacer, pero este modelo ha sido el que mejor se ha ajustado a lo que busco. En las métricas se puede ver claramente que tuvo un mejor rendimiento comparado con los otros modelos probados.

