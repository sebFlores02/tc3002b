# TC3002B_IA

## Descripción del Proyecto

Este proyecto se enfoca en la clasificación binaria de datos mediante técnicas de aprendizaje supervisado. Para el desarrollo de las distintas fases, se utilizaron múltiples datasets provenientes de la plataforma Kaggle. Esta decisión se tomó con el objetivo de mejorar la robustez del modelo y cubrir distintos escenarios de evaluación. Aunque los datasets presentan diferencias, se emplearon estrategias de preprocesamiento y técnicas de modelado similares para asegurar la coherencia entre los experimentos. A lo largo del proceso se probaron diversas arquitecturas y algoritmos de clasificación inspirados en investigaciones de estado del arte, con el fin de identificar el modelo que ofreciera el mejor rendimiento.

## Descripción de los Datasets

Como se mencionó anteriormente, ambos datasets fueron encontrados en la plataforma Kaggle. Estos datasets fueron elegidos porque cumplían con lo necesario para ser usados en aprendizaje supervisado.

El primer dataset, que se puede encontrar como "data.csv", fue el dataset usado para el avance de las primeras dos fases. Este dataset fue construido para la predicción de si un empleado iba a pertenecer a la empresa al finalizar el año. Contaba con 10,000 registros de información de empleados. Incluía información demográfica, personal, desempeño, satisfacción con la empresa, entre muchas otras features. En total contaba con 26 features, siendo la última nuestra target feature, que era de clasificación binaria (Yes/No).
Link: https://www.kaggle.com/datasets/ziya07/employee-attrition-prediction-dataset

El segundo dataset, guardado como "alzheimers_disease_data.csv", cuenta con información médica de 2,149 pacientes. Tiene un total de 35 features, siendo la penúltima nuestra target feature llamada "Diagnosis", una clasificación binaria (0/1) dependiendo de si el paciente fue diagnosticado o no con la enfermedad.
Link: https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset

## Estado del Arte

Para desarrollar el proyecto se consultó de papers científicos de los cuales se obtuvieron ejemplos de proyectos e investigaciones anteriores relacionados al tema elegido. Estos papers de investigación fueron críticos para encontrar experimentos que se pusieron a prueba, tanto de modelos, como de configuración y estadísticas para medir qué tan robusto era mi modelo.

1) https://pmc.ncbi.nlm.nih.gov/articles/PMC10801181/pdf/fbioe-11-1336255.pdf
Este paper fue encontrado en una búsqueda de Google en la página de la 'National Library of Medicine'. Me llamó la atención por múltiples razones, principalmente por la confianza que me dio al verlo publicado en esa librería. Además, seguía un enfoque muy similar al dataset elegido para mi proyecto y usaba una arquitectura sumamente interesante que creaba una solución híbrida. Este paper es muy detallado, con múltiples experimentos, mucho análisis, comparaciones y estadísticas. Fue crítico para las últimas fases ya que usé la misma infraestructura para replicar los experimentos con mi dataset y con mi procesado de datos, además de para implementar estadísticas más avanzadas.

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

### Versión Inicial

Se realizó una versión inicial de pruebas, la implementación y los resultados se pueden encontrar aquí [Modulo2/deprecated/ModelosIniciales.md](https://github.com/sebFlores02/tc3002b/blob/main/Modulo2/deprecated/ModelosIniciales.md)

### Versión Final

Para la implementación final usé el paper del estado del arte, el cual contaba con múltiples experimentos y fases. Todos los modelos fueron implementados con la ayuda del framework sklearn, que permite el uso de modelos de forma rápida y flexible. Además, para la etapa inicial se usó la configuración base indicada en el paper.

#### Fase Uno Experimento Uno

Inicialmente, el paper hacía uso de 8 modelos, todos con configuración básica. Estos modelos se entrenaban con el dataset balanceado, sin excluir columnas. Para esta arquitectura se evaluaron métricas que se discutirán en la próxima sección. Los modelos utilizados fueron: DecisionTreeClassifier, GaussianNB, LogisticRegression, RandomForestClassifier, LinearDiscriminantAnalysis, AdaBoostClassifier, y KNeighborsClassifier.

#### Fase Uno Experimento Dos

Esta propuesta incluye una arquitectura más compleja: un modelo híbrido que combina varios modelos y selecciona subconjuntos de features buscando optimizar la información ingresada al modelo.

Para esta etapa se usaron técnicas más avanzadas como:

1) SelectKBest: Permite elegir las características más significativas para la predicción. Se puede definir el valor de "k" manualmente o buscar la mejor combinación posible, lo cual fue lo que hice en conjunto con la siguiente herramienta.
2) VotingClassifier: Crea un modelo combinado que entrena distintos modelos por separado, pero une sus predicciones para generar una predicción final. Este enfoque compensa debilidades individuales y produce modelos más estables. Existen dos configuraciones: 'soft' y 'hard'. Usé 'hard', que selecciona la clase más votada entre todos los modelos. El paper usaba 5 modelos en el VotingClassifier: DecisionTreeClassifier, SVC, GaussianNB, LogisticRegression y RandomForestClassifier, los cuales también usé.
   
Para este experimento buscamos conocer el numero optimo de columnas o la "k" optima para el siguiente experimento. Usamos el modelo conjunto para evaluar ciertas metricas que se discturian proximamente y se evaluaron para sets de caracteristicas. Estos sets fuero extraidos de la lectrua y son los siguientes. [2, 4, 6, 7, 8, 9, 10, 12] 

#### Fase Dos Experimento Uno

Para este experimento se hizo uso de conceptos muy similares al experimento anterior, donde se hizo uso de un mismo voting classifier con la misma configuración, simplemente haciendo uso del mejor subconjunto definido por el experimento anterior. El mejor resultado fue para cuando se usaron los 6 features más significativos para la predicción. La única diferencia en este experimento es un concepto nuevo conocido como cross_validate.

Este concepto es más relacionado al tema de métricas y validación, pero en términos simples nos permite validar si el modelo funciona no solo para una segmentación detallada o específica de datos. Además, el paper usado usa esta técnica para asegurarse que el modelo no está sobre ajustado antes de ser utilizado para las métricas y resultados, ya que queremos evitar que el modelo memorice en lugar de aprender. Para este experimento el paper recomienda un valor de 5.

#### Fase Dos Experimento Dos

Este experimento fue más enfocado a la evaluación de características por lo que se explicará más en la siguiente sección. Aunque, se hizo uso de cross_validation pero esta vez el paper de investigación recomendaba un k fold de 10.

## Evaluación inicial del modelo

### Versión Inicial

Se realizó una versión inicial de pruebas, la implementación y los resultados se pueden encontrar aquí: [Modulo2/deprecated/ModelosIniciales.md](https://github.com/sebFlores02/tc3002b/blob/main/Modulo2/deprecated/ModelosIniciales.md)

### Versión Final

Para la evaluación de los modelos hice uso de métricas vistas en clase así como métricas más avanzadas propuestas en el artículo del estado del arte. Debido a que las métricas se usaban de manera frecuente decidí crear dos funciones reutilizables para optimizar el código. La primera siendo "calcular_matriz_confusion" que regresaba las estadísticas que se pueden obtener haciendo uso de los verdaderos positivos, negativos y falsos positivos y negativos. En esta función se calculaban las siguientes métricas: precision, accuracy, sensitivity, specificity, f1, mcc. Todas estas fueron calculadas de manera manual. La siguiente función es "evaluate_model" esta regresa todas las métricas incluyendo las de la función anterior, específicamente regresa auc que es una métrica discutida en el paper de investigación seleccionado.

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

El mejor resultado para el modelo base fue el Random Forest y se usará como referencia para la explicación. EStas metricas fueron obtenidas del paper y de la clase y fueron calcualdas manualmente haciendo uso de la matriz de confusión: Verdaderos Positivos (TP), Falsos Positivos (FP), Verdaderos Negativos (TN), y Falsos Negativos (FN).

Precision:
Indica la cantidad de predicciones positivas correctas.
Precision = TP/TP+FP

Accuracy:
Porcentaje total de predicciones correctas.
Accuracy = TP / TP + TN + FP + FN

Recall o Sensitivity:
Capacidad para identificar correctamente los positivos.
Recall = TP / TP + FN

Specificity:
Capacidad para identificar correctamente los negativos.
Specificity = TN / TN + FP

F1-Score:
Esta medida mide el balance entre la precisión y la sensibilidad y se usa para cuando existe un desbalance entre clases. Es mas especifica porque penaliza los falsos positivos y negativos.
F1 = 2 * (Precision * Recall / Precision + Recall)

MCC (Matthews Correlation Coefficient):
Esta medida es brilla en el desbalance de clases y es muy confiable para la claisifiacio2n binaria ya que considera todos los valores que existen en la matriz de confusión.
MCC = (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
Esta metrica tiene valores entre -1 y 1 donde el -1 es una clasifiacion erronea, el 0 es un desempeño aleatorio y el 1 es una clasificaciòn correcta.

​Hcaemos uso de multiples formulas para considerar todos los valores y asegurarnos que no estamos eligiendo las metricas que hagan que nuestro modelo parezca eficiente cuando puede no serlo, esto puede ocurrir si solo ahcemos iuso de una metrica como accuracy. Para el set de modelos base podemos ver como el modelo de Random FOrest es el que mejor calcula los valores de la matriz de confusion.

#### Comparación ROC

La metrica "Área Bajo la Curva" grafica la tasa de verdaderos positivos contra la tasa de falsos positivos. Esta metrica nos inidica si el modelo es capaz de distinguir las clases, realizar clasificaionces correctas ys e complementa con la metricas como accuracy y precision Esta misma grafica fue extraido del paper del estado del arte.

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

F-Score permite cuantificar la significancia de una feature para la predicción o clasificación de las clases. Esot significa que entre masyor putnaje tenga una features es mas relevante para la clasificación y predicción. Es un proceso matematico que permite orednar y seleccionar las feauters mas importantes para reducir la cantidad de informacion de un dataset sin sacrificar las caracerisicas mas importantes para la prediccion. Para este proyecto podemos observar como "FunctionalAssessment" es la caracteristica mas importante para la predicción ya que cuenta con un mayor puntaje comparado al resoto de features.

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

En este proceso hacemos uso de los features más importantes que fueron calculados mediante el F-Score. Estamos evaluando subconjuntos de características y estamos generando y evaluando un modelo para cada uno de los subconjuntos y evaluando con las métricas previamente analizadas. Esto nos permite obtener el número óptimo de subconjuntos que será usado para entrenar el modelo final. Estos subconjuntos fueron indicados en el paper. Para nuestra situación obtuvimos que el subconjunto con los mejores resultados fue de 6. Podemos observar cómo entrenar y probar con pocos subconjuntos afecta negativamente el desempeño del modelo.

#### Mejor subconjunto:

6 características
Accuracy: 0.92%

Esta métrica de accuracy mide la capacidad de construir un modelo efectivo con el conjunto seleccionado. Usa métricas under the hood como cross validation para llegar al resultado.

### Resultados Fase Dos Experimento Uno

#### Cross Validation

Para esta etapa se hizo uso de la técnica de cross validation, la configuración sugerida por el paper de investigación fueron k = 5.
Esta técnica permite evaluar el modelo múltiples veces y obteniendo un promedio después del proceso. De manera más específica, si tenemos 5 "folds" dividimos nuestra información, usamos 4 partes para entrenar y una para validar, y realizamos 5 rotaciones para evaluar de manera profunda el modelo. El promedio de las rotaciones es el resultado, esto lo hacemos para asegurarnos que nuestro modelo esté aprendiendo y no memorizando. Esta técnica fue usada en la metodología del paper consultado. Podemos observar que al evaluar con las mismas métricas obtuvimos los mejores resultados hasta el momento. Esto nos indica que el modelo verdaderamente está aprendiendo y es robusto.

| Accuracy    | 93.41% |
| ------ | ---------|
| Precision   | 94.10% |  
| Sensitivity | 92.63% |  
| F1-Score    | 93.34% |

#### Evaluación Modelo Final

Finalmente, evaluamos de la misma manera que con los modelos base, ya que realizar una comparación de cross validation para el modelo híbrido no es justo ya que obtenemos mucha más profundidad con cross validation. Podemos observar que los resultados son muy buenos, estando cerca o por encima del 90% en la mayoría de las métricas, indicando un buen desempeño en todas las métricas asegurando que el modelo es robusto y de utilidad.

| Accuracy    | 91.99% |
| ------ | ---------| 
| Precision   | 87.16% |  
| Sensitivity | 91.49% |  
| Specificity | 92.28% |
| F1-Score    | 89.27% |
| MCC         | 82.95% |

### Resultados Fase Dos Experimento Uno

Además, el paper de investigación hace uso de validaciones para la selección de características. Esta comparación es importante ya que existen múltiples técnicas para realizar la optimización.
De manera simplificada:

- F-Score mide de manera individual la relación con la clase sin considerar otras interacciones y es ideal para datos numéricos en clasificaciones binarias.
- Chi-Square se usa para detectar relaciones no lineales entre variables categóricas y clases deseadas.
- Mutual Information considera conjuntos y permite saber cómo una característica reduce la incertidumbre de la predicción de una clase.
- RFE usa la iteración para eliminar las clases con menos peso o menos importantes en cada "paso" dado.
- Lasso permite eliminar características irrelevantes y penaliza usar un número excesivo de características.

| Método     | Accuracy |
| ------ | ---------| 
| F-Score    | 93.407%  |  
| Chi-Square | 92.857%  |  
| MutInfo    | 87.313%  |
| RFE        | 87.513%  |
| Lasso      | 93.406%  |

Podemos observar cómo la propuesta inicial "F-Score" obtuvo los mejores resultados. Para esta comparación se usa accuracy que, como ya se discutió, indica qué tan bueno es el modelo que se puede construir con el subconjunto seleccionado.

## Usar el Modelo

Para hacer uso del modelo, se puede clonar el repositorio, acceder a la carpeta del módulo donde se encuentran los archivos. El proyecto se desarrolló en un formato .ipynb que permite correr bloques de código y permite separar los pasos y lógica. Para probar el modelo con nuevos datos se puede hacer uso del archivo "predictions.ipynb". Este archivo abre un modelo previamente guardado con pickle. Pickle es una librería que te permite guardar modelos y configuraciones e importarlos para realizar evaluaciones. Además, también puedes elegir qué dataset quieres usar para la evaluación. Es un frame de pandas que cuenta con las columnas del archivo original. Este frame es procesado de manera simple para que sea compatible con lo que espera el modelo para que pueda correrse de manera correcta. Al correr el bloque de código podemos observar la predicción de la clase. En caso de que se pase una query que tenga la clase output, ésta es separada e ignorada y se usa simplemente para la validación para no influir con el modelo. Finalmente, se guardan las predicciones en un archivo CSV para la validación.
## Referencias

[1] Javeed A, Anderberg P, Ghazi AN, Noor A, Elmståhl S, Berglund JS. Breaking barriers: a statistical and machine learning-based hybrid system for predicting dementia. Front Bioeng Biotechnol. 2024 Jan 8;11:1336255. doi: 10.3389/fbioe.2023.1336255. PMID: 38260734; PMCID: PMC10801181.

## Conclusiones

Hasta ahora, el modelo que mejor ha funcionado ha sido el Random Forest, usando la configuración tomada del artículo número 3. Todavía faltan varias fases por completar y hay muchos experimentos por hacer, pero este modelo ha sido el que mejor se ha ajustado a lo que busco. En las métricas se puede ver claramente que tuvo un mejor rendimiento comparado con los otros modelos probados.

