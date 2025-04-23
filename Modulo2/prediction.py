import pandas as pd
import pickle
import sys
from matriz_confusion import calcular_matriz_confusion

def main():
    # python3 prediction.py <archivo_datos.csv>

    # Cargar el modelo y componentes
    with open('modelo_alzheimer.pkl', 'rb') as f:
        components = pickle.load(f)

    model = components['model']
    encoder = components['encoder']
    scaler = components['scaler']
    label_encoder = components['label_encoder']
    numerical_cols = components['numerical_cols']
    categorical_cols = components['categorical_cols']

    # Cargar datos
    df = pd.read_csv(sys.argv[1])

    # Separar diagnosis si existe
    if 'Diagnosis' in df.columns:
        diagnosis_real = df['Diagnosis'].copy()
        X = df.drop('Diagnosis', axis=1)
        tiene_diagnosis = True
    else:
        X = df.copy()
        tiene_diagnosis = False

    # Preprocesar datos
    X_cat = encoder.transform(X[categorical_cols])
    X_num = scaler.transform(X[numerical_cols])

    # Convertir a DataFrames y combinar
    feature_names = encoder.get_feature_names_out(categorical_cols)
    X_cat_df = pd.DataFrame(X_cat, columns=feature_names)
    X_num_df = pd.DataFrame(X_num, columns=numerical_cols)
    X_processed = pd.concat([X_num_df, X_cat_df], axis=1)

    # Hacer predicciones
    y_pred_encoded = model.predict(X_processed)
    y_pred_proba = model.predict_proba(X_processed)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    # Crear resultados
    resultados = pd.DataFrame({
        'Prediccion': y_pred,
    })

    # Evaluar si hay diagnóstico real
    if tiene_diagnosis:
        resultados['Diagnosis_Real'] = diagnosis_real
        y_true_encoded = label_encoder.transform(diagnosis_real)

        calcular_matriz_confusion(y_true_encoded, y_pred)

    # Guardar resultados
    output_file = "predicciones.csv"
    resultados.to_csv(output_file, index=False)
    print(resultados.head())

if __name__ == "__main__":
    main()