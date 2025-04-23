import pandas as pd
import pickle
import sys
from matriz_confusion import calcular_matriz_confusion

def main():
    with open('modelo_alzheimer.pkl', 'rb') as f:
        components = pickle.load(f)

    model = components['model']
    encoder = components['encoder']
    scaler = components['scaler']
    label_encoder = components['label_encoder']
    numerical_cols = components['numerical_cols']
    categorical_cols = components['categorical_cols']

    df = pd.read_csv(sys.argv[1])

    if 'Diagnosis' in df.columns:
        diagnosis_real = df['Diagnosis'].copy()
        X = df.drop('Diagnosis', axis=1)
        tiene_diagnosis = True
    else:
        X = df.copy()
        tiene_diagnosis = False

    X_cat = encoder.transform(X[categorical_cols])
    X_num = scaler.transform(X[numerical_cols])

    feature_names = encoder.get_feature_names_out(categorical_cols)
    X_cat_df = pd.DataFrame(X_cat, columns=feature_names)
    X_num_df = pd.DataFrame(X_num, columns=numerical_cols)
    X_processed = pd.concat([X_num_df, X_cat_df], axis=1)

    y_pred_encoded = model.predict(X_processed)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    resultados = pd.DataFrame({
        'Prediccion': y_pred,
    })

    if tiene_diagnosis:
        resultados['Diagnosis_Real'] = diagnosis_real
        y_true_encoded = label_encoder.transform(diagnosis_real)

        calcular_matriz_confusion(y_true_encoded, y_pred)

    output_file = "predicciones.csv"
    resultados.to_csv(output_file, index=False)
    print(resultados.head())

if __name__ == "__main__":
    main()