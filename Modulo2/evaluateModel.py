from matriz_confusion import calcular_matriz_confusion
from sklearn.metrics import roc_curve, auc

def evaluate_model(name, model, X_test, y_test):
    # Predicciones de clase
    y_pred = model.predict(X_test)

    precision, acc, sensitivity, specificity, f1, mcc = calcular_matriz_confusion(y_test, y_pred)


    # Calcular AUC-ROC
    roc_auc = 0
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        print(f"AUC-ROC: {roc_auc:.4f}")

        return {
            'name': name,
            'accuracy': acc,
            'precision': precision,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1': f1,
            'mcc': mcc,
            'auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr
        }
    else:
        return {
            'name': name,
            'accuracy': acc,
            'precision': precision,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1': f1,
            'mcc': mcc,
            'auc': roc_auc
        }