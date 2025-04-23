def calcular_matriz_confusion(y_true, y_pred):
    """Calcula la matriz de confusión y métricas"""
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            if y_true[i] == 1:
                TP += 1
            else:
                FP += 1
        else:
            if y_true[i] == 0:
                TN += 1
            else:
                FN += 1

    print('         ', 'label neg ', ' label pos')
    print('pred neg    ', TN, "        ", FN)
    print('pred pos    ', FP, "        ", TP)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2*precision*recall / (precision + recall) if (precision + recall) > 0 else 0

    print("TP:", TP)
    print("TN:", TN)
    print("FP:", FP)
    print("FN:", FN)
    print("Precisión:", precision)
    print("Recall:", recall)
    print("F1:", f1)

