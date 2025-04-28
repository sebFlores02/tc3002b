def calcular_matriz_confusion(y_true, y_pred):
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

    precision = TP / (TP + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    f1 = 2 * precision * sensitivity / (precision + sensitivity)
    numerator = (TP * TN) - (FP * FN)
    denominator = ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
    mcc = numerator / denominator

    return precision, accuracy, sensitivity, specificity, f1, mcc