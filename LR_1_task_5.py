import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, accuracy_score, recall_score,
                             precision_score, f1_score, roc_curve,
                             roc_auc_score)

df = pd.read_csv('data_metrics.csv')
print(df.head())

thresh = 0.5
df['predicted_RF'] = (df.model_RF >= thresh).astype('int')
df['predicted_LR'] = (df.model_LR >= thresh).astype('int')
print(df.head())

y_true = df.actual_label.values
y_RF = df.predicted_RF.values
y_LR = df.predicted_LR.values

# ---------- Матриця помилок ----------
print(confusion_matrix(y_true, y_RF))


def find_TP(y_true, y_pred):
    # істинно позитивні (y_true = 1, y_pred = 1)
    return np.sum((y_true == 1) & (y_pred == 1))


def find_FN(y_true, y_pred):
    # хибно негативні (y_true = 1, y_pred = 0)
    return np.sum((y_true == 1) & (y_pred == 0))


def find_FP(y_true, y_pred):
    # хибно позитивні (y_true = 0, y_pred = 1)
    return np.sum((y_true == 0) & (y_pred == 1))


def find_TN(y_true, y_pred):
    # істинно негативні (y_true = 0, y_pred = 0)
    return np.sum((y_true == 0) & (y_pred == 0))


print('TP:', find_TP(y_true, y_RF))
print('FN:', find_FN(y_true, y_RF))
print('FP:', find_FP(y_true, y_RF))
print('TN:', find_TN(y_true, y_RF))


def find_conf_matrix_values(y_true, y_pred):
    TP = find_TP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    TN = find_TN(y_true, y_pred)
    return TP, FN, FP, TN


def varchenko_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])


print(varchenko_confusion_matrix(y_true, y_RF))

assert np.array_equal(varchenko_confusion_matrix(y_true, y_RF),
                      confusion_matrix(y_true, y_RF)), \
    'varchenko_confusion_matrix() is not correct for RF'
assert np.array_equal(varchenko_confusion_matrix(y_true, y_LR),
                      confusion_matrix(y_true, y_LR)), \
    'varchenko_confusion_matrix() is not correct for LR'

# ---------- Accuracy ----------
print(accuracy_score(y_true, y_RF))


def varchenko_accuracy_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)


assert np.isclose(varchenko_accuracy_score(y_true, y_RF),
                  accuracy_score(y_true, y_RF)), \
    'varchenko_accuracy_score failed on RF'
assert np.isclose(varchenko_accuracy_score(y_true, y_LR),
                  accuracy_score(y_true, y_LR)), \
    'varchenko_accuracy_score failed on LR'
print('Accuracy RF: %.3f' % varchenko_accuracy_score(y_true, y_RF))
print('Accuracy LR: %.3f' % varchenko_accuracy_score(y_true, y_LR))

# ---------- Recall ----------
print(recall_score(y_true, y_RF))


def varchenko_recall_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FN)


assert np.isclose(varchenko_recall_score(y_true, y_RF),
                  recall_score(y_true, y_RF)), \
    'varchenko_recall_score failed on RF'
assert np.isclose(varchenko_recall_score(y_true, y_LR),
                  recall_score(y_true, y_LR)), \
    'varchenko_recall_score failed on LR'
print('Recall RF: %.3f' % varchenko_recall_score(y_true, y_RF))
print('Recall LR: %.3f' % varchenko_recall_score(y_true, y_LR))

# ---------- Precision ----------
print(precision_score(y_true, y_RF))


def varchenko_precision_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FP)


assert np.isclose(varchenko_precision_score(y_true, y_RF),
                  precision_score(y_true, y_RF)), \
    'varchenko_precision_score failed on RF'
assert np.isclose(varchenko_precision_score(y_true, y_LR),
                  precision_score(y_true, y_LR)), \
    'varchenko_precision_score failed on LR'
print('Precision RF: %.3f' % varchenko_precision_score(y_true, y_RF))
print('Precision LR: %.3f' % varchenko_precision_score(y_true, y_LR))

# ---------- F1 ----------
print(f1_score(y_true, y_RF))


def varchenko_f1_score(y_true, y_pred):
    recall = varchenko_recall_score(y_true, y_pred)
    precision = varchenko_precision_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)


assert np.isclose(varchenko_f1_score(y_true, y_RF),
                  f1_score(y_true, y_RF)), \
    'varchenko_f1_score failed on RF'
assert np.isclose(varchenko_f1_score(y_true, y_LR),
                  f1_score(y_true, y_LR)), \
    'varchenko_f1_score failed on LR'
print('F1 RF: %.3f' % varchenko_f1_score(y_true, y_RF))
print('F1 LR: %.3f' % varchenko_f1_score(y_true, y_LR))

# ---------- Порівняння порогів ----------
y_RF_025 = (df.model_RF >= 0.25).astype('int').values

print('\nscores with threshold = 0.5')
print('Accuracy RF: %.3f' % varchenko_accuracy_score(y_true, y_RF))
print('Recall RF: %.3f' % varchenko_recall_score(y_true, y_RF))
print('Precision RF: %.3f' % varchenko_precision_score(y_true, y_RF))
print('F1 RF: %.3f' % varchenko_f1_score(y_true, y_RF))

print('\nscores with threshold = 0.25')
print('Accuracy RF: %.3f' % varchenko_accuracy_score(y_true, y_RF_025))
print('Recall RF: %.3f' % varchenko_recall_score(y_true, y_RF_025))
print('Precision RF: %.3f' % varchenko_precision_score(y_true, y_RF_025))
print('F1 RF: %.3f' % varchenko_f1_score(y_true, y_RF_025))

# ---------- ROC-криві ----------
fpr_RF, tpr_RF, thresholds_RF = roc_curve(y_true, df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(y_true, df.model_LR.values)

plt.plot(fpr_RF, tpr_RF, 'r-', label='RF')
plt.plot(fpr_LR, tpr_LR, 'b-', label='LR')
plt.plot([0, 1], [0, 1], 'k-', label='random')
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

# ---------- AUC ----------
auc_RF = roc_auc_score(y_true, df.model_RF.values)
auc_LR = roc_auc_score(y_true, df.model_LR.values)
print('AUC RF: %.3f' % auc_RF)
print('AUC LR: %.3f' % auc_LR)

plt.plot(fpr_RF, tpr_RF, 'r-', label='RF AUC: %.3f' % auc_RF)
plt.plot(fpr_LR, tpr_LR, 'b-', label='LR AUC: %.3f' % auc_LR)
plt.plot([0, 1], [0, 1], 'k-', label='random')
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()