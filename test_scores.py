# -*- coding: utf-8 -*-
from sklearn.metrics import accuracy_score, matthews_corrcoef, balanced_accuracy_score, confusion_matrix, roc_auc_score, average_precision_score

import numpy as np


def calculate_performace(y_prob, y_test):
    predicted_labels = np.round(y_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels).ravel()
    BACC = balanced_accuracy_score(y_test, predicted_labels)
    AUC = roc_auc_score(y_test, y_prob)
    AUPR = average_precision_score(y_test, y_prob)
    MCC = matthews_corrcoef(y_test, predicted_labels)
    recall = tp / (tp + fn)
    precision   = tp / (tp + fp)
    specificity = tn / (tn + fp)
    f1_score = 2*precision*recall / (precision + recall)
    return tp, fp, tn, fn, BACC, MCC, f1_score, AUC, AUPR


def get_average_metrics(metrics):
    metrics = np.array(metrics)
    return np.average(metrics, axis=0).tolist()
