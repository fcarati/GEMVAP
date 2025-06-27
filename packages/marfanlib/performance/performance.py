import numpy as np
import pandas as pd
import math
from sklearn import metrics
from sklearn.model_selection import KFold
from marfanlib.util.util import which_min, which


def to_discrete_binary(x, threshold, binarization="greater"):
    if binarization == "greater":
        x = [0 if value < threshold else 1 for value in x]
    elif binarization == "less":
        x = [0 if value > threshold else 1 for value in x]
    else:
        raise "Error in binarization option: either 'greater' or 'less'"

    return x


def confusion_matrix_metrics(tp, fp, tn, fn):
    n = tp + fp + tn + fn

    tpr = float(tp) / (tp + fn) if tp + fn > 0 else 0
    tnr = float(tn) / (tn + fp) if tn + fp > 0 else 0
    fpr = float(fp) / (fp + tn) if fp + tn > 0 else 0
    ppv = float(tp) / (tp + fp) if tp + fp > 0 else 0
    npv = float(tn) / (tn + fn) if tn + fn > 0 else 0

    res = {
        'n': n,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tpr': tpr,  # recall / sensitivity
        'fpr': fpr,
        'npv': npv,
        'ppv': ppv,  # precision
        'tnr': tnr   # specificity
    }

    return res


def confusion_matrix(preds, truth, threshold=None, binarization="greater"):
    """
    Elaboration of metrics as of in https://en.wikipedia.org/wiki/Confusion_matrix
    """

    if threshold is not None:
        preds = to_discrete_binary(preds, threshold, binarization)

    tp = len(which([(preds[i] == 1 and truth[i] == 1) for i in range(len(preds))]))
    tn = len(which([(preds[i] == 0 and truth[i] == 0) for i in range(len(preds))]))
    fp = len(which([(preds[i] == 1 and truth[i] == 0) for i in range(len(preds))]))
    fn = len(which([(preds[i] == 0 and truth[i] == 1) for i in range(len(preds))]))

    res = confusion_matrix_metrics(tp, fp, tn, fn)

    return res


def roc01(tpr, fpr):
    return [math.sqrt(((1-tpr[i]) * (1-tpr[i])) + (fpr[i] * fpr[i])) for i in range(len(tpr))]


def cv_roc(preds, truth, nfold=10, plot=False):
    """
    Performs n-fold cross-validation to calculate the ROC curve and AUC.

    Args:
    preds (numpy array): A 1D numpy array containing predicted scores.
    truth (numpy array): A 1D numpy array containing binary truth values.
    nfold (int): The number of cross-validation folds to perform.
    plot (bool): Whether or not to plot the ROC curve.

    Returns:
    A list of dictionaries, with each dictionary containing the following keys:
        'preds_test_binary': A 1D numpy array containing the binary predicted values.
        'truth_test': A 1D numpy array containing the truth values for the test set.
        'mcc': The Matthews Correlation Coefficient (MCC) score for the test set.
        'threshold_train': The threshold value used to make binary predictions for the train set.
        'fpr_train': A 1D numpy array containing the false positive rates for the train set.
        'tpr_train': A 1D numpy array containing the true positive rates for the train set.
        'dist01': The distance to the point (0,1) on the ROC curve for the train set.
        'roc_auc_train': The Area Under the Curve (AUC) value for the train set.

    """
    # Remove NaN values
    preds_isnan_boolean = np.isnan(preds)
    preds_valid = preds[~preds_isnan_boolean]
    truth_valid = truth[~preds_isnan_boolean]

    # Initialize mean false positive rate
    mean_fpr = np.linspace(0, 1, 100)

    # Initialize KFold object    
    kf = KFold(n_splits=nfold)
    ret = []
    
    # Perform nfold cross-validation
    for train_index, test_index in kf.split(truth_valid):
        truth_valid_train = truth_valid[train_index]
        preds_valid_train = preds_valid[train_index]

        truth_valid_test = truth_valid[test_index]
        preds_valid_test = preds_valid[test_index]

        # Calculate ROC curve for train set
        fpr, tpr, thresholds = metrics.roc_curve(truth_valid_train, preds_valid_train)

        dist01 = roc01(tpr, fpr)
        threshold_train = thresholds[which_min(dist01)[0]]
        roc_auc_train = metrics.auc(fpr, tpr)
        preds_valid_test_binary = [0 if x < threshold_train else 1 for x in preds_valid_test]

        results = {'preds_test_binary': preds_valid_test_binary,
                   'truth_test': truth_valid_test,
                   'mcc': mcc(preds_valid_test_binary, truth_valid_test),
                   'threshold_train': threshold_train,
                   'fpr_train': fpr,
                   'tpr_train': tpr,
                   'dist01': dist01,
                   'roc_auc_train': roc_auc_train}

        ret.append(results)
    return ret


def mcc(preds, truth):
    cm = confusion_matrix(preds, truth)

    tp = cm['tp']
    tn = cm['tn']
    fp = cm['fp']
    fn = cm['fn']

    # print "MCC values: " + str(tp) + ", " + str(tn) + ", " + str(fp) + ", " + str(fn)

    a = tp+fp if tp+fp > 0 else 1
    b = tp+fn if tp+fn > 0 else 1
    c = tn+fp if tn+fp > 0 else 1
    d = tn+fn if tn+fn > 0 else 1

    return (tp*tn - fp*fn) / math.sqrt(a*b*c*d)


def count_split_threshold(x, idx_threshold):
    less = 0
    if idx_threshold > 0:
        less = sum(x[0:idx_threshold])  # idx_threshold index is excluded, so [0:(idx_threshold-1)]
    more = sum(x[idx_threshold:len(x)])

    return (less, more)
