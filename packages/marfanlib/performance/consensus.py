import pandas as pd
from marfanlib.util.util import which_min, which_count
from marfanlib.performance.performance import confusion_matrix_metrics, roc01, count_split_threshold
from marfanlib.performance.classification import prediction_matrix


def consensus_stats(a, b, thresholds, binarization, y_percentage=False):
    a_stats = consensus_confusion_matrix(a, thresholds, binarization, y_percentage)
    b_stats = consensus_confusion_matrix(b, thresholds, binarization, y_percentage)
    npreds = len(thresholds)

    stats = a_stats.join(b_stats, on="consensuses", lsuffix='_a', rsuffix='_b')
    stats = stats.rename(index=str, columns={
        "consensuses_a": "consensuses",
        "fp_a": "fp",
        "tp_a": "tp",
        "fp_b": "tn",
        "tp_b": "fn"
    })
    stats = stats[["consensuses", "tp", "fp", "tn", "fn"]]

    cmm = pd.DataFrame(data=[dict(
        confusion_matrix_metrics(
            stats["tp"][i],
            stats["fp"][i],
            stats["tn"][i],
            stats["fn"][i]),
        **{"consensuses": i, "npreds": npreds}) for i in stats["consensuses"]])

    ret = {
        'stats': cmm,
        'roc01': which_min(roc01(cmm["tpr"], cmm["fpr"]))[0]
    }

    return ret

def calculate_f1_score(precision, recall):
    f1_score = []
    for a in range(precision.shape[0]):
        if (precision[a] + recall[a]) == 0:
            f1_score.append(0)  # To handle the case when precision + recall is zero to avoid division by zero.
        else:
            f1_score.append(2 * (precision[a] * recall[a]) / (precision[a] + recall[a]))
    return f1_score
    
    
def consensus_stats_bis(a, b, thresholds, binarization, y_percentage=False):
    a_stats = consensus_confusion_matrix(a, thresholds, binarization, y_percentage)
    b_stats = consensus_confusion_matrix(b, thresholds, binarization, y_percentage)
    npreds = len(thresholds)

    stats = a_stats.join(b_stats, on="consensuses", lsuffix='_a', rsuffix='_b')
    stats = stats.rename(index=str, columns={
        "consensuses_a": "consensuses",
        "fp_a": "fp",
        "tp_a": "tp",
        "fp_b": "tn",
        "tp_b": "fn"
    })
    stats = stats[["consensuses", "tp", "fp", "tn", "fn"]]

    cmm = pd.DataFrame(data=[dict(
        confusion_matrix_metrics(
            stats["tp"][i],
            stats["fp"][i],
            stats["tn"][i],
            stats["fn"][i]),
        **{"consensuses": i, "npreds": npreds}) for i in stats["consensuses"]])
    ret = {
        'stats': cmm,
        'roc01': which_min([x*-1 for x in calculate_f1_score(cmm["ppv"], cmm["tpr"])])
    }

    return ret

def consensus_incremental(a, b, ordered_predictors, thresholds, binarization, y_percentage=False):
    '''
    The function then computes a consensus classifier for each incremental subset of predictors in ordered_predictors, and aggregates the   performance statistics for each consensus classifier. The function returns a dictionary containing the incremental_predictors dictionary, which contains the performance statistics for each incremental subset of predictors, a pandas dataframe stats containing the performance statistics for each combination of predictors and their consensus, and an integer roc01 representing the index of the best consensus as determined by ROC01 score.
    '''
    incremental_predictors = {}
    bests = []
    for i in range(len(ordered_predictors)):
        selected_predictors = ordered_predictors[0:i+1]
        consensus_i_performance = consensus_stats(
            a[selected_predictors],
            b[selected_predictors],
            {k: thresholds[k] for k in selected_predictors},
            binarization,
            y_percentage
        )
        stats = consensus_i_performance["stats"]
        r01 = consensus_i_performance["roc01"]
        consensus_best = stats[stats["consensuses"] == r01]
        bests += [consensus_best]
        incremental_predictors[i+1] = consensus_i_performance
    bests = pd.concat(bests)
    bests = bests.assign(idx=pd.Series(range(len(selected_predictors))).values)

    
    ret = {
        'incremental_predictors': incremental_predictors,
        'stats': bests,
        'roc01': which_min(roc01(list(bests["tpr"]), list(bests["fpr"])))[0]
    }

    return ret


def consensus_incremental_bis(a, b, ordered_predictors, thresholds, binarization, y_percentage=False):
    '''
    The function then computes a consensus classifier for each incremental subset of predictors in ordered_predictors, and aggregates the   performance statistics for each consensus classifier. The function returns a dictionary containing the incremental_predictors dictionary, which contains the performance statistics for each incremental subset of predictors, a pandas dataframe stats containing the performance statistics for each combination of predictors and their consensus, and an integer roc01 representing the index of the best consensus as determined by ROC01 score.
    '''
    incremental_predictors = {}
    bests = []
    for i in range(len(ordered_predictors)):
        selected_predictors = ordered_predictors[0:i+1]
        consensus_i_performance = consensus_stats_bis(
            a[selected_predictors],
            b[selected_predictors],
            {k: thresholds[k] for k in selected_predictors},
            binarization,
            y_percentage
        )
        stats = consensus_i_performance["stats"]
        r01 = consensus_i_performance["roc01"]
        consensus_best = stats[stats["consensuses"] == r01[0]]
        bests += [consensus_best]
        incremental_predictors[i+1] = consensus_i_performance

    bests = pd.concat(bests)
    bests = bests.assign(idx=pd.Series(range(len(ordered_predictors))).values)
    bests.reset_index(drop=True, inplace=True)
    bests['F1_score'] = calculate_f1_score(bests["ppv"], bests["tpr"])
    
    ret = {
        'incremental_predictors': incremental_predictors,
        'stats': bests,
        'roc01': which_min([x*-1 for x in calculate_f1_score(bests["ppv"], bests["tpr"])])[0]
    }

    return ret


def consensus_confusion_matrix(x, thresholds, binarization, y_percentage=False):
    pm = prediction_matrix(x, {k: thresholds[k] for k in x.columns}, binarization)
    consensus = pm.sum(axis=1)
    consensuses = range(len(thresholds)+1)

    y_value = [which_count(consensus == l) for l in consensuses]
    if y_percentage:
        y_value = [round(v/float(x.shape[0]), 4) for v in y_value]

    consensus_sum = [count_split_threshold(y_value, l) for l in consensuses]

    ret = pd.DataFrame(data={
        "consensuses": consensuses,
        "count": y_value,
        "tp": [k[1] for k in consensus_sum],
        "fp": [k[0] for k in consensus_sum]
    })

    return ret
