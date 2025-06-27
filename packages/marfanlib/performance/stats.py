import numpy as np
import pandas as pd
from math import log
from scipy import stats
from marfanlib.util.util import which


def ks(a, b, predictors):
    """
    Compute the Kolmogorov-Smirnov (KS) statistic for each predictor in the
    given dataframes a and b.
    
    Parameters:
    a (pandas.DataFrame): A dataframe containing one set of samples.
    b (pandas.DataFrame): A dataframe containing another set of samples.
    predictors (list): A list of column names to use as predictors.
    
    Returns:
    A dictionary of KS statistic and p-value pairs for each predictor.
    """
    pvalues = {}
    for score in predictors:
        if a[score].dropna().empty:
            continue
        if b[score].dropna().empty:
            continue
        res = stats.ks_2samp(
            b[score].dropna(),
            a[score].dropna()
        )
        pvalues[score] = [res[0], res[1]]

    ret = pvalues

    return ret


def variant_frequency(x, pred_names, performance):
    outcome_boolean_matrix = {}
    for pred_name in pred_names:

        # retrieve predictor threshold for both case and control
        case_threshold = performance['threshold']['case'][pred_name]
        control_threshold = performance['threshold']['control'][pred_name]

        # boolean vector of whether the value is !NA (True) or NA (False)
        notnan_boolean = ~np.isnan(x[pred_name])

        # boolean of values greater (less) than case (control) threshold
        case_boolean = x[pred_name] >= case_threshold
        control_boolean = x[pred_name] <= control_threshold

        valid_case = notnan_boolean & case_boolean
        valid_control = notnan_boolean & control_boolean

        outcome_boolean_matrix[pred_name] = {
            'case': valid_case,
            'control': valid_control
        }

    all_index_pos = []
    all_index_neg = []
    for pred_name in pred_names:
        idx_pos = which(outcome_boolean_matrix[pred_name]['case'])
        idx_neg = which(outcome_boolean_matrix[pred_name]['control'])
        all_index_pos = all_index_pos + list(idx_pos)
        all_index_neg = all_index_neg + list(idx_neg)

    count_pos = list(pd.DataFrame({'idx': all_index_pos}).groupby(['idx']).size())
    freq_pos = pd.DataFrame({'count': count_pos}).groupby(['count']).size()

    count_neg = list(pd.DataFrame({'idx': all_index_neg}).groupby(['idx']).size())
    freq_neg = pd.DataFrame({'count': count_neg}).groupby(['count']).size()

    ret = {
        'case': freq_pos,
        'control': freq_neg
    }

    return ret
