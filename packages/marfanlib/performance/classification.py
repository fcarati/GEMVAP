import numpy as np
import pandas as pd
from marfanlib.performance.performance import cv_roc, confusion_matrix, to_discrete_binary
from marfanlib.util.util import which, which_count, to_binary
from marfanlib.wrapper.tableS1.helper import is_case_only_names, is_control_only_names, is_intersection_names


def prediction_matrix(data, threshold_dict, binarization="greater"):
    predictors = data.columns
    pm = {p: to_discrete_binary(data[p], threshold_dict[p], binarization) for p in predictors}
    return pd.DataFrame(pm)


def predict(tableS1, bool, pred_names, performance):
    outcomes = {}
    for pred_name in pred_names:
        preds = tableS1[pred_name].values
        preds = preds[which(bool)]

        preds_notnan_boolean = [not np.isnan(x) for x in preds]
        preds = preds[which(preds_notnan_boolean)]

        outcome = [x >= performance['threshold']['case'][pred_name] for x in preds]
        positive = which_count(outcome)
        negative = len(outcome) - positive

        res = {
            'positive': positive,
            'negative': negative,
            'n': len(outcome)
        }

        outcomes[pred_name] = res

    df = pd.DataFrame.from_dict(outcomes, orient='index')
    df['thres_control'] = df.index.to_series().map(performance['threshold']['case'])
    df['pos/neg'] = df['positive'] / df['negative']

    return df


def percentile_based_classification(tableS1, pred_names, percentile):
    patho_thres = {}
    control_thres = {}
    columns_over_patho = {}
    columns_below_control = {}

    for pred_name in pred_names:
        preds = tableS1[pred_name].values
        preds_notnan_boolean = [not np.isnan(x) for x in preds]
        preds = preds[which(preds_notnan_boolean)]

        truth_patho = to_binary(is_case_only(tableS1))[which(preds_notnan_boolean)]
        truth_control = to_binary(is_control_only(tableS1))[which(preds_notnan_boolean)]

        perf_patho_threshold = np.percentile(preds, percentile)
        perf_control_threshold = np.percentile(preds, 100 - percentile)

        cm_patho = confusion_matrix(preds, truth_patho, perf_patho_threshold)
        cm_control = confusion_matrix(preds, truth_control, perf_control_threshold, binarization='less')

        patho_thres[pred_name] = perf_patho_threshold
        control_thres[pred_name] = perf_control_threshold

        columns_over_patho[pred_name] = cm_patho
        columns_below_control[pred_name] = cm_control

    df_case = pd.DataFrame.from_dict(columns_over_patho, orient='index')
    df_case['thres_case'] = df_case.index.to_series().map(patho_thres)
    df_case.columns = ['case_' + str(col) for col in df_case.columns]

    df_control = pd.DataFrame.from_dict(columns_below_control, orient='index')
    df_control['thres_control'] = df_control.index.to_series().map(patho_thres)
    df_control.columns = ['control_' + str(col) for col in df_control.columns]

    df = pd.concat([df_case, df_control], axis=1)

    ret = {
        'threshold': {
            'case': patho_thres,
            'control': control_thres
        },
        'annotation': {
            'intersection': {
                'case': columns_over_patho,
                'control': columns_below_control
            }
        },
        'df': df
    }

    return ret


def roc_based_classification(tableS1, pred_names, pathogenic, control, nfold=10):
    # Initialize empty dictionaries to store the performance metrics and confusion matrices for each predictor
    patho_thres = {}
    control_thres = {}
    columns_over_patho = {}
    columns_below_control = {}
    
    # Loop over each predictor
    for pred_name in pred_names:
        # Get the predictions and true labels for pathogenic variants
        preds = tableS1[pred_name].values
        truth_patho = is_case_only_names(tableS1, pathogenic, control).values
        
        # Compute the performance metrics for pathogenic variants using n-fold cross-validation.
        # Store the threshold with best performance
        perf_patho = cv_roc(preds, truth_patho, nfold)
        #perf_patho_threshold = max([x['threshold_train'] for x in perf_patho])
        perf_patho_threshold = sum([x['threshold_train'] for x in perf_patho])/nfold

        # Get the predictions and true labels for control variants
        truth_control = is_control_only_names(tableS1, pathogenic, control).values
        
        # Compute the performance metrics for control variants using n-fold cross-validation and store the threshold with best performance
        perf_control = cv_roc(preds, truth_control, nfold)
        #perf_control_threshold = min([x['threshold_train'] for x in perf_control])
        perf_control_threshold = sum([x['threshold_train'] for x in perf_control])/nfold

        # Store the pathogenic and control thresholds for this predictor
        patho_thres[pred_name] = perf_patho_threshold
        control_thres[pred_name] = perf_control_threshold
        
        # Compute the confusion matrices for this predictor using the pathogenic and control thresholds
        cm_patho = confusion_matrix(preds, truth_patho, perf_patho_threshold)
        cm_control = confusion_matrix(preds, truth_control, perf_control_threshold, binarization='less')

        # Store the confusion matrices for this predictor        
        columns_over_patho[pred_name] = cm_patho
        columns_below_control[pred_name] = cm_control
    
    # Create dataframes to store the confusion matrices and thresholds for each predictor
    df_case = pd.DataFrame.from_dict(columns_over_patho, orient='index')
    df_case['thres_case'] = df_case.index.to_series().map(patho_thres)
    df_case.columns = ['case_' + str(col) for col in df_case.columns]

    # Concatenate the dataframes for pathogenic and control variants
    df_control = pd.DataFrame.from_dict(columns_below_control, orient='index')
    df_control['thres_control'] = df_control.index.to_series().map(patho_thres)
    df_control.columns = ['control_' + str(col) for col in df_control.columns]

    df = pd.concat([df_case, df_control], axis=1)
    
    # Create a dictionary to store the thresholds, annotations and dataframes for each predictor
    ret = {
        'threshold': {
            'case': patho_thres,
            'control': control_thres
        },
        'annotation': {
            'intersection': {
                'case': columns_over_patho,
                'control': columns_below_control
            }
        },
        'df': df
    }

    return ret
