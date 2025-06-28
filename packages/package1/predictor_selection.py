import math
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import os.path
import pandas as pd
import re
import sys
import warnings

from sklearn.metrics import roc_curve, auc
from scipy import stats
from math import log, isnan
from IPython.display import display
from plotly.graph_objs import Figure, Bar, Layout, Scatter
import matplotlib.pyplot as plt

# Add packages path to system path if not already present
packages_path = os.getcwd() + '/packages/'
if not packages_path in sys.path:
    sys.path.append(packages_path)

from marfanlib.performance.classification import predict, percentile_based_classification, roc_based_classification, prediction_matrix
from marfanlib.performance.performance import confusion_matrix, confusion_matrix_metrics, to_discrete_binary
from marfanlib.performance.consensus import consensus_stats, consensus_incremental, consensus_incremental_bis, consensus_confusion_matrix
from marfanlib.performance.stats import variant_frequency, ks
from marfanlib.ml.feature_selection import mrmr
from marfanlib.util.util import which, which_count, which_max, which_min, to_binary
from marfanlib.wrapper.tableS1.cohorts import cohorts
from marfanlib.wrapper.tableS1.columns import get_scores, get_rankscores, predictors_group, top_predictors_by_ks
from marfanlib.wrapper.tableS1.dataviz import plot_pca, plot_tsne
from marfanlib.wrapper.tableS1.helper import is_case_only_names, is_control_only_names, is_intersection_names, is_missense, is_left, is_cysteine_mutated, is_by_column
from marfanlib.wrapper.tableS1.plots.ks import ks_barchart
from marfanlib.wrapper.tableS1.plots.distribution import boxplots_groups
from marfanlib.util.util import which_min, which_count
from marfanlib.performance.performance import confusion_matrix_metrics, roc01, count_split_threshold

class DataProcessor:
    def __init__(self, data_path, seed=42):
        self.data_path = data_path
        self.seed = seed
        self.data = None
        self.is_cys = None
        self.is_case = None
        self.is_ctrl = None
        self.is_inte = None
        self.pathogenic = ["HGMD", "UMD", "FRANKEN", "MUTDB", "PARIS", "GENT"]
        self.control = ["GNOMAD_EXOMES", "GNOMAD_GENOMES"]
        
        # Set seed for reproducibility
        random.seed(seed)
        
    
    def read_data(self):
        # Read in data
        self.data = pd.read_csv(self.data_path, sep='\t', low_memory=False, na_values=['.'])
    
    def create_filters(self):
        # Filter data by criteria
        self.is_cys = is_cysteine_mutated(self.data)
        self.is_case = is_case_only_names(self.data, self.pathogenic, self.control)
        self.is_ctrl = is_control_only_names(self.data, self.pathogenic, self.control)
        self.is_inte = is_intersection_names(self.data, self.pathogenic, self.control)
        self.is_left = is_left(self.data, self.pathogenic, self.control)
        self.is_denovo = self.data["DENOVO"] == 1
        self.training_sets = (self.data["ESP"] == 1) | (self.data["EXAC"] == 1) | (self.data["1000G"] == 1) | (self.data["HGMD_2016"] == 1)
        self.is_mis = self.data["Consequence"] == "missense_variant"
        
    def top_pred_cons(self, top_predictors, th):
        self.data["consensus_score"] = self.data.apply(lambda row: sum([1 if (row.loc[tp]>=th[tp]) 
                                                                        else 0 for tp in top_predictors]), axis=1)
        self.data["top_predictor_count"] = self.data.apply(lambda row: sum([row.loc[tp] 
                                                                            for tp in top_predictors]), axis=1)
        self.data["consensus_score_norm"] = self.data.apply(lambda row: row.consensus_score*
                                        1.0/row.top_predictor_count if row.top_predictor_count else 0, axis=1)
        self.data["consensus_score_scaled"] = self.data.apply(lambda row:
                                         np.rint(row.consensus_score_norm*len(top_predictors)), axis=1)
        
        return self.data['consensus_score']
    
    def aa_stack_bar(self, used_filter):
        # Amino acid stacked barplot
        aa_list = sorted(self.data[used_filter]["aaref"].unique())
        pathoabs=[len(self.data[used_filter & self.is_case & (self.data['aaref']==x)]) for x in aa_list]
        controlabs=[len(self.data[used_filter & self.is_ctrl & (self.data['aaref']==x)]) for x in aa_list]
        interabs=[len(self.data[used_filter & self.is_inte & (self.data['aaref']==x)]) for x in aa_list]
        ratio=[len(self.data[used_filter & self.is_case & (self.data['aaref']==x)])/
               (0.00000001+len(self.data[used_filter & self.is_ctrl & (self.data['aaref']==x)])) for x in aa_list]

        d = {"AAlist": aa_list, "pathocount": pathoabs, "intersectcount": interabs, "controlcount": controlabs, "ratio": ratio}
        aa_pd = pd.DataFrame(d)
        aa_pd.sort_values(by="ratio", ascending=False, inplace=True)
        return aa_pd
    
    
def predictors_computation(df, used_filter, it):
    """
    Compute predictors from the given dataframe using the ks() function and
    return a sorted dictionary of p-values and their associated predictors,
    as well as the list of predictors used.

    Parameters:
    df (pandas.DataFrame): A dataframe containing sample data.

    Returns:
    A tuple of three values: 
    - A dictionary of p-values and their associated predictors, sorted by p-value
    - A list of predictors used
    - A dictionary of KS statistic and p-value pairs for each predictor
    """
    # loading predictors
    scorelist = [x for x in df.data.columns.tolist() if re.search("rankscore", x)]
    predictors = scorelist
    
    # computing KS statistics
    ks_case_ctrl_bf = ks(
    df.data[df.is_case & used_filter],
    df.data[df.is_ctrl & used_filter],
    predictors)
    
    # extracting p-values and sorting them by ascending order
    data = {k: v[1] for (k, v) in ks_case_ctrl_bf.items()}
    data = dict(sorted(data.items(), key=lambda item: item[1]))
    if it == 2:
        predictors = [x for x in predictors if x not in ["ClinPred_Score","MVP_rankscore"]]
        del ks_case_ctrl_bf["MVP_rankscore"]
    return data, predictors, ks_case_ctrl_bf

        
def minimised_predictor_list(predictor_list, to_remove, ks_case_ctrl_bf, df, filtering):
    ordered_predictors_by_ks = top_predictors_by_ks(ks_case_ctrl_bf, n=len(predictor_list), reverse=True, sort_by=0)
    mrmr_filt = (df.is_case | df.is_ctrl) & filtering
    mrmr_X = df.data[mrmr_filt][predictor_list]
    mrmr_nan_filt = mrmr_X.isnull().sum(1) <= 0
    mrmr_X = mrmr_X[mrmr_nan_filt]
    mrmr_Y = df.is_case[which(mrmr_filt & mrmr_nan_filt)]
    mrmr_res = mrmr(mrmr_X, mrmr_Y, mrmr_X.shape[1], seed=42)  # use this one for the full report
    ordered_predictors_by_mrmr = list(mrmr_res["name"])
    return ordered_predictors_by_mrmr, ordered_predictors_by_ks


def consensus_incremental_data(a, b, ordered_predictors, th, bi, binary):
    if binary == True:
        ci = consensus_incremental_bis(a, b, ordered_predictors, th, bi)
    if binary == False:
        ci = consensus_incremental(a, b, ordered_predictors, th, bi)
    ci_metrics = ci["stats"][ci["stats"]["idx"] == ci["roc01"]]
    ci_npreds = ci_metrics["npreds"].values[0]
    ci_cons = ci_metrics["consensuses"].values[0]
    top = ordered_predictors[0:ci_npreds]
    return {"ci": ci, "npreds": ci_npreds, "cons": ci_cons, "top": top, "metrics": ci_metrics}


def build_trace(x, thresholds, name,y_percentage, textposition="outside", binarization="greater" , text=True):
    pm = prediction_matrix(x, {k: thresholds[k] for k in x.columns}, binarization)
    consensus = pm.sum(axis=1)
    consensuses = range(len(thresholds)+1)
    
    y_value = [which_count(consensus == l) for l in consensuses]
    if y_percentage:
        y_value_trans = [0 for v in y_value]
        if x.shape[0] > 0:
            y_value_trans = [round(v/float(x.shape[0]), 2) for v in y_value]
        y_value = y_value_trans
        
    return y_value