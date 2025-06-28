import re
import warnings


def get_scores(tableS1):
    regex = re.compile(r'_score$')
    columns = filter(regex.search, tableS1.columns.values)

    # manual removal because some colums are strings. For instance, SIFT_score is not "0.23;0.02"
    columns = filter(lambda a: a != 'SIFT_score', columns)
    columns = filter(lambda a: a != 'MutationTaster_score', columns)
    columns = filter(lambda a: a != 'FATHMM_score', columns)
    columns = filter(lambda a: a != 'PROVEAN_score', columns)
    columns = filter(lambda a: a != 'VEST3_score', columns)

    warnings.warn("Warning: SIFT_score, MutationTaster_score, FATHMM_score, PROVEAN_score and VEST3_score columns have been removed because they contain string values instead of numeric values.")

    return columns


def get_rankscores(tableS1):
    regex = re.compile(r'_rankscore$')
    columns = filter(regex.search, tableS1.columns.values)

    return columns


def predictors_group(group="top6manual"):
    manually_selected_best_six_predictors = [
        "MutationAssessor_score_rankscore",
        "M-CAP_rankscore",
        "REVEL_rankscore",
        "VEST3_rankscore",
        "MutPred_rankscore",
        "MetaSVM_rankscore"
    ]

    ret = []

    if group == "top6manual":
        ret = manually_selected_best_six_predictors

    return ret


def top_predictors_by_ks(ks_result, n=10, reverse=True, sort_by=1):
    """
    - sort_by: 0 = test score, 1 = pvalue
    """
    rank = [(k, v[sort_by]) for k, v in ks_result.items()]
    rank.sort(key=lambda x: x[1], reverse=reverse)
    top_n_rank = rank[0:n]
    top_pred = [x[0] for x in top_n_rank]
    return top_pred
