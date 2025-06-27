import pandas as pd
from sklearn import feature_selection
from marfanlib.util.util import setdiff

# This code defines a function called "mrmr" which performs a feature selection technique called Minimum Redundancy Maximum Relevance (mRMR).

def mrmr(X, Y, n=3, seed=42):
    x = X
    y = Y
    s = []  # index of selected variables
    s_score = []
    ncols = x.shape[1]
    mic = feature_selection.mutual_info_classif(X=x, y=y, random_state=seed)
    for i in range(n):
        candidate_variables = setdiff(range(ncols), s)
        cvs = {}  # candidate_variables_scores
        for j in candidate_variables:
            mic_j = mic[j]
            mif_j = 0
            coeff = 0

            if len(s) > 0:
                mif_j = feature_selection.mutual_info_regression(X=x[x.columns[s]], y=x[x.columns[j]], random_state=seed).sum()
                coeff = 1.0/len(s)

            score = mic_j + coeff * mif_j
            cvs[j] = score

        cvs_sorted = sorted(cvs.items(), reverse=True)
        selected = cvs_sorted[0]

        selected_index = selected[0]
        selected_score = selected[1]

        s += [selected_index]
        s_score += [selected_score]

    s_columns = x.columns[s]

    data = pd.DataFrame(data={
        "iter": [i+1 for i in range(n)],
        "index": s,
        "name": s_columns,
        "score": s_score
    })

    return data
