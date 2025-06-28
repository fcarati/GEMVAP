import pandas as pd


def is_by_column(x, column):
    return x[column] == 1


def is_franken(x):
    return x["FRANKEN"] == 1


def is_umd(x):
    return x["UMD"] == 1


def is_hgmd(x):
    return x["HGMD"] == 1


def is_mutdb(x):
    return x["MUTDB"] == 1


def is_gnomade(x):
    return x["GNOMAD_EXOMES"] == 1


def is_gnomadg(x):
    return x["GNOMAD_GENOMES"] == 1


def is_missense(x, missense_variants):
    return x["cDNA"].isin(missense_variants)


def is_cysteine_mutated(x):
    return x["aaref"] == "C"


def is_case_only(x):
    """
    @deprecated, please use is_case_only_names
    """
    pathogenic = is_franken(x) | is_umd(x) | is_hgmd(x) | is_mutdb(x)
    healthy = is_gnomade(x) | is_gnomadg(x)
    return pathogenic & ~healthy


def is_control_only(x):
    """
    @deprecated, please use is_control_only_names
    """
    pathogenic = is_franken(x) | is_umd(x) | is_hgmd(x) | is_mutdb(x)
    healthy = is_gnomade(x) | is_gnomadg(x)
    return healthy & ~pathogenic


def is_intersection(x):
    """
    @deprecated, please use is_intersection_names
    """
    pathogenic = is_franken(x) | is_umd(x) | is_hgmd(x) | is_mutdb(x)
    healthy = is_gnomade(x) | is_gnomadg(x)
    return healthy & pathogenic


def is_case_only_names(x, case, control):
    pathogenic = pd.DataFrame([is_by_column(x, c) for c in case]).transpose().apply(any, axis=1)
    healthy = pd.DataFrame([is_by_column(x, c) for c in control]).transpose().apply(any, axis=1)
    return pathogenic & ~healthy


def is_control_only_names(x, case, control):
    pathogenic = pd.DataFrame([is_by_column(x, c) for c in case]).transpose().apply(any, axis=1)
    healthy = pd.DataFrame([is_by_column(x, c) for c in control]).transpose().apply(any, axis=1)
    return healthy & ~pathogenic


def is_intersection_names(x, case, control):
    pathogenic = pd.DataFrame([is_by_column(x, c) for c in case]).transpose().apply(any, axis=1)
    healthy = pd.DataFrame([is_by_column(x, c) for c in control]).transpose().apply(any, axis=1)
    return healthy & pathogenic


def is_left(x, case, control):
    pathogenic = pd.DataFrame([is_by_column(x, c) for c in case]).transpose().apply(any, axis=1)
    healthy = pd.DataFrame([is_by_column(x, c) for c in control]).transpose().apply(any, axis=1)
    return ~healthy & ~pathogenic
