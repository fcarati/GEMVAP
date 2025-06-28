import pandas as pd


def cohorts(tableS1, missense_variants=None, log=False):
    """

    :param tableS1: panda DataFrame representing the tableS1
    :param missense_variants:
    :param log:
    :return:
    """

    patho1 = tableS1["HGMD"] == 1
    patho2 = tableS1["UMD"] == 1
    patho3 = tableS1["FRANKEN"] == 1
    patho4 = tableS1["MUTDB"] == 1
    gnomad1 = tableS1["GNOMAD_EXOMES"] == 1
    gnomad2 = tableS1["GNOMAD_GENOMES"] == 1

    intersection_boolean = (patho1 | patho2 | patho3 | patho4) & (gnomad1 | gnomad2)
    if missense_variants is not None:
        intersection_boolean = intersection_boolean & tableS1["cDNA"].isin(missense_variants)
    intersection_binary = intersection_boolean.values.astype(int)
    intersection = tableS1[intersection_boolean]

    pathoDB_boolean = (patho1 | patho2 | patho3 | patho4) & ~(gnomad1 | gnomad2)
    if missense_variants is not None:
        pathoDB_boolean = pathoDB_boolean & tableS1["cDNA"].isin(missense_variants)
    pathoDB_binary = pathoDB_boolean.values.astype(int)
    pathoDB = tableS1[pathoDB_boolean]

    gnomAD_boolean = (gnomad1 | gnomad2) & ~(patho1 | patho2 | patho3 | patho4)
    if missense_variants is not None:
        gnomAD_boolean = gnomAD_boolean & tableS1["cDNA"].isin(missense_variants)
    gnomAD_binary = gnomAD_boolean.values.astype(int)
    gnomAD = tableS1[gnomAD_boolean]

    if log:
        print("Table of databases lengths:")
        print("UMD:\t\t\t{}\nHGMD:\t\t\t{}\nFRANKEN:\t\t{}\nMUTDB:\t\t\t{}\nGNOMADEX:\t\t{}\nGNOMADGEN:\t\t{}" \
            .format(len(tableS1[patho2]), len(tableS1[patho1]), len(tableS1[patho3]), len(tableS1[patho4]),
                    len(tableS1[gnomad1]), len(tableS1[gnomad2])))

    ret = {
        'case': {
            'table': pathoDB,
            'boolean': pathoDB_boolean,
            'binary': pathoDB_binary
        },
        'control': {
            'table': gnomAD,
            'boolean': gnomAD_boolean,
            'binary': gnomAD_binary
        },
        'intersection': {
            'table': intersection,
            'boolean': intersection_boolean,
            'binary': intersection_binary
        }
    }

    return ret
