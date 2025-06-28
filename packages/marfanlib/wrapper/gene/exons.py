from marfanlib.util.norm import norm_chr_name_single


def wrapper_gene_exons_bed(filename, sep='\t', norm_chr=False, log=True):
    """

    From the file the function retrieves only the three mandatory columns of BED files.

    :param filename: path of the bed file with exon information, eg <path>/FBN1_exons.bed
    :param sep: character representing the character that separes fields in the file
    :param log: boolean

    :return: list: list of dicts with keys 'chrom', 'start', 'end', 'width'.
    """
    gene = open(filename, 'r')
    lines = [x for x in gene]

    return _wrapper_gene_exons_bed(lines, sep=sep, norm_chr=norm_chr, log=log)


def _wrapper_gene_exons_bed(lines, sep='\t', norm_chr=False, log=True):
    """

    From the file the function retrieves only the three mandatory columns of BED files.

    :param lines: list of strings
    :param sep: character representing the character that separes fields in the file
    :param log: boolean

    :return: list: list of dicts with keys 'chrom', 'start', 'end', 'width'.
    """
    table = [line.split(sep) for line in lines]
    table = [[x.strip() for x in entry] for entry in table]

    exons = []
    size = 0

    for chrom, start, end, gene_exon in table:
        if norm_chr:
            chrom = norm_chr_name_single(chrom)

        width = int(end) - int(start)  # bed files are 0-based
        data = {
            "chrom": chrom,
            "start": int(start),
            "end": int(end),
            "width": width
        }
        exons.append(data)
        size += int(end) - int(start)

    if log:
        starts = [e["start"] for e in exons]
        ends = [e["end"] for e in exons]
        print "The gene is {} bases long".format(max(ends) - min(starts))
        print "The exons are {} bases long".format(size)

    return exons