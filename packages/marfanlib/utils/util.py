def which(x):
    return [index for index, item in enumerate(x) if item]


def which_max(x):
    x_max_bool = [v == max(x) for v in x]
    return which(x_max_bool)


def which_min(x):
    x_min_bool = [v == min(x) for v in x]
    return which(x_min_bool)


def which_count(x):
    return len(which(x))


def to_binary(x):
    return x.values.astype(int)


def setdiff(a, b):
    return list(set(a) - set(b))
