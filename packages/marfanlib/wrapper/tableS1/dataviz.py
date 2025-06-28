import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from marfanlib.util.util import which


def plot_pca(pd, case_boolean, control_boolean, intersection_boolean, alpha=0.3, fig_width=16, fig_height=12):
    data = pd.dropna(axis=0, how='any')
    data_patho_bin = case_boolean[data.index]
    data_control_bin = control_boolean[data.index]
    data_intersection_bin = intersection_boolean[data.index]

    pca = PCA(n_components=2)
    pca.fit(data)
    datared = pca.transform(data)

    dx = [x[0] for x in datared]
    dy = [x[1] for x in datared]

    dx_patho = [dx[x] for x in which(data_patho_bin)]
    dy_patho = [dy[x] for x in which(data_patho_bin)]

    dx_control = [dx[x] for x in which(data_control_bin)]
    dy_control = [dy[x] for x in which(data_control_bin)]

    dx_intersection = [dx[x] for x in which(data_intersection_bin)]
    dy_intersection = [dy[x] for x in which(data_intersection_bin)]

    if fig_width is not None and fig_height is not None:
        plt.rcParams["figure.figsize"] = [fig_width, fig_height]

    plt.title('PCA')
    plt.plot(dx_patho, dy_patho, 'ro', alpha=alpha)
    plt.plot(dx_control, dy_control, 'g^', alpha=alpha)
    plt.plot(dx_intersection, dy_intersection, 'b^', alpha=1)
    plt.ylabel('PCA1')
    plt.xlabel('PCA2')

    if fig_width is not None and fig_height is not None:
        plt.show()


def plot_tsne(pd, case_boolean, control_boolean, intersection_boolean, seed=42, alpha=0.3, fig_width=16, fig_height=12):
    data = pd.dropna(axis=0, how='any')
    data_patho_bin = case_boolean[data.index]
    data_control_bin = control_boolean[data.index]
    data_intersection_bin = intersection_boolean[data.index]

    datared = TSNE(n_components=2, random_state=seed, n_iter=1000).fit_transform(data)

    dx = [x[0] for x in datared]
    dy = [x[1] for x in datared]

    dx_patho = [dx[x] for x in which(data_patho_bin)]
    dy_patho = [dy[x] for x in which(data_patho_bin)]

    dx_control = [dx[x] for x in which(data_control_bin)]
    dy_control = [dy[x] for x in which(data_control_bin)]

    dx_intersection = [dx[x] for x in which(data_intersection_bin)]
    dy_intersection = [dy[x] for x in which(data_intersection_bin)]

    if fig_width is not None and fig_height is not None:
        plt.rcParams["figure.figsize"] = [fig_width, fig_height]

    plt.title('t-SNE')
    plt.plot(dx_patho, dy_patho, 'ro', alpha=alpha)
    plt.plot(dx_control, dy_control, 'g^', alpha=alpha)
    plt.plot(dx_intersection, dy_intersection, 'b^', alpha=1)
    plt.ylabel('t-SNE1')
    plt.xlabel('t-SNE2')

    if fig_width is not None and fig_height is not None:
        plt.show()
