from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


def elbow_method(df):
    """

    :param df:
    :return:
    """
    st_scaler = StandardScaler()
    X_scaled = st_scaler.fit_transform(df[["latitude_start", "longitude_start"]])

    k_max = 10
    clusters = []
    losses = []

    # elbow method to specify the appropriate number of cluster
    for k in range(k_max):
        model = KMeans(n_clusters=k + 1)
        model.fit(X_scaled)
        clusters.append(k + 1)
        losses.append(model.inertia_)
        print(str(k+1)+'/' + str(k_max))

    plt.plot(clusters, losses)
    plt.title("Elbow method - K-means++")
    plt.show()

    return {
        'X_scaled': X_scaled
    }
