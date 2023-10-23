import numpy as np
from numpy.typing import ArrayLike
import random
from typing import Iterable, Tuple, Callable, Dict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Colormap
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn import metrics, mixture


CLUSTERING_METHOD_FACTORY = {
    "kmeans": KMeans,
    "spectral": SpectralClustering,
    "heirarchical": AgglomerativeClustering,
}

CLUSTERING_METRIC_FACTORY = {
    "silhoutte_avg": metrics.silhouette_score,
    "silhoutte_sample": metrics.silhouette_samples,
    "ari": metrics.adjusted_rand_score,
    "ami": metrics.adjusted_mutual_info_score,
    "nmi": metrics.normalized_mutual_info_score,
    "homogeneity": metrics.homogeneity_score,
    "completeness": metrics.completeness_score,
    "vmeasure": metrics.v_measure_score,
}


class Clustering:
    """Class for clustering analysis."""

    def __init__(self):
        pass

    def compute_metric(
        self,
        true_labels: ArrayLike,
        pred_labels: ArrayLike,
        metric: Dict = {
            "ari": True,
            "nmi": True,
            "ami": True,
            "homogeneity": True,
            "completeness": True,
            "vmeasure": True,
        },
    ) -> Dict:
        """Computes clustering metric when true labels are known.

        Args:
            true_labels (ArrayLike): True labels of data points.
            pred_labels (ArrayLike): Predicted cluster labels from a clustering algorithm.
            metric (dict, optional): Dictionary of metrics to use for evaluation. If value is False, then metric will not be computed.
                Defaults to { "ari": True, "nmi": True, "ami": True, "homogeneity": True, "completeness": True, "vmeasure": True}.

        Returns:
            Dict: A dictionary of computed metric values where the keys are the metrics.
        """

        scores = {k: [] for k in metric.keys()}

        for key, value in metric.items():
            if value:
                scores[key].append(
                    CLUSTERING_METRIC_FACTORY[key](true_labels, pred_labels)
                )

        return scores

    def cluster_model_selection(
        self,
        data: ArrayLike,
        true_labels: ArrayLike,
        method: str,
        n_clusters: Iterable,
        metric: Dict = {
            "ari": True,
            "nmi": True,
            "ami": True,
            "homogeneity": True,
            "completeness": True,
            "vmeasure": True,
        },
        save_here: str = "clustering_scores.svg",
        **kwargs,
    ) -> None:
        """Plots a graph of number of clusters vs clustering score for model selection.

        Args:
            data (ArrayLike): Data to be clustered.
            true_labels (ArrayLike): True labels associated with the data.
            method (str): Clustering method to use. See CLUSTERING_METHOD_FACTORY for options.
            n_clusters (Iterable): Set of number of clusters to evaluate.
            metric (Dict, optional): Dictionary of metrics to use for evaluation. If value is False, then metric will not be computed.
            Defaults to { "ari": True, "nmi": True, "ami": True, "homogeneity": True, "completeness": True, "vmeasure": True}.
            save_here (str, optional): Path where the model selection graph will be saved.
                Defaults to "clustering_scores.svg".
        """
        score_list = []
        for k in n_clusters:
            obj, pred_lbl, _ = self.clustering(data, method, k, **kwargs)
            score_list.append(self.compute_metric(true_labels, pred_lbl, metric))

        scores = {key: [d.get(key) for d in score_list] for key in metric.keys()}

        plt.figure(figsize=(8, 8))
        for label, values in scores.items():
            plt.plot(
                n_clusters,
                values,
                "bx-",
                label=label,
                alpha=0.9,
                lw=2,
                dashes=[random.randint(1, 6), random.randint(1, 6)],
            )
        plt.legend()
        plt.xlabel("Number of Clusters")
        plt.ylabel("Clustering Score")
        plt.savefig(save_here)

    def clustering(
        self, data: ArrayLike, method: str, n_clusters: int, **kwargs
    ) -> Tuple:
        """Clustering function.

        Args:
            data (ArrayLike): Data to be clustered.
            method (str): Method to use for clustering. See CLUSTERING_METHOD_FACTORY for options.
            n_clusters (int): Number of clusters.

        Returns:
            Tuple: Tuple of the clustering method object, the predicted clustering labels and clustering centres.
        """

        clustering_obj = CLUSTERING_METHOD_FACTORY[method](
            n_clusters=n_clusters, **kwargs
        )
        clustering_obj.fit(data)
        clustering_labels = clustering_obj.labels_
        if hasattr(clustering_obj, "cluster_centers_"):
            cluster_centres = clustering_obj.cluster_centers_
        else:
            cluster_centres = None

        return clustering_obj, clustering_labels, cluster_centres

    def kmeans_elbow(self, data: ArrayLike, n_clusters: Iterable, **kwargs) -> Callable:
        """Plots number of clusters vs Inertia, and returns the best KMeans model based on the elbow method.

        Args:
            data (ArrayLike): Data to be clustered.
            n_clusters (Iterable): Number of clusters to evaluate.

        Returns:
            Callable: Best KMeans model object.
        """

        def optimal_number_of_clusters(n_clusters: Iterable, sse: np.float32) -> int:
            """Computes the optimal number of clusters based on elbow method.

            Args:
                n_clusters (Iterable): Range of cluster numbers to evaluate.
                sse (np.float32): sum of squared errors.

            Returns:
                int: The best number of clusters.
            """
            x1, y1 = min(n_clusters), sse[0]
            x2, y2 = max(n_clusters), sse[len(sse) - 1]

            distances = []
            for idx, i in enumerate(n_clusters):
                x0 = i
                y0 = sse[idx]
                numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
                denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
                distances.append(numerator / denominator)

            return n_clusters[distances.index(max(distances))]

        sum_of_squared_distances = []
        kmeans_objs = {}

        for k in n_clusters:
            km = KMeans(init="k-means++", n_clusters=k, **kwargs)
            km = km.fit(data)
            sum_of_squared_distances.append(km.inertia_)
            kmeans_objs[k] = km

        best_k = optimal_number_of_clusters(n_clusters, sum_of_squared_distances)
        best_km = kmeans_objs[best_k]

        plt.figure(figsize=(8, 8))
        plt.plot(n_clusters, sum_of_squared_distances, "bx-")
        plt.axvline(x=best_k, color="red", linestyle="--")
        plt.xlabel("k Clusters")
        plt.ylabel("Inertia")
        plt.title("Elbow Method for Optimal k")
        plt.savefig("kmeans_elbow.png")
        plt.show()

        return best_km

    def plot_silhouette_analysis(
        self, X: ArrayLike, X_embedding: ArrayLike, n_clusters: Iterable, save_here: str
    ) -> None:
        """Plots number of clusters vs silhoutte score for silhoutte analysis. Useful
            when true labels are not known.

        Args:
            X (ArrayLike): Data to be clustered.
            X_embedding (ArrayLike): Embedding of X for the UMAP visualisation to visualise the clusters of the high dimensional data.
            n_clusters (Iterable): Range of cluster numbers to evaluate.
            save_here (str): Path where the plot will be saved.
        """
        for k in n_clusters:
            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

            ax1.set_ylim([0, len(X) + (k + 1) * 10])

            # seed of 42 for reproducibility
            km = KMeans(n_clusters=k, random_state=42)
            km = km.fit(X)
            cluster_labels = km.predict(X)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = CLUSTERING_METRIC_FACTORY["silhouette_avg"](
                X, cluster_labels
            )
            print(
                "For n_clusters =",
                k,
                "The average silhouette_score is :",
                silhouette_avg,
            )

            # Compute the silhouette scores for each sample
            sample_silhouette_values = CLUSTERING_METRIC_FACTORY["silhoutte_sample"](
                X, cluster_labels
            )

            y_lower = 10
            for i in range(k):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[
                    cluster_labels == i
                ]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                cmap = cm.get_cmap("nipy_spectral")
                color = cmap(float(i) / k)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks

            # 2nd Plot showing the actual clusters formed
            cmap = cm.get_cmap("nipy_spectral")
            silhouette_colors = cmap(cluster_labels.astype(float) / k)
            ax2.scatter(
                X_embedding[:, 0],
                X_embedding[:, 1],
                marker=".",
                s=30,
                lw=0,
                alpha=0.7,
                c=silhouette_colors,
                edgecolor="k",
            )

            ax2.set_title("The visualization of the clustered data")
            ax2.set_xlabel("UMAP 1")
            ax2.set_ylabel("UMAP 2")

            plt.suptitle(
                (
                    "Silhouette analysis for KMeans clustering on sample data "
                    "with n_clusters = %d" % n_clusters
                ),
                fontsize=14,
                fontweight="bold",
            )

            plt.savefig()

        plt.show()

    def plot_clustering(
        self,
        cluster_labels_pred: ArrayLike,
        cluster_centers: ArrayLike,
        data_embedding: ArrayLike,
        dataset: str,
        title: str,
        color_palette: Colormap,
        save_here: str,
    ) -> None:
        """Plots the clusters returned by the clustering method.

        Args:
            cluster_labels_pred (ArrayLike): Cluster labels as predicted by the clustering
                method.
            cluster_centers (ArrayLike): Cluster centers as predicted by the clustering
                method.
            data_embedding (ArrayLike): Embedding of the data (if high-dimensional) to
                use to plot a 2D plot.
            dataset (str): Name of the dataset used in the analysis.
            title (str): Title to use for the plot. Will appear in the plot.
            color_palette (Colormap): Colour map to use for the clusters.
            save_here (str): Path where the plot will be saved.
        """
        # to get cluster centers for methods like gmm do the following:
        # for label in np.unique(labels_pred):
        #    cluster_centers.append(X.loc[labels_pred == label].mean(axis=0))
        # for kmeans do km.cluster_centres_

        plt.figure(figsize=(5, 5), constrained_layout=True)

        n_clusters = len(set(cluster_labels_pred))
        # keeping cluster centre to maybe plot it, but also ok to remove it
        if color_palette is None:
            cmap = cm.get_cmap("nipy_spectral")
            color_palette = cmap(cluster_labels_pred.astype(float) / n_clusters)
        for k, col in zip(range(n_clusters), color_palette):
            my_members = cluster_labels_pred == k
            # cluster_center = cluster_centers[k]
            plt.plot(
                data_embedding[my_members, 0],
                data_embedding[my_members, 1],
                "w",
                markerfacecolor=col,
                marker=".",
                markersize=10,
                label=k,
            )
        plt.title(title + f"Clustering of {dataset} Dataset")
        # plt.xticks(())
        # plt.yticks(())
        # plt.subplots_adjust(hspace=0.35, bottom=0.02)
        plt.legend()
        plt.savefig(save_here)
        plt.show()

    def gaussian_mixture_model(
        self, X: ArrayLike, n_components: Iterable, cv_type: str
    ) -> Tuple:
        """Runs a Gaussian mixture model (GMM) on the data provided.

        Args:
            X (ArrayLike): Data to be clustered by GMM.
            n_components (Iterable): Range of mixture components to evaluate to pick
                the best performing.
            cv_type (str): Covariance type. One of ['tied', 'full']. Refer GMM
                documentation for more details

        Returns:
            Tuple: Tuple with the best performing GMM, and the Akaike and Bayesian
                information criteria (AIC, BIC) for all components.
        """
        # gmm.fit(X) takes ages so use X_PCA or similar
        # cv_types = ['tied', 'full']

        lowest_bic = np.infty
        lowest_aic = np.infty
        bic = []
        aic = []

        for n in n_components:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(
                n_components=n, covariance_type=cv_type, random_state=42
            )

            gmm.fit(X)

            bic_ = gmm.bic(X)
            aic_ = gmm.aic(X)

            if bic_ < lowest_bic:
                lowest_bic = bic_
                best_gmm = gmm

            if aic_ < lowest_aic:
                lowest_aic = aic_

            bic.append(bic_)
            aic.append(aic_)

        return best_gmm, aic, bic

    def plot_gmm_score(
        self,
        x: ArrayLike,
        y: ArrayLike,
        title: str = "BIC score per model",
        save_here: str = "gmm_bic.svg",
    ) -> None:
        """Plots the AIC or BIC score associated against the number of components used
            for the GMM.

        Args:
            x (ArrayLike): Range of number of components used for the GMM.
            y (ArrayLike): AIC/BIC or any other score used to evaluate the number of
                components for each value.
            title (str, optional): Title to use for the plot. Defaults to "BIC score per model".
            save_here (str, optional): Path where the plot will be saved. Defaults to "gmm_bic.svg".
        """
        plt.figure(figsize=(8, 8))
        plt.bar(x, y, width=0.2)
        plt.xticks(x)
        plt.ylim([y.min() * 1.01 - 0.01 * y.max(), y.max()])
        plt.title(title)

        xpos = np.mod(y.argmin(), len(x)) + 0.65 + 0.2 * np.floor(y.argmin() / len(x))
        plt.text(xpos, y.min() * 0.97 + 0.03 * y.max(), "*", fontsize=14)
        plt.xlabel("Number of components")
        plt.savefig(save_here)
