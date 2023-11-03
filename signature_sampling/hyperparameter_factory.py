from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn import metrics

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

CLASSIFIER_FACTORY = {
    "Logistic": LogisticRegression(max_iter=5000, random_state=42),
    "RF": RandomForestClassifier(random_state=42),
    "SVM-RBF": SVC(probability=True, random_state=42),
    "SVM-Poly3": SVC(kernel="poly", probability=True, random_state=42),
    "LinearSVM": LinearSVC(max_iter=5000, dual=False, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "EBM": ExplainableBoostingClassifier(random_state=42),
}
