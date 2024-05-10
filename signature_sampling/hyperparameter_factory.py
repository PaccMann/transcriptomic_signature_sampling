from interpret.glassbox import ExplainableBoostingClassifier
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

from signature_sampling.crossover_sampling import CrossoverSampling
from signature_sampling.mlp_skorch_wrapper import skorch_mlp_wrapper
from signature_sampling.sampler import BaseSampler, SMOTESampler

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
    "Logistic": LogisticRegression,
    "RF": RandomForestClassifier,
    "SVM-RBF": SVC,
    "SVM-Poly3": SVC,
    "LinearSVM": LinearSVC,
    "KNN": KNeighborsClassifier,
    "EBM": ExplainableBoostingClassifier,
    "MLP": skorch_mlp_wrapper,
}

SAMPLING_FACTORY = {
    "local_crossover": CrossoverSampling,
    "global_crossover": CrossoverSampling,
    "smote": SMOTESampler,
    "gamma_poisson": BaseSampler,
    "poisson": BaseSampler,
    "replacement": BaseSampler,
    "unaugmented": BaseSampler,
}
