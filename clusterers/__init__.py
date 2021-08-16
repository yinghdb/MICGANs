from clusterers import (base_clusterer, random_labels, crp_clusterer)

clusterer_dict = {
    'supervised': base_clusterer.BaseClusterer,
    'random_labels': random_labels.Clusterer,
    'crp': crp_clusterer.Clusterer
}
