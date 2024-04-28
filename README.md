# K-Means Clustering Pipeline for Gene Expression Data

This Python project implements a K-Means clustering pipeline to analyze gene expression data obtained from The Cancer Genome Atlas (TCGA) Pan-Cancer analysis project. The dataset consists of 881 samples, each representing one of five distinct cancer subtypes, with gene expression values for 20,531 genes.

## Project Overview

1. **Data Retrieval:**
   - The TCGA dataset is downloaded programmatically from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq).
   - The dataset is stored in a compressed tar file, and Python's `tarfile` module is used to extract the data.

```python
# Data Retrieval
uci_tcga_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00401/"
archive_name = "TCGA-PANCAN-HiSeq-801x20531.tar.gz"

# Build the URL
full_download_url = urllib.parse.urljoin(uci_tcga_url, archive_name)

# Download the file
r = urllib.request.urlretrieve(full_download_url, archive_name)

# Extract the data from the archive
tar = tarfile.open(archive_name, "r:gz")
tar.extractall()
tar.close()
```

2. **Data Preprocessing:**
   - Gene expression data and labels are loaded using NumPy.
   - Label encoding is performed to convert cancer type abbreviations into integers.
   - A preprocessing pipeline is constructed, including MinMaxScaler for feature scaling and PCA for dimensionality reduction.

```python
# Data Preprocessing
datafile = "TCGA-PANCAN-HiSeq-801x20531/data.csv"
labels_file = "TCGA-PANCAN-HiSeq-801x20531/labels.csv"

data = np.genfromtxt(datafile, delimiter=",", usecols=range(1, 20532), skip_header=1)
true_label_names = np.genfromtxt(labels_file, delimiter=",", usecols=(1,), skip_header=1, dtype=str)

label_encoder = LabelEncoder()
true_labels = label_encoder.fit_transform(true_label_names)
n_clusters = len(label_encoder.classes_)

# Constructing the Preprocessing Pipeline
preprocessor = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        ("pca", PCA(n_components=2, random_state=42)),
    ]
)
```

3. **K-Means Clustering:**
   - A K-Means clustering pipeline is built with user-defined arguments, overriding default values.
   - The pipeline includes K-Means with customized initialization, number of initializations, and maximum iterations.

```python
# K-Means Clustering Pipeline
clusterer = Pipeline(
    [
        (
            "kmeans",
            KMeans(
                n_clusters=n_clusters,
                init="k-means++",
                n_init=50,
                max_iter=500,
                random_state=42,
            ),
        ),
    ]
)
```

4. **Model Fitting and Evaluation:**
   - The complete pipeline is fitted to the gene expression data.
   - Evaluation metrics, such as silhouette score and adjusted Rand score, are computed.

```python
# Model Fitting and Evaluation
pipe = Pipeline([("preprocessor", preprocessor), ("clusterer", clusterer)])
pipe.fit(data)
preprocessed_data = pipe["preprocessor"].transform(data)
predicted_labels = pipe["clusterer"]["kmeans"].labels_

silhouette = silhouette_score(preprocessed_data, predicted_labels)
ari = adjusted_rand_score(true_labels, predicted_labels)
```

5. **Visualization:**
   - The results of clustering are visualized using scatter plots.
   - Principal components are plotted with predicted clusters and true labels.

```python
# Visualization
pcadf = pd.DataFrame(
    pipe["preprocessor"].transform(data),
    columns=["component_1", "component_2"],
)

pcadf["predicted_cluster"] = pipe["clusterer"]["kmeans"].labels_
pcadf["true_label"] = label_encoder.inverse_transform(true_labels)

plt.figure(figsize=(8, 8))
scat = sns.scatterplot(
    "component_1",
    "component_2",
    s=50,
    data=pcadf,
    hue="predicted_cluster",
    style="true_label",
    palette="Set2",
)
scat.set_title("Clustering results from TCGA Pan-Cancer\nGene Expression Data")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
plt.show()
```

6. **Parameter Tuning:**
   - The project includes a parameter tuning section where the number of principal components is varied.
   - Silhouette coefficients and Adjusted Rand Index (ARI) scores are computed for different numbers of components.

```python
# Parameter Tuning
silhouette_scores = []
ari_scores = []

for n in range(2, 11):
    pipe["preprocessor"]["pca"].n_components = n
    pipe.fit(data)

    silhouette_coef = silhouette_score(
        pipe["preprocessor"].transform(data),
        pipe["clusterer"]["kmeans"].labels_,
    )
    ari = adjusted_rand_score(
        true_labels,
        pipe["clusterer"]["kmeans"].labels_,
    )

    silhouette_scores.append(silhouette_coef)
    ari_scores.append(ari)

# Visualization of Parameter Tuning Results
plt.figure(figsize=(6, 6))
plt.plot(
    range(2, 11),
    silhouette_scores,
    c="#008fd5",
    label="Silhouette Coefficient",
)
plt.plot(range(2, 11), ari_scores, c="#fc4f30", label="ARI")
plt.xlabel("n_components")
plt.legend()
plt.title("Clustering Performance\nas a Function of n_components")
plt.tight_layout()
plt.show()
```

## Conclusion

This project showcases the construction of a comprehensive K-Means clustering pipeline for gene expression data. It includes data retrieval, preprocessing, model fitting, evaluation, and visualization. The parameter tuning section provides insights into the impact of the number of principal components on clustering performance. The visualizations offer a clear understanding of the clustering results in the context of cancer subtypes.

Last updated on: 2024-02-11

Last updated on: 2024-02-11

Last updated on: 2024-02-12

Last updated on: 2024-02-12

Last updated on: 2024-02-13

Last updated on: 2024-02-14

Last updated on: 2024-02-16

Last updated on: 2024-02-17

Last updated on: 2024-02-18

Last updated on: 2024-02-22

Last updated on: 2024-02-28

Last updated on: 2024-02-29

Last updated on: 2024-03-04

Last updated on: 2024-03-12

Last updated on: 2024-03-19

Last updated on: 2024-03-29

Last updated on: 2024-04-07

Last updated on: 2024-04-07

Last updated on: 2024-04-08

Last updated on: 2024-04-08

Last updated on: 2024-04-11

Last updated on: 2024-04-11

Last updated on: 2024-04-13

Last updated on: 2024-04-16

Last updated on: 2024-04-21

Last updated on: 2024-04-22

Last updated on: 2024-04-23

Last updated on: 2024-04-23

Last updated on: 2024-04-23

Last updated on: 2024-04-26

Last updated on: 2024-04-28