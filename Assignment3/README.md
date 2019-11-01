# Instructions on Running Code

The code is the directory makes use of the default template, but I have extended the command line options for additional flexibility and to have a more organized directory structure. Use the following commands to run the test examples

## kMeans on YeastGene Dataset

```bash
python kmeans.py datasets/yeast/YeastGene.csv datasets/yeast/YeastGene_Initial_Centroids.csv 6 7
```

## PCA on YeastGene Clusters

```bash
python pca.py output/yeast/YeastGene_kmeans_cluster.csv
```

## Hierarchical on Utilities Dataset

```bash
python hierarchical.py datasets/Utilities.csv 1
```