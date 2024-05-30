# TP1 - Document Clustering with the JoSE Model on the BBC-News Dataset

This project is conducted as part of the evaluation for the mixture-models and co-clustering module of the Master 2 MLDS - Machine Learning for Data Science program.

## Team Members

- Abdesselam BENAMEUR
- Hakim IGUENI

## Overview

This R markdown file provides a comprehensive guide to document clustering using the JOSE model on the BBC-News dataset. It covers the following:

1. Installing necessary packages.
2. Importing the required libraries.
3. Loading and examining the dataset.
4. Performing clustering using various algorithms.
5. Evaluating and comparing the results.

## Installing Packages

Ensure you have the following R packages installed:

```r
install.packages("NbClust")
install.packages("skmeans")
install.packages("movMF")
install.packages("R.matlab")
install.packages("aricode")
```

## Importing Packages

Load the required libraries:

```r
library(NbClust)
library(skmeans)
library(movMF)
library(R.matlab)
library(aricode)
```

## Loading the Data

The dataset can be downloaded from the provided URL:

```r
url <- "https://cifre.s3.eu-north-1.amazonaws.com/bbc_dataset.mat"
bbc_dataset <- readMat(url)
bbc_jose <- bbc_dataset$jose
bbc_doc_term <- bbc_dataset$doc.term
bbc_labels <- as.vector(bbc_dataset$labels)
```

## Data Dimensions

Determine the dimensions of the datasets:

```r
dim(bbc_jose)
dim(bbc_doc_term)
```

### Dataset Dimensions

- **Document-Term Matrix (bbc_doc_term):** 
  - Number of documents: 2225
  - Number of words: 2000

- **Document Embeddings (bbc_jose):**
  - Number of documents: 2225
  - Dimension size: 100

## Clustering Algorithms

### K-means Clustering

```r
set.seed(123)
res.kmeans <- NbClust(bbc_doc_term, distance = "euclidean", min.nc = 4, max.nc = 6, method = "kmeans")
clusters.kmeans <- res.kmeans$Best.partition
table(clusters.kmeans, bbc_labels)

nmi.kmeans <- NMI(clusters.kmeans, bbc_labels)
paste("NMI:", nmi.kmeans)

ari.kmeans <- ARI(clusters.kmeans, bbc_labels)
paste("ARI:", ari.kmeans)
```

### Spherical K-means Clustering

```r
set.seed(123)
res.skmeans <- skmeans(bbc_doc_term, 5)
clusters.skmeans <- res.skmeans$cluster
table(clusters.skmeans, bbc_labels)

nmi.skmeans <- NMI(clusters.skmeans, bbc_labels)
paste("NMI:", nmi.skmeans)

ari.skmeans <- ARI(clusters.skmeans, bbc_labels)
paste("ARI:", ari.skmeans)
```

### von-Mises Fisher Mixture Model

```r
set.seed(123)
res.movMF <- movMF(bbc_doc_term, 5, kappa=list(common = TRUE), nruns=5, maxit=200)
clusters.movMF <- apply(res.movMF$P,1,which.max)
table(clusters.movMF, bbc_labels)

nmi.movMF <- NMI(clusters.movMF, bbc_labels)
paste("NMI:", nmi.movMF)

ari.movMF <- ARI(clusters.movMF, bbc_labels)
paste("ARI:", ari.movMF)
```

### Hierarchical Clustering

```r
res.hclust <- hclust(dist(bbc_doc_term), method = "ward.D2")
clusters.hclust <- cutree(res.hclust, 5)
table(clusters.hclust, bbc_labels)

nmi.hclust <- NMI(clusters.hclust, bbc_labels)
paste("NMI:", nmi.hclust)

ari.hclust <- ARI(clusters.hclust, bbc_labels)
paste("ARI:", ari.hclust)
```

## Results Summary

### Document-Term Matrix Clustering Results

| Clustering Algorithm  | NMI   | ARI   |
|-----------------------|-------|-------|
| K-means               | 0.063 | 0.063 |
| Spherical K-means     | 0.276 | 0.212 |
| von-Mises Fisher MM   | 0.262 | 0.191 |
| Hierarchical Clustering| 0.075 | 0.066 |

### Document Embeddings Clustering Results

| Clustering Algorithm  | NMI   | ARI   |
|-----------------------|-------|-------|
| K-means               | 0.857 | 0.894 |
| Spherical K-means     | 0.848 | 0.886 |
| von-Mises Fisher MM   | 0.845 | 0.884 |
| Hierarchical Clustering| 0.844 | 0.875 |

### Analysis

- **Document-Term Matrix Clustering:**
  - K-means shows the lowest performance.
  - Spherical K-means performs better, indicating the importance of considering the spherical nature of the data.
  - The von-Mises Fisher Mixture Model also performs well, capturing the distributional characteristics effectively.
  - Hierarchical clustering provides marginal improvements but still shows low performance.

- **Document Embeddings Clustering:**
  - Applying JoSE embeddings significantly improves clustering results across all algorithms.
  - K-means with JoSE embeddings achieves the highest NMI and ARI scores.
  - Spherical K-means, von-Mises Fisher Mixture Model, and Hierarchical Clustering also perform well, indicating that the embeddings better capture the semantic relationships in the data.

In summary, JoSE embeddings enhance clustering performance, leading to more accurate and cohesive results.
