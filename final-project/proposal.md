# Social Network Analysis on the Twitch Gamers Dataset

## Dataset
We use the Twitch Gamers Social Network dataset (Rozemberczki & Sarkar, 2021), collected from the Twitch public API in Spring 2018. The graph represents mutual follow relationships between streamers, where each node carries rich metadata about the user’s activity and status.

- Nodes: 168,114 streamers
- Edges: 6,797, 557 mutual follow relationships
- Node features: views, languages, lifetime, affiliate status, churn, mature content


## Part 1 (Me) - Exploratory Data Analysis
This part examines the structural and statistical properties of the dataset before applying any machine learning.
- Degree distribution: ft a power-law model and compute the exponent α to determine whether the network is scale-free
- Clustering coefcient and transitivity: measure the tendency of the network to form tightly connected local groups.
- Node feature distributions: analyze the distribution of language, afliate rate, view count, account lifetime, and churn rate.
- Correlation analysis: compute a correlation heatmap between graph-structural features (degree, PageRank, clustering coefcient) and node attributes (views, afliate, lifetime).


## Part 2 (Me) - Community Detection
This part identifes natural groupings within the network using an unsupervised graph algorithm and examines what drives community formation.
- Louvain algorithm: detect communities by maximizing modularity; report the modularity score Q as a quality measure.
- Community profling: for each major community, report dominant language, language purity, afliate rate, and median view count.
- NMI evaluation: measure the Normalized Mutual Information between detected communities and language labels to determine whether the network is organized by language or by other factors.


## Part 3 - Node Classification
This part trains supervised classifers to predict streamer attributes and compares the predictive power of diferent feature sources.

**Prediction tasks:**
- Afliate status prediction (affiliate, binary)
- Churn prediction (dead account, binary)
- Broadcast language prediction (multi-class, top-5 languages)

**Feature sets evaluated for each task:**
- Network-only: degree, PageRank, local clustering coefcient
- Content-only: views, mature fag, account lifetime, language
- Combined: all features above

Models (Random Forest and XGBoost) are evaluated via 5-fold cross-validation using AUC-ROC for binary tasks and macro-F1 for language prediction. The comparison across feature sets reveals whether network position or content metadata is the stronger predictor for each task.


## Part 4 - Link Prediction
This part predicts whether two streamers who do not currently follow each other are likely to form a mutual follow relationship.
- Graph-only features: Common Neighbors, Jaccard Coefcient, Adamic-Adar, and Preferential Attachment — classical structural similarity scores between node pairs.
- Graph + node features: the above augmented with pairwise attribute similarity: language match, views ratio, afliate match, and lifetime diference.
- Evaluation: Logistic Regression and XGBoost are trained on both feature sets; AUC-ROC is compared to quantify how much node-level similarity improves prediction beyond graph structure alone