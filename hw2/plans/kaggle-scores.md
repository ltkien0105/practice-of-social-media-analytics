# Kaggle Scores

Top-1 target: **0.96200**

| Submission File | Positives | Kaggle Score | Notes |
|---|---|---|---|
| submission.csv | 364 | — | Baseline Louvain res=1.0 (duplicate of submission_res1.csv) |
| submission_res0.5.csv | 345 | — | Louvain res=0.5 |
| submission_res0.75.csv | 365 | 0.85000 | Louvain res=0.75 |
| submission_res1.csv | 364 | **0.85800** | Louvain res=1.0 baseline |
| submission_res1.25.csv | 369 | — | Louvain res=1.25 |
| submission_res1.5.csv | 308 | — | Louvain res=1.5 |
| submission_res2.csv | 322 | — | Louvain res=2.0 |
| submission_leiden_mod.csv | 363 | — | Leiden default modularity (Q=0.869) |
| submission_leiden_rb0.5.csv | 354 | — | Leiden RB res=0.5 |
| submission_leiden_rb0.75.csv | 361 | 0.84600 | Leiden RB res=0.75 (highest Q=0.869) |
| submission_leiden_rb1.csv | 363 | — | Leiden RB res=1.0 |
| submission_leiden_rb1.25.csv | 362 | — | Leiden RB res=1.25 |
| submission_leiden_rb1.5.csv | 338 | — | Leiden RB res=1.5 |
| submission_leiden_rb2.csv | 335 | — | Leiden RB res=2.0 |
| submission_hier_level0.csv | 97 | — | Hierarchical Louvain level 0 (finest) |
| submission_hier_level1.csv | 243 | — | Hierarchical Louvain level 1 |
| submission_hier_level2.csv | 320 | — | Hierarchical Louvain level 2 |
| submission_hier_level3.csv | 360 | — | Hierarchical Louvain level 3 |
| submission_hier_level4.csv | 364 | — | Hierarchical Louvain level 4 (= baseline) |
| submission_consensus_t0.3.csv | 386 | — | Consensus 20-run Louvain, vote>=0.3 |
| submission_consensus_t0.5.csv | 361 | — | Consensus 20-run Louvain, vote>=0.5 |
| submission_consensus_t0.7.csv | 341 | — | Consensus 20-run Louvain, vote>=0.7 |
| submission_consensus_t0.9.csv | 306 | — | Consensus 20-run Louvain, vote>=0.9 |
| submission_clf_gb.csv | 448 | **0.92200** | Classifier GB (7 features, edges+xLeiden labels) |
| submission_clf_lr.csv | 447 | — | Classifier LogReg (7 features) |
| submission_sgb_t0.5.csv | 592 | 0.78800 | Enhanced sklearn-GB (13 feat, mixed-SP labels) — OVERSHOOT |
| submission_xgb_t0.5.csv | 528 | 0.73200 | Enhanced XGBoost (13 feat) — OVERSHOOT |
| submission_lgbm_t0.5.csv | 592 | 0.78800 | Enhanced LightGBM (13 feat) — OVERSHOOT |
| submission_tuned_t0.3.csv | 462 | — | Tuned 8-seed ensemble, thr=0.3 |
| submission_tuned_t0.4.csv | 453 | — | Tuned 8-seed ensemble, thr=0.4 |
| submission_tuned_t0.45.csv | 450 | — | Tuned 8-seed ensemble, thr=0.45 |
| submission_tuned_t0.5.csv | 447 | — | Tuned 8-seed ensemble, thr=0.5 (≈ clf_gb baseline) |
| submission_tuned_t0.55.csv | 446 | — | Tuned 8-seed ensemble, thr=0.55 |
| submission_tuned_t0.6.csv | 443 | — | Tuned 8-seed ensemble, thr=0.6 |
| submission_tuned_t0.7.csv | 438 | — | Tuned 8-seed ensemble, thr=0.7 |
| submission_tuned_top420.csv | 420 | — | Tuned top-N by prob, N=420 (high precision) |
| submission_tuned_top448.csv | 448 | — | Tuned top-N by prob, N=448 |
| submission_tuned_top460.csv | 460 | — | Tuned top-N by prob, N=460 |
| submission_tuned_top470.csv | 470 | — | Tuned top-N by prob, N=470 |
| submission_tuned_top478.csv | 478 | — | Tuned top-N by prob, N=478 (estimated true count) |
| submission_tuned_top490.csv | 490 | — | Tuned top-N by prob, N=490 |
| submission_tuned_top500.csv | 500 | — | Tuned top-N by prob, N=500 |
| submission_tuned_t0.5.csv | 447 | 0.92000 | Tuned 8-seed ensemble, thr=0.5 |
| submission_tuned_top478.csv | 478 | **0.93200** | Tuned ensemble, top-478 by prob — new best |
| submission_hidden_t0.3.csv | 629 | — | Hidden-edge classifier, thr=0.3 (broad) |
| submission_hidden_t0.4.csv | 463 | — | Hidden-edge classifier, thr=0.4 |
| submission_hidden_t0.5.csv | 461 | — | Hidden-edge classifier, thr=0.5 |
| submission_hidden_t0.6.csv | 460 | — | Hidden-edge classifier, thr=0.6 |
| submission_hidden_t0.7.csv | 459 | — | Hidden-edge classifier, thr=0.7 |
| submission_hidden_top448.csv | 448 | — | Hidden-edge top-N, N=448 |
| submission_hidden_top478.csv | 478 | — | Hidden-edge top-N, N=478 (estimated true count) |
| submission_hidden_top500.csv | 500 | — | Hidden-edge top-N, N=500 |
| submission_hidden_top520.csv | 520 | — | Hidden-edge top-N, N=520 |
| submission_hidden_top540.csv | 540 | — | Hidden-edge top-N, N=540 |
| submission_hidden_top478.csv | 478 | 0.93200 | Hidden-edge top-N=478 |
| submission_hidden_top500.csv | 500 | 0.93200 | Hidden-edge top-N=500 |
| submission_hidden_t0.5.csv | 461 | **0.94600** | Hidden-edge thr=0.5 — current best |
| submission_ens_t3h7_top460.csv | 460 | — | Ensemble 0.3×tuned + 0.7×hidden, top-460 |
| submission_ens_t5h5_top460.csv | 460 | — | Ensemble 0.5×tuned + 0.5×hidden, top-460 |
| submission_ens_t5h5_top465.csv | 465 | — | Ensemble 50/50, top-465 |
| submission_ens_t7h3_top460.csv | 460 | — | Ensemble 0.7×tuned + 0.3×hidden, top-460 |
| submission_ens_t3h7_t0.5.csv | 460 | — | Ensemble 30/70, thr=0.5 |
| submission_ens_t5h5_t0.5.csv | 456 | — | Ensemble 50/50, thr=0.5 |
| submission_ens_t3h7_top460.csv | 460 | **0.94800** | Ensemble 0.3×tuned + 0.7×hidden, top-460 — new best |
| submission_ens_t5h5_top460.csv | 460 | 0.94600 | Ensemble 50/50 — no better than hidden alone |
| submission_fine_t20h80_top460.csv | 460 | — | Fine grid 0.20/0.80 |
| submission_fine_t25h75_top460.csv | 460 | — | Fine grid 0.25/0.75 |
| submission_fine_t35h65_top460.csv | 460 | — | Fine grid 0.35/0.65 |
| submission_rankavg_top460.csv | 460 | — | Borda rank-average of tuned & hidden |
| submission_geomean_top460.csv | 460 | — | Geometric mean of probabilities |
| submission_max_top460.csv | 460 | — | Per-pair max(tuned, hidden) |
| submission_fine_t25h75_top460.csv | 460 | 0.94600 | Fine 0.25/0.75 |
| submission_fine_t35h65_top460.csv | 460 | 0.94800 | Fine 0.35/0.65 (tied best) |
| submission_rankavg_top460.csv | 460 | 0.94600 | Borda rank-avg |
| submission_walktrap_t0.5.csv | 461 | — | Walktrap-augmented hidden-edge classifier |
| submission_walktrap_top460.csv | 460 | — | Walktrap top-460 |
| submission_3rankavg_top460.csv | 460 | — | 3-way Borda rank-avg (tuned+hidden+walk) |
| submission_3w_t25h45w30_top460.csv | 460 | — | 3-way blend, weighted toward hidden |
| submission_3w_t30h50w20_top460.csv | 460 | 0.94800 | 3-way blend, more tuned (tied best) |
| submission_3rankavg_top460.csv | 460 | 0.94800 | Borda over 3 (tied best) |
| submission_walktrap_t0.5.csv | 461 | 0.94600 | Walktrap alone (= hidden) |
| submission_holdout_t0.5.csv | 668 | — | Holdout classifier — overshoots |
| submission_holdout_top460.csv | 460 | — | Holdout top-460 |
| submission_holdblend_ho30_top460.csv | 460 | 0.94600 | 4-way blend (tuned 0.2, hidden 0.5, holdout 0.3) — no gain |
| submission_holdblend_ho40_top460.csv | 460 | — | 4-way blend, more holdout (0.4) |
| submission_holdblend_ho30b_top460.csv | 460 | 0.94400 | Less tuned, more holdout — slight drop |
| submission_4rank_top460.csv | 460 | 0.92200 | Rank-avg of all 4 — holdout's bad ranking drags it down |
| submission_n2v_cos_top460.csv | 460 | 0.94600 | Node2Vec DeepWalk (dim=128, walks=10×80) cosine similarity, top-460 |
| submission_n2v_clf_top460.csv | 460 | 0.66800 | Node2Vec hadamard product + GBM classifier — catastrophic, embedding products lose community signal |
| submission_blend_n2v20_top460.csv | 460 | 0.94800 | Baseline + 20% n2v_clf — same as baseline (one neutral flip) |
| submission_blend_n2v30_top460.csv | 460 | 0.94400 | Baseline + 30% n2v_clf — bad signal pulls down |

## Summary

**Best achievable**: `submission_ens_t3h7_top460.csv` → **0.94800** (0.3×tuned_probs + 0.7×hidden_edge_probs, top-460 by averaged probability).

**Plateau confirmed at 0.948**. Held-out edge training degenerated (sp dominated, saturated probabilities). Walktrap features unused (lou_cons dominates). Linear/rank/geometric ensemble of correlated models can't break through. Node2Vec cosine alone hits 0.946 (very close to baseline) — embeddings have signal but cannot exceed the structural-feature ensemble.

To reach 0.96+ likely needs:
- Biased Node2Vec walks (p=1, q=0.5) for stronger community sampling — pending
- Graph neural network (e.g., GCN on PyG) — significant additional engineering
- A genuinely different label source than train.csv edges

## Methods Notes

- **Louvain res sweep**: `main.py` — networkx + python-louvain
- **Leiden sweep**: `leiden_sweep.py` — igraph + leidenalg, modularity & RBConfig
- **Hierarchical**: `hierarchical_louvain.py` — dendrogram levels
- **Consensus**: `consensus_louvain.py` — 20 Louvain runs, vote thresholds
- **Classifier (baseline)**: `classifier_pipeline.py` — sklearn GB + LR, 7 features (cn, jaccard, AA, log-PA, same-Leiden, same-Louvain, same-component); labels = edges vs cross-Leiden same-comp
- **Enhanced classifier**: `enhanced_classifier.py` — adds 6 features (RA, shortest-path BFS cap=4, Louvain consensus 10 runs, Leiden consensus 5 res, log-deg-u, log-deg-v); 3 models (sklearn-GB, XGBoost, LightGBM); labels = edges + same-Leiden non-edges vs cross-Leiden non-edges. **Overshoots — same-Leiden non-edges as positives is a wrong assumption.**
- **Tuned classifier**: `classifier_tuned.py` — same 7 features + labels as `classifier_pipeline.py`, but ENSEMBLE of 8 seeds (averaged probs) + threshold sweep + count-target submissions. Probability stats: median=0.081 (negatives), bimodal distribution.
- **Hidden-edge classifier**: `classifier_hidden_edge.py` — 12 features (adds RA, shortest_path BFS cap=4, Louvain consensus over 8 runs, log_deg_u, log_deg_v). Same labels as classifier_pipeline BUT half of positive edge samples have features computed with the (u,v) edge HIDDEN — so sp varies in positive training samples (avoids trivial label leak). 6-seed ensemble. Feature importance: lou_cons=0.83 (dominant), sp=0.13, log_pa=0.015, same_leid=0.015. Optimal threshold=0.5 (461 positives) due to mass of pairs tied at prob=0.352 (unknown-region default).
- **Ensemble**: `ensemble_probs.py` — weighted average of tuned + hidden probability vectors. Rank correlation 0.81; top-500 disagreement is 42 pairs. Weights w_tuned/w_hidden in {0.0/1.0, 0.3/0.7, 0.5/0.5, 0.7/0.3, 1.0/0.0}.
- **Node2Vec**: `node2vec-experiment.py` — DeepWalk-style walks (10 per node, length 80, p=q=1), Word2Vec SGNS (dim=128, window=10, neg=10, 5 epochs, 8 workers, 49 min train time). Two predictors: pure cosine similarity of normalized embeddings (0.946); hadamard product + 6-seed GBM ensemble on edges-vs-cross-Leiden labels (0.668 — catastrophic). Cosine top-460 overlaps 437/460 with the 0.948 baseline.

## To Try Next (if scores below disappoint)
- Probability calibration / finer threshold sweep
- Ensemble: classifier_gb (448) + consensus voting tiebreaker
- Replace Leiden-based labels with Louvain-consensus-based labels (decouple feature/label sources)
- Walktrap or Infomap community detection
- Spectral embedding + classifier on (u, v) embedding pair
