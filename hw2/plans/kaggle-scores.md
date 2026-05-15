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

## Methods Notes

- **Louvain res sweep**: `main.py` — networkx + python-louvain
- **Leiden sweep**: `leiden_sweep.py` — igraph + leidenalg, modularity & RBConfig
- **Hierarchical**: `hierarchical_louvain.py` — dendrogram levels
- **Consensus**: `consensus_louvain.py` — 20 Louvain runs, vote thresholds
- **Classifier (baseline)**: `classifier_pipeline.py` — sklearn GB + LR, 7 features (cn, jaccard, AA, log-PA, same-Leiden, same-Louvain, same-component); labels = edges vs cross-Leiden same-comp
- **Enhanced classifier**: `enhanced_classifier.py` — adds 6 features (RA, shortest-path BFS cap=4, Louvain consensus 10 runs, Leiden consensus 5 res, log-deg-u, log-deg-v); 3 models (sklearn-GB, XGBoost, LightGBM); labels = edges + same-Leiden non-edges vs cross-Leiden non-edges. **Overshoots — same-Leiden non-edges as positives is a wrong assumption.**
- **Tuned classifier**: `classifier_tuned.py` — same 7 features + labels as `classifier_pipeline.py`, but ENSEMBLE of 8 seeds (averaged probs) + threshold sweep + count-target submissions. Probability stats: median=0.081 (negatives), bimodal distribution.

## To Try Next (if scores below disappoint)
- Probability calibration / finer threshold sweep
- Ensemble: classifier_gb (448) + consensus voting tiebreaker
- Replace Leiden-based labels with Louvain-consensus-based labels (decouple feature/label sources)
- Walktrap or Infomap community detection
- Spectral embedding + classifier on (u, v) embedding pair
