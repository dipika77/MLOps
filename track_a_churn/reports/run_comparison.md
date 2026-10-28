| run | model family | key hyperparams | accuracy | precision | recall | f1 | roc_auc |
|---|---|---|---|---|---|---|---|
| logreg_c0.1 | LogisticRegression | C=0.1, penalty=l2, max_iter=1000 | 0.7417 | 0.5087 | 0.7861 | **0.6176** | 0.8408 |
| rf_shallow_100trees | RandomForest | n_estimators=100, max_depth=5 | 0.7381 | 0.5043 | 0.7888 | **0.6152** | 0.8388 |
| logreg_c1.0 | LogisticRegression | C=1.0, penalty=l2, max_iter=1000 | 0.7381 | 0.5043 | 0.7834 | **0.6136** | 0.8415 |
| rf_deep_300trees | RandomForest | n_estimators=300, max_depth=12 | 0.7530 | 0.5261 | 0.7005 | **0.6009** | 0.8340 |
