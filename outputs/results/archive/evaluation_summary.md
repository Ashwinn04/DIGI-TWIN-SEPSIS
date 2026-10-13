# Model Evaluation Summary

## Performance Metrics

                 auroc   auprc  sensitivity_at_80_specificity  accuracy  precision  recall  specificity  f1_score  optimal_threshold  accuracy_r85  precision_r85  recall_r85  specificity_r85  f1_score_r85  threshold_r85  auroc_ci_lower  auroc_ci_upper  auprc_ci_lower  auprc_ci_upper
GRU-D           0.5132  0.2346                         0.2065    0.6341     0.2508  0.3198       0.7246    0.2811             0.6912        0.3034         0.2229      0.8502           0.1459        0.3532         0.1588        0.471615        0.553596        0.201828        0.276636
LSTM            0.4877  0.2378                         0.1700    0.7699     0.4000  0.0567       0.9755    0.0993             0.9732        0.3080         0.2241      0.8502           0.1517        0.3547         0.1460        0.446959        0.525872        0.200986        0.282635
CNN-LSTM        0.4935  0.2191                         0.1822    0.3433     0.2309  0.8300       0.2030    0.3612             0.1839        0.3261         0.2290      0.8502           0.1750        0.3608         0.1575        0.453868        0.535037        0.190202        0.258067
Transformer     0.4829  0.2264                         0.2267    0.7337     0.2957  0.1377       0.9055    0.1878             0.8930        0.2763         0.2160      0.8502           0.1109        0.3445         0.1309        0.441598        0.522980        0.196991        0.268201
Random          0.4621  0.4012                         0.2227    0.5598     0.4444  0.2445       0.7833    0.3155             0.7580        0.4221         0.4062      0.8515           0.1176        0.5501         0.1314        0.412067        0.510469        0.352704        0.460873
Majority Class  0.5000  0.4149                         0.0000    0.5851     0.0000  0.0000       1.0000    0.0000                inf        0.4149         0.4149      1.0000           0.0000        0.5864         0.0000        0.500000        0.500000        0.371377        0.456522

## Key Findings

- **Best performing model**: GRU-D (AUROC: 0.5132)
- **Dataset size**: 1104 records, 23 patients
- **Class distribution**: 41.5% positive cases
- **Evaluation completed**: 2025-11-03 20:13:03

## Literature Comparison

See `outputs/results/literature_comparison.md`.
