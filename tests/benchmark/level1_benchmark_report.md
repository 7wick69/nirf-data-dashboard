# NIRF Level 1 benchmark report

- Official catalogue questions: **40**
- Classifier/template cases: **160**
- Correct cases: **160/160 (100.0%)**
- Questions with all examples passing: **40/40**

## Classifier results by intent

| Intent | Passed | Total | Accuracy |
|---|---:|---:|---:|
| correlation_analysis | 20 | 20 | 100.0% |
| distribution_analysis | 32 | 32 | 100.0% |
| geographical_analysis | 20 | 20 | 100.0% |
| multivariate_analysis | 20 | 20 | 100.0% |
| outlier_analysis | 20 | 20 | 100.0% |
| rank_movement | 12 | 12 | 100.0% |
| statistical_summary | 8 | 8 | 100.0% |
| transition_analysis | 16 | 16 | 100.0% |
| trend_analysis | 12 | 12 | 100.0% |

## Analytics operation coverage audit

| Status | Unique operations | Questions |
|---|---:|---:|
| implemented | 12 | 14 |
| partial | 6 | 6 |
| not_implemented | 19 | 20 |

> Important: classifier accuracy is not end-to-end analytics accuracy. The operation coverage is a static audit against the current analytics-engine dispatch/methods. A question is not considered fully implemented until query parsing, slot validation, analytics, chart rendering and explanation are each verified.

## Failed classification/template cases

None.
