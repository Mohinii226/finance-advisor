| world | detector | alerts_per_cust_month | precision | event_recall | recall_creep | recall_spike | recall_medical | ttd_creep | f1 | avg_precision | base_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DEV (tuned on) | naive | 1.53 | 0.025 | 0.833 | 0.914 | 0.65 | 1.0 | 2.0 | 0.049 | 0.029 | 0.012 |
| DEV (tuned on) | balanced | 0.268 | 0.078 | 0.482 | 0.552 | 0.275 | 0.75 | 3.0 | 0.135 | 0.029 | 0.012 |
| DEV (tuned on) | low_noise | 0.083 | 0.103 | 0.237 | 0.328 | 0.125 | 0.188 | 3.0 | 0.144 | 0.039 | 0.012 |
| TEST (unseen) | naive | 1.509 | 0.018 | 0.711 | 0.878 | 0.488 | 0.923 | 3.0 | 0.036 | 0.024 | 0.008 |
| TEST (unseen) | balanced | 0.27 | 0.058 | 0.392 | 0.537 | 0.14 | 0.769 | 4.0 | 0.101 | 0.026 | 0.008 |
| TEST (unseen) | low_noise | 0.09 | 0.071 | 0.206 | 0.341 | 0.093 | 0.154 | 4.0 | 0.106 | 0.037 | 0.008 |
| TEST, perfect categoriser | naive | 1.369 | 0.021 | 0.732 | 0.902 | 0.488 | 1.0 | 2.0 | 0.041 | 0.037 | 0.01 |
| TEST, perfect categoriser | balanced | 0.25 | 0.063 | 0.412 | 0.439 | 0.209 | 1.0 | 4.0 | 0.109 | 0.031 | 0.009 |
| TEST, perfect categoriser | low_noise | 0.075 | 0.064 | 0.175 | 0.268 | 0.07 | 0.231 | 3.0 | 0.093 | 0.047 | 0.009 |
