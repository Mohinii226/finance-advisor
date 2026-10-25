# Forecast backtest results

150 simulated customers, 36 months, rolling-origin backtest over the last 12 months. Slow per-series models (ARIMA, Prophet) are run on a subset; every comparison uses only the customers all models share.

| model          |     MAE |   WAPE |   sMAPE |   MASE |   win_vs_naive |   series |   origins |   seconds |
|:---------------|--------:|-------:|--------:|-------:|---------------:|---------:|----------:|----------:|
| gbm_l1         | 11564.5 |  0.138 |   0.131 |  0.816 |          0.657 |      150 |        12 |     3.391 |
| gbm_l2         | 12054.6 |  0.144 |   0.136 |  0.855 |          0.624 |      150 |        12 |     3.465 |
| median6        | 12400.2 |  0.148 |   0.144 |  0.889 |          0.637 |      150 |        12 |     0.002 |
| mean6          | 12509.6 |  0.15  |   0.145 |  0.889 |          0.641 |      150 |        12 |     0.001 |
| ses            | 12936.7 |  0.155 |   0.148 |  0.914 |          0.639 |      150 |        12 |     0.007 |
| mean3          | 13655.1 |  0.163 |   0.156 |  0.957 |          0.609 |      150 |        12 |     0.002 |
| naive_last     | 16465.7 |  0.197 |   0.191 |  1.178 |          0     |      150 |        12 |     0.003 |
| seasonal_naive | 16666.1 |  0.199 |   0.185 |  1.106 |          0.517 |      150 |        12 |     0.001 |

WAPE difference vs naive_last (negative = better):

```
         model   baseline  wape_diff  ci_low  ci_high  significant
        gbm_l1 naive_last    -0.0586 -0.0674  -0.0500         True
        gbm_l2 naive_last    -0.0528 -0.0620  -0.0445         True
       median6 naive_last    -0.0486 -0.0575  -0.0404         True
         mean6 naive_last    -0.0473 -0.0540  -0.0406         True
           ses naive_last    -0.0422 -0.0503  -0.0346         True
         mean3 naive_last    -0.0336 -0.0407  -0.0269         True
seasonal_naive naive_last     0.0024 -0.0130   0.0190        False
```

WAPE difference vs the strongest simple baseline (median6):

```
         model baseline  wape_diff  ci_low  ci_high  significant
        gbm_l1  median6    -0.0100 -0.0146  -0.0056         True
        gbm_l2  median6    -0.0041 -0.0101   0.0020        False
         mean6  median6     0.0013 -0.0027   0.0057        False
           ses  median6     0.0064  0.0010   0.0126         True
         mean3  median6     0.0150  0.0090   0.0210         True
    naive_last  median6     0.0486  0.0404   0.0575         True
seasonal_naive  median6     0.0510  0.0367   0.0671         True
```

Interval calibration:

```
 model  target  rel_low  rel_high  coverage  n_test
gbm_l1     0.8   -0.169     0.257     0.794     900
```
