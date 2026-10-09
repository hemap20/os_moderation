# Threshold overview — v6 full dataset (381 files) (category ALL, language ALL)

## e2b_nothinking
| prompt/mode | t=0.0 | t=0.3 | t=0.6 | t=0.8 | t=0.9 | t=1.0 |
|---|---|---|---|---|---|---|
| v6/raw | 0.74/0.32 | 0.68/0.42 | 0.64/0.58 | 0.53/0.75 | 0.52/0.75 | 0.0/1.0 |
| v6/sa | 0.74/0.29 | 0.68/0.42 | 0.63/0.6 | 0.52/0.76 | 0.51/0.76 | 0.0/1.0 |
| v6/pp_full | 0.74/0.38 | 0.68/0.49 | 0.54/0.73 | 0.5/0.77 | 0.49/0.77 | 0.0/1.0 |

- Best at t=0.6: **v6/raw** — recall 0.64 (136/213), specificity 0.58 (97/167), flag_precision 0.15
- Best at t=0.8: **v6/raw** — recall 0.53 (112/213), specificity 0.75 (125/167), flag_precision 0.2
- Best at t=0.9: **v6/raw** — recall 0.52 (110/213), specificity 0.75 (125/167), flag_precision 0.2

_Differences of 1-2 files are within dev-set noise on this dataset size._

## gemini-3.1-flash-lite
| prompt/mode | t=0.0 | t=0.3 | t=0.6 | t=0.8 | t=0.9 | t=1.0 |
|---|---|---|---|---|---|---|
| v6/raw | 0.66/0.77 | 0.65/0.83 | 0.63/0.92 | 0.59/0.95 | 0.59/0.95 | 0.0/1.0 |
| v6/sa | 0.66/0.77 | 0.65/0.83 | 0.63/0.92 | 0.59/0.95 | 0.59/0.95 | 0.0/1.0 |
| v6/pp_full | 0.66/0.77 | 0.65/0.83 | 0.63/0.92 | 0.59/0.95 | 0.59/0.95 | 0.0/1.0 |

- Best at t=0.6: **v6/raw** — recall 0.63 (134/214), specificity 0.92 (154/167), flag_precision 0.43
- Best at t=0.8: **v6/raw** — recall 0.59 (127/214), specificity 0.95 (159/167), flag_precision 0.47
- Best at t=0.9: **v6/raw** — recall 0.59 (127/214), specificity 0.95 (159/167), flag_precision 0.47

_Differences of 1-2 files are within dev-set noise on this dataset size._

## gemini-3.5-flash-lite
| prompt/mode | t=0.0 | t=0.3 | t=0.6 | t=0.8 | t=0.9 | t=1.0 |
|---|---|---|---|---|---|---|
| v6/raw | 0.09/0.99 | 0.09/0.99 | 0.08/0.99 | 0.08/0.99 | 0.08/0.99 | 0.0/1.0 |
| v6/sa | 0.09/0.99 | 0.09/0.99 | 0.08/0.99 | 0.07/0.99 | 0.07/0.99 | 0.0/1.0 |
| v6/pp_full | 0.09/0.99 | 0.09/0.99 | 0.08/0.99 | 0.07/0.99 | 0.07/0.99 | 0.0/1.0 |

- Best at t=0.6: **v6/raw** — recall 0.08 (18/213), specificity 0.99 (166/167), flag_precision 0.75
- Best at t=0.8: **v6/raw** — recall 0.08 (16/213), specificity 0.99 (166/167), flag_precision 0.76
- Best at t=0.9: **v6/raw** — recall 0.08 (16/213), specificity 0.99 (166/167), flag_precision 0.76

_Differences of 1-2 files are within dev-set noise on this dataset size._
