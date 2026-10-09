# Best prompt version + scoring mode, by model

Differences of 1-2 files are within dev-set noise on this dataset size.

## e2b_conformer5s_text_nothinking
### threshold >= 0.6
- **v5 / raw** — recall 0.8, specificity 0.64, flag_precision 0.46 (fp_files=9, fn_files=5)
- **v5 / sa** — recall 0.8, specificity 0.64, flag_precision 0.46 (fp_files=9, fn_files=5)
- **v5 / sa_drop_invalid** — recall 0.8, specificity 0.64, flag_precision 0.46 (fp_files=9, fn_files=5)

### threshold >= 0.8
- **v5 / raw** — recall 0.76, specificity 0.72, flag_precision 0.53 (fp_files=7, fn_files=6)
- **v5 / sa** — recall 0.76, specificity 0.72, flag_precision 0.53 (fp_files=7, fn_files=6)
- **v5 / sa_drop_invalid** — recall 0.76, specificity 0.72, flag_precision 0.53 (fp_files=7, fn_files=6)

### threshold >= 0.9
- **v5 / raw** — recall 0.76, specificity 0.76, flag_precision 0.56 (fp_files=6, fn_files=6)
- **v5 / sa** — recall 0.76, specificity 0.76, flag_precision 0.56 (fp_files=6, fn_files=6)
- **v5 / sa_drop_invalid** — recall 0.76, specificity 0.76, flag_precision 0.56 (fp_files=6, fn_files=6)

## e2b_nothinking
### threshold >= 0.6
- **v5 / raw** — recall 1.0, specificity 0.32, flag_precision 0.24 (fp_files=17, fn_files=0)
- **v5 / sa** — recall 1.0, specificity 0.32, flag_precision 0.24 (fp_files=17, fn_files=0)
- **v5 / sa_drop_invalid** — recall 1.0, specificity 0.32, flag_precision 0.24 (fp_files=17, fn_files=0)

### threshold >= 0.8
- **v6 / pp_full** — recall 0.92, specificity 0.56, flag_precision 0.59 (fp_files=11, fn_files=2)
- **v6 / sa** — recall 0.92, specificity 0.52, flag_precision 0.54 (fp_files=12, fn_files=2)
- **v6 / sa_drop_invalid** — recall 0.92, specificity 0.52, flag_precision 0.54 (fp_files=12, fn_files=2)

### threshold >= 0.9
- **v6 / pp_full** — recall 0.92, specificity 0.6, flag_precision 0.61 (fp_files=10, fn_files=2)
- **v6 / sa** — recall 0.92, specificity 0.52, flag_precision 0.55 (fp_files=12, fn_files=2)
- **v6 / sa_drop_invalid** — recall 0.92, specificity 0.52, flag_precision 0.55 (fp_files=12, fn_files=2)

## e2b_thinking
### threshold >= 0.6
- **v3 / raw** — recall 1.0, specificity 0.42, flag_precision 0.38 (fp_files=7, fn_files=0)
- **v3 / sa** — recall 1.0, specificity 0.42, flag_precision 0.38 (fp_files=7, fn_files=0)
- **v3 / sa_drop_invalid** — recall 1.0, specificity 0.42, flag_precision 0.38 (fp_files=7, fn_files=0)

### threshold >= 0.8
- **v5 / raw** — recall 0.88, specificity 0.48, flag_precision 0.32 (fp_files=13, fn_files=3)
- **v5 / sa** — recall 0.88, specificity 0.48, flag_precision 0.32 (fp_files=13, fn_files=3)
- **v5 / sa_drop_invalid** — recall 0.88, specificity 0.48, flag_precision 0.32 (fp_files=13, fn_files=3)

### threshold >= 0.9
- **v5 / raw** — recall 0.88, specificity 0.52, flag_precision 0.34 (fp_files=12, fn_files=3)
- **v5 / sa** — recall 0.88, specificity 0.52, flag_precision 0.34 (fp_files=12, fn_files=3)
- **v5 / sa_drop_invalid** — recall 0.88, specificity 0.52, flag_precision 0.34 (fp_files=12, fn_files=3)

## e4b_nothinking
### threshold >= 0.6
- **v8 / pp_full** — recall 0.8, specificity 0.6, flag_precision 0.35 (fp_files=10, fn_files=5)
- **v8 / sa** — recall 0.8, specificity 0.56, flag_precision 0.33 (fp_files=11, fn_files=5)
- **v8 / sa_drop_invalid** — recall 0.8, specificity 0.56, flag_precision 0.33 (fp_files=11, fn_files=5)

### threshold >= 0.8
- **v7 / sa** — recall 0.72, specificity 0.88, flag_precision 0.4 (fp_files=3, fn_files=7)
- **v7 / sa_drop_invalid** — recall 0.72, specificity 0.88, flag_precision 0.4 (fp_files=3, fn_files=7)
- **v7 / pp_full** — recall 0.72, specificity 0.88, flag_precision 0.4 (fp_files=3, fn_files=7)

### threshold >= 0.9
- **v7 / sa** — recall 0.72, specificity 0.88, flag_precision 0.4 (fp_files=3, fn_files=7)
- **v7 / sa_drop_invalid** — recall 0.72, specificity 0.88, flag_precision 0.4 (fp_files=3, fn_files=7)
- **v7 / pp_full** — recall 0.72, specificity 0.88, flag_precision 0.4 (fp_files=3, fn_files=7)

## e4b_thinking
### threshold >= 0.6
- **v5 / raw** — recall 1.0, specificity 0.32, flag_precision 0.31 (fp_files=17, fn_files=0)
- **v5 / sa** — recall 1.0, specificity 0.32, flag_precision 0.31 (fp_files=17, fn_files=0)
- **v5 / sa_drop_invalid** — recall 1.0, specificity 0.32, flag_precision 0.31 (fp_files=17, fn_files=0)

### threshold >= 0.8
- **v5 / raw** — recall 0.96, specificity 0.56, flag_precision 0.38 (fp_files=11, fn_files=1)
- **v5 / sa** — recall 0.96, specificity 0.56, flag_precision 0.38 (fp_files=11, fn_files=1)
- **v5 / sa_drop_invalid** — recall 0.96, specificity 0.56, flag_precision 0.38 (fp_files=11, fn_files=1)

### threshold >= 0.9
- **v5 / raw** — recall 0.96, specificity 0.56, flag_precision 0.39 (fp_files=11, fn_files=1)
- **v5 / sa** — recall 0.96, specificity 0.56, flag_precision 0.39 (fp_files=11, fn_files=1)
- **v5 / sa_drop_invalid** — recall 0.96, specificity 0.56, flag_precision 0.39 (fp_files=11, fn_files=1)
