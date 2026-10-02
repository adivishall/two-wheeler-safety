# Detector accuracy by measured image condition — real held-out images

- Model: `runs/detect/traffic_model-2/weights/best.pt`; AP@50 at conf ≥ 0.001, NMS IoU 0.7; 95% image-bootstrap CIs; † = fewer than 10 instances (not interpretable).
- Conditions are **measured from pixels and boxes, not labelled** — each is a proxy (`modules/image_conditions.py`). Cut points fitted on val, reused on test: sharpness terciles [334.5, 621.7], dark < 60.0 mean luminance, glare ≥ 5% pixels ≥ 250.
- **Not measurable on this data**: resolution (all images pre-resized: 512x512, 640x640), camera angle, weather, true time of day, and anything past the detector (tracking, association, OCR, fines) — those need labelled field footage (`docs/FIELD_EVALUATION.md`).

## Findings

Per class, strata whose AP differs from the rest of the split with a CI excluding zero on val, among **25** comparisons with ≥ 10 instances on both sides — at 95%, about 1 would clear the bar by chance, so **only `confirmed` rows** (test agrees in sign, its own CI excluding zero) should be read as findings.

| condition | class | val ΔAP [95% CI] | test ΔAP [95% CI] | status |
|---|---|---|---|---|
| object size = **small vs large** (352 img) | TripleRiding | -0.271 [-0.478, -0.070] | -0.079 [-0.253, +0.023] | same direction, not significant |
| labelled rider boxes = **2-3** (35 img) | WithoutHelmet | -0.244 [-0.401, -0.093] | +0.071 [-0.074, +0.247] | not replicated |
| lighting (luminance proxy) = **normal** (299 img) | WithoutHelmet | -0.180 [-0.287, -0.038] | +0.150 [-0.073, +0.379] | not replicated |
| object size = **small vs large** (352 img) | WithoutHelmet | -0.156 [-0.281, -0.013] | -0.071 [-0.266, +0.113] | same direction, not significant |
| sharpness (Laplacian proxy) = **low** (117 img) | TripleRiding | -0.151 [-0.309, -0.007] | -0.038 [-0.158, +0.028] | same direction, not significant |
| lighting (luminance proxy) = **normal** (299 img) | Plate | -0.115 [-0.170, -0.055] | -0.079 [-0.173, +0.005] | same direction, not significant |
| lighting (luminance proxy) = **dark** (22 img) | Plate | +0.102 [+0.025, +0.161] | +0.123 [+0.064, +0.182] | confirmed |
| lighting (luminance proxy) = **bright** (31 img) | Plate | +0.121 [+0.073, +0.169] | -0.037 [-0.253, +0.140] | not replicated |
| sharpness (Laplacian proxy) = **mid** (118 img) | TripleRiding | +0.148 [+0.001, +0.294] | +0.042 [-0.030, +0.161] | same direction, not significant |
| lighting (luminance proxy) = **bright** (31 img) | WithoutHelmet | +0.273 [+0.205, +0.339] | -0.047 [-0.284, +0.207] | not replicated |
| labelled rider boxes = **1** (214 img) | WithoutHelmet | +0.366 [+0.220, +0.514] | +0.175 [+0.023, +0.337] | confirmed |

## val — 352 images

Overall mAP@50 0.7214.

### lighting (luminance proxy)

| value | images | Plate | WithHelmet | WithoutHelmet | TripleRiding |
|---|---:|---|---|---|---|
| bright | 31 | 0.98 [0.96, 0.99] n=18 | 0.99 [0.99, 0.99] n=2† | 0.99 [0.97, 0.99] n=13 | 0.74 [0.00, 0.99] n=4† |
| dark | 22 | 0.97 [0.90, 0.99] n=20 | 0.00 [0.00, 0.00] n=1† | 0.78 [0.59, 0.95] n=14 | — |
| normal | 299 | 0.85 [0.81, 0.90] n=209 | 0.39 [0.24, 0.59] n=23 | 0.71 [0.65, 0.78] n=182 | 0.87 [0.77, 0.95] n=54 |

### sharpness (Laplacian proxy)

| value | images | Plate | WithHelmet | WithoutHelmet | TripleRiding |
|---|---:|---|---|---|---|
| high | 117 | 0.86 [0.79, 0.93] n=112 | 0.17 [0.00, 0.99] n=2† | 0.75 [0.66, 0.84] n=94 | 0.99 [0.99, 0.99] n=2† |
| low | 117 | 0.86 [0.78, 0.94] n=59 | 0.69 [0.36, 0.95] n=9† | 0.77 [0.66, 0.88] n=51 | 0.81 [0.67, 0.94] n=38 |
| mid | 118 | 0.89 [0.82, 0.95] n=76 | 0.34 [0.18, 0.58] n=15 | 0.70 [0.57, 0.82] n=64 | 0.96 [0.89, 0.99] n=18 |

### glare

| value | images | Plate | WithHelmet | WithoutHelmet | TripleRiding |
|---|---:|---|---|---|---|
| glare | 49 | 0.88 [0.77, 0.97] n=43 | 0.53 [0.19, 0.90] n=7† | 0.62 [0.45, 0.80] n=33 | 0.99 [0.99, 0.99] n=5† |
| no glare | 303 | 0.87 [0.82, 0.92] n=204 | 0.41 [0.22, 0.62] n=19 | 0.76 [0.69, 0.82] n=176 | 0.85 [0.74, 0.94] n=53 |

### labelled rider boxes

| value | images | Plate | WithHelmet | WithoutHelmet | TripleRiding |
|---|---:|---|---|---|---|
| 0 | 102 | 0.95 [0.86, 0.99] n=29 | — | — | — |
| 1 | 214 | 0.87 [0.82, 0.92] n=158 | 0.23 [0.04, 0.59] n=9† | 0.84 [0.78, 0.89] n=149 | 0.89 [0.80, 0.97] n=56 |
| 2-3 | 35 | 0.86 [0.74, 0.94] n=57 | 0.60 [0.40, 0.82] n=17 | 0.56 [0.42, 0.70] n=56 | 0.83 [0.69, 0.99] n=2† |
| 4+ | 1 | 0.99 [0.99, 0.99] n=3† | — | 0.99 [0.99, 0.99] n=4† | — |

### source

| value | images | Plate | WithHelmet | WithoutHelmet | TripleRiding |
|---|---:|---|---|---|---|
| ds1 | 295 | 0.88 [0.83, 0.91] n=247 | 0.42 [0.25, 0.61] n=26 | 0.75 [0.69, 0.81] n=209 | — |
| dst | 57 | — | — | — | 0.89 [0.79, 0.97] n=58 |

### Recall at the operating thresholds

| condition | Plate | WithHelmet | WithoutHelmet | TripleRiding |
|---|---:|---:|---:|---:|
| lighting (luminance proxy): bright | 0.94 (18) | 1.00 (2) | 1.00 (13) | 0.75 (4) |
| lighting (luminance proxy): dark | 0.95 (20) | 0.00 (1) | 0.64 (14) | — |
| lighting (luminance proxy): normal | 0.79 (209) | 0.70 (23) | 0.70 (182) | 0.87 (54) |
| sharpness (Laplacian proxy): high | 0.81 (112) | 0.50 (2) | 0.70 (94) | 1.00 (2) |
| sharpness (Laplacian proxy): low | 0.80 (59) | 0.78 (9) | 0.75 (51) | 0.82 (38) |
| sharpness (Laplacian proxy): mid | 0.84 (76) | 0.67 (15) | 0.72 (64) | 0.94 (18) |
| glare: glare | 0.84 (43) | 0.86 (7) | 0.64 (33) | 1.00 (5) |
| glare: no glare | 0.81 (204) | 0.63 (19) | 0.73 (176) | 0.85 (53) |
| labelled rider boxes: 0 | 0.93 (29) | — | — | — |
| labelled rider boxes: 1 | 0.82 (158) | 0.44 (9) | 0.81 (149) | 0.88 (56) |
| labelled rider boxes: 2-3 | 0.77 (57) | 0.82 (17) | 0.48 (56) | 0.50 (2) |
| labelled rider boxes: 4+ | 0.33 (3) | — | 0.50 (4) | — |
| source: ds1 | 0.82 (247) | 0.69 (26) | 0.72 (209) | — |
| source: dst | — | — | — | 0.86 (58) |

### Object size (relative √area; COCO range rule)

| class | small | medium | large | small − large (paired) |
|---|---|---|---|---|
| Plate | 0.86 [0.80, 0.93] n=83 | 0.89 [0.83, 0.95] n=81 | 0.85 [0.78, 0.93] n=83 | +0.011 [-0.087, +0.112] |
| WithHelmet | 0.53 [0.24, 0.82] n=9† | 0.69 [0.40, 0.99] n=8† | 0.20 [0.00, 0.50] n=9† | +0.335 [-0.097, +0.689] |
| WithoutHelmet | 0.69 [0.57, 0.80] n=70 | 0.73 [0.62, 0.83] n=69 | 0.84 [0.76, 0.91] n=70 | -0.156 [-0.281, -0.013] |
| TripleRiding | 0.69 [0.49, 0.89] n=20 | 0.95 [0.83, 0.99] n=19 | 0.96 [0.90, 0.99] n=19 | -0.271 [-0.478, -0.070] |

## test — 175 images

Overall mAP@50 0.7502.

### lighting (luminance proxy)

| value | images | Plate | WithHelmet | WithoutHelmet | TripleRiding |
|---|---:|---|---|---|---|
| bright | 13 | 0.85 [0.62, 0.99] n=6† | — | 0.69 [0.49, 0.99] n=10 | 0.50 [0.25, 0.99] n=1† |
| dark | 13 | 0.99 [0.99, 0.99] n=13 | — | 0.37 [0.12, 0.82] n=4† | — |
| normal | 149 | 0.88 [0.81, 0.94] n=112 | 0.44 [0.21, 0.74] n=27 | 0.75 [0.66, 0.84] n=91 | 0.97 [0.92, 0.99] n=29 |

### sharpness (Laplacian proxy)

| value | images | Plate | WithHelmet | WithoutHelmet | TripleRiding |
|---|---:|---|---|---|---|
| high | 50 | 0.86 [0.77, 0.95] n=53 | 0.42 [0.00, 0.99] n=15 | 0.70 [0.57, 0.85] n=41 | — |
| low | 56 | 0.93 [0.84, 0.99] n=27 | 0.74 [0.20, 0.99] n=4† | 0.77 [0.63, 0.91] n=25 | 0.95 [0.83, 0.99] n=15 |
| mid | 69 | 0.91 [0.82, 0.97] n=51 | 0.56 [0.16, 0.91] n=8† | 0.77 [0.66, 0.86] n=39 | 0.98 [0.95, 0.99] n=15 |

### glare

| value | images | Plate | WithHelmet | WithoutHelmet | TripleRiding |
|---|---:|---|---|---|---|
| glare | 28 | 0.87 [0.74, 0.98] n=31 | 0.33 [0.00, 0.89] n=4† | 0.71 [0.61, 0.84] n=29 | 0.99 [0.99, 0.99] n=2† |
| no glare | 147 | 0.89 [0.83, 0.95] n=100 | 0.44 [0.19, 0.80] n=23 | 0.75 [0.65, 0.85] n=76 | 0.96 [0.90, 0.99] n=28 |

### labelled rider boxes

| value | images | Plate | WithHelmet | WithoutHelmet | TripleRiding |
|---|---:|---|---|---|---|
| 0 | 52 | 0.96 [0.91, 0.99] n=23 | — | — | — |
| 1 | 100 | 0.88 [0.80, 0.95] n=68 | 0.03 [0.00, 0.22] n=4† | 0.80 [0.71, 0.89] n=66 | 0.97 [0.90, 0.99] n=30 |
| 2-3 | 20 | 0.89 [0.77, 0.99] n=34 | 0.80 [0.55, 0.96] n=9† | 0.80 [0.68, 0.93] n=33 | — |
| 4+ | 3 | 0.77 [0.50, 0.99] n=6† | 0.56 [0.34, 0.99] n=14 | 0.49 [0.00, 0.74] n=6† | — |

### source

| value | images | Plate | WithHelmet | WithoutHelmet | TripleRiding |
|---|---:|---|---|---|---|
| ds1 | 145 | 0.89 [0.83, 0.94] n=131 | 0.42 [0.19, 0.73] n=27 | 0.74 [0.67, 0.82] n=105 | — |
| dst | 30 | — | — | — | 0.97 [0.92, 0.99] n=30 |

### Recall at the operating thresholds

| condition | Plate | WithHelmet | WithoutHelmet | TripleRiding |
|---|---:|---:|---:|---:|
| lighting (luminance proxy): bright | 1.00 (6) | — | 0.70 (10) | 1.00 (1) |
| lighting (luminance proxy): dark | 1.00 (13) | — | 0.50 (4) | — |
| lighting (luminance proxy): normal | 0.85 (112) | 0.56 (27) | 0.78 (91) | 0.97 (29) |
| sharpness (Laplacian proxy): high | 0.81 (53) | 0.40 (15) | 0.68 (41) | — |
| sharpness (Laplacian proxy): low | 0.96 (27) | 1.00 (4) | 0.80 (25) | 0.93 (15) |
| sharpness (Laplacian proxy): mid | 0.88 (51) | 0.62 (8) | 0.82 (39) | 1.00 (15) |
| glare: glare | 0.90 (31) | 0.50 (4) | 0.76 (29) | 1.00 (2) |
| glare: no glare | 0.86 (100) | 0.57 (23) | 0.76 (76) | 0.96 (28) |
| labelled rider boxes: 0 | 0.91 (23) | — | — | — |
| labelled rider boxes: 1 | 0.88 (68) | 0.00 (4) | 0.82 (66) | 0.97 (30) |
| labelled rider boxes: 2-3 | 0.88 (34) | 1.00 (9) | 0.70 (33) | — |
| labelled rider boxes: 4+ | 0.50 (6) | 0.43 (14) | 0.50 (6) | — |
| source: ds1 | 0.87 (131) | 0.56 (27) | 0.76 (105) | — |
| source: dst | — | — | — | 0.97 (30) |

### Object size (relative √area; COCO range rule)

| class | small | medium | large | small − large (paired) |
|---|---|---|---|---|
| Plate | 0.88 [0.78, 0.98] n=49 | 0.91 [0.81, 0.98] n=38 | 0.88 [0.78, 0.97] n=44 | +0.003 [-0.125, +0.142] |
| WithHelmet | 0.23 [0.05, 0.99] n=2† | 0.59 [0.15, 0.94] n=9† | 0.44 [0.00, 0.81] n=16 | -0.211 [-0.600, +0.552] |
| WithoutHelmet | 0.68 [0.55, 0.80] n=45 | 0.86 [0.72, 0.96] n=25 | 0.75 [0.60, 0.90] n=35 | -0.071 [-0.266, +0.113] |
| TripleRiding | 0.90 [0.71, 0.99] n=8† | 0.99 [0.99, 0.99] n=8† | 0.98 [0.93, 0.99] n=14 | -0.079 [-0.253, +0.023] |
