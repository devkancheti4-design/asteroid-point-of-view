# The training-free path rule: our model in one file

[`path_rule_model.py`](path_rule_model.py) decides whether an asteroid is **potentially hazardous** (a PHA) from its
orbit alone. It needs **no training**, and it is **deterministic**: it uses no random numbers and gives the same
answer on any computer. It depends only on numpy (and pandas for the demo).

## How it works

NASA/JPL calls an asteroid a PHA when two things hold:

- its orbit comes within **0.05 au** of Earth's orbit, a distance called the MOID;
- its absolute magnitude is **H ≤ 22**, which means roughly 140 m across or larger.

Machine-learning studies try to learn this from close-approach records (Earth's point of view). This model takes
the asteroid's point of view:

1. From the asteroid's own orbital elements (*a, e, i*, node and perihelion argument), it computes the exact minimum
   distance δ between the asteroid's path and Earth's orbit, both treated as fixed curves in space. The method: a
   2,048-point grid along the asteroid's orbit, an exact Newton projection onto Earth's ellipse, and golden-section
   refinement of the deepest minima.
2. It computes the hazard index max(δ / 0.05 au, 10^((H−22)/5)), and predicts a PHA if and only if the index is ≤ 1.

It never reads JPL's MOID or PHA fields. The geometry is the same code as [`src/orbit_geometry.py`](../src/orbit_geometry.py),
which is used for every result in the paper.

## Use it

```python
from path_rule_model import predict
pha, score, distance_au = predict(a, e, i, om, w, H)   # numpy arrays; angles in degrees, as in JPL's database
```

`pha` is 0 or 1, `score` is −log of the hazard index (higher means more hazardous, useful for ranking), and
`distance_au` is the path-to-path distance.

## Demo on the NASA/JPL catalogue

```bash
python model/path_rule_model.py
```

It works on a fresh clone: it reads the catalogue snapshot of 11 November 2026 from `datasets/` (or from a CSV path
given on the command line). The output:

```
42,888 near-Earth asteroids (2,551 PHAs according to NASA/JPL), no training, 12.1 s
accuracy 99.86%   F1 0.989   PHAs found 2,513 of 2,551   missed 38   false alarms 20 among 40,337 other asteroids

asteroid        H  path distance  ours         NASA
Apophis     19.09      0.0001 au  PHA          PHA
Bennu       20.21      0.0029 au  PHA          PHA
Toutatis    15.29      0.0066 au  PHA          PHA
Eros        10.40      0.1489 au  not a PHA    not a PHA
Ganymed      9.17      0.3424 au  not a PHA    not a PHA
2024 YR4    23.93      0.0027 au  not a PHA    not a PHA
```

Eros and Ganymed are large, but their paths never come close to Earth's orbit. 2024 YR4's path does, but it is too
small (*H* = 23.9) to count as a PHA.

## Against machine-learning models

The comparison is in [`../comparison/`](../comparison/). It uses NASA's PHA flag as the label for every model and the
306,590 JPL close approaches, split by asteroid. Our model reaches 99.79% accuracy and F1 0.988. Nine models, including
a CNN, GPBoost and a tuned random forest, reach at most 92.6% and F1 0.53. Our model makes 648 errors against at least
22,602, and misses 468 of the 26,784 hazardous approaches against at least 4,614, with exact paired tests at
p < 10⁻¹⁰⁸⁵.

Because NASA's definition is itself geometric, a rule that computes the geometry exactly reaches the ceiling. The
point is that hazard classification needs no training, while models trained on close-approach records fall far
short.
