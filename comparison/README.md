# Full model code: our path rule against nine Earth's-view models, with their research papers

[`compare_models.py`](compare_models.py) is the complete code for every method in the paper's model comparison, in
one file that imports nothing from `src/`:

- **our training-free path rule**;
- the three further models of Table IX: a **CNN**, **GPBoost** and a **tuned random forest**;
- the six models of prior work from Table IV: logistic regression, *k*-nearest neighbours, random forest, XGBoost,
  LightGBM and a multilayer perceptron.

It uses the same data, label, features, folds, seeds and hyperparameters as the paper's pipeline. Its report compares
every fold with the paper's stored results. It also runs an exact paired McNemar test of the rule against each model,
using the row-level predictions. The file runs from this repository and also on Kaggle: see
[KAGGLE_GUIDE.md](KAGGLE_GUIDE.md) and the package built by [`make_kaggle_package.py`](make_kaggle_package.py).
The papers behind each method are listed below, and as BibTeX in [references.bib](references.bib).

## Run it

```bash
.venv/bin/python comparison/compare_models.py                    # all six units, then the report
.venv/bin/python comparison/compare_models.py --units C-grouped  # one unit, then the report on what exists
.venv/bin/python comparison/compare_models.py --report           # the report only, from comparison/output/
.venv/bin/python comparison/compare_models.py --max-folds 1      # a quick smoke test
```

| Unit | Rows | Split | Methods | Time on the authors' laptop |
|---|---|---|---|---|
| `B-cad` | 33 Earth approaches within 0.05 au, Sept–Nov 2026 (NASA CAD, the comparison sheet's table) | leave-one-out | rule, CNN, GPBoost, tuned RF | 0.5 min |
| `B-sentry` | 2,241 objects of NASA's Sentry impact-risk list with a catalogue *H* | 5 × 5 repeated stratified | rule, CNN, GPBoost, tuned RF | 10 min |
| `E-grouped` | 306,590 JPL close approaches of 36,817 asteroids (Earth, ≤ 0.5 au, 1910–2024) | 5-fold grouped by asteroid | rule and the six prior-work models | 2 min |
| `E-row` | the same | 5-fold over rows (the leaky prior-work split) | rule and the six prior-work models | 2 min |
| `C-grouped` | the same | 5-fold grouped by asteroid | rule, CNN, GPBoost, tuned RF | 15 min |
| `C-row` | the same | 5-fold over rows | rule, CNN, GPBoost, tuned RF | 25 min |

The label is JPL's PHA flag (SBDB snapshot of 2026-11-11) for every method. The six prior-work models use the
NeoWs-style features of prior studies: absolute magnitude, the diameter range NeoWs derives from it, relative
velocity and miss distance. The CNN, GPBoost and the tuned forest use the CAD table's own columns: miss distance with
its 3σ range, relative and asymptotic speed, and *H*. Our rule uses only the asteroid's orbital elements and *H*.
Every split is identical across methods, and nothing is tuned on a test fold.

## The methods and their research papers

### Ours: the training-free path rule

From the asteroid's own elements (*a, e, i, Ω, ω*), the code computes the exact minimum distance δ between the
asteroid's path and Earth's orbit, taken as a fixed ellipse. It uses a 2,048-point grid over the asteroid's true
anomaly, an exact Newton projection onto Earth's ellipse, and golden-section refinement of the four deepest minima.
The rule flags a PHA when max(δ / 0.05 au, 10^((H−22)/5)) ≤ 1. It needs no training and no close-approach data, and it
never reads JPL's MOID or PHA flag.

| Paper | Role | Link |
|---|---|---|
| Satwik and Kancheti, *The Asteroid's Point of View* (this work) | the method | this repository |
| NASA/JPL CNEOS, *NEO Groups* | the PHA definition: MOID ≤ 0.05 au and *H* ≤ 22 | <https://cneos.jpl.nasa.gov/about/neo_groups.html> |
| Standish, *Keplerian Elements for Approximate Positions of the Major Planets* | Earth's orbit (J2000 mean elements) | <https://ssd.jpl.nasa.gov/planets/approx_pos.html> |
| Gronchi (2005), *Celest. Mech. Dyn. Astron.* 93:295–329 | MOID: the critical points of the distance between two orbits | [doi:10.1007/s10569-005-1623-5](https://doi.org/10.1007/s10569-005-1623-5) |
| Wiśniowski and Rickman (2013), *Acta Astron.* 63:293–307 | MOID: a fast geometric method | — |
| Hedo, Ruíz and Peláez (2018), *MNRAS* 479:3288–3299 | MOID: an effective method | [doi:10.1093/mnras/sty1598](https://doi.org/10.1093/mnras/sty1598) |

### Model 1: one-dimensional CNN

The network is `Conv1d(1→32, k=3)` → ReLU → `Conv1d(32→64, k=3)` → ReLU → `Linear(64·d→64)` → ReLU →
`Dropout(0.2)` → `Linear(64→1)`, applied to the standardized feature vector of length *d*. It is trained with Adam
(learning rate 10⁻³, weight decay 10⁻⁵) on binary cross-entropy. On the 306,590 approaches it uses batches of 2,048,
at most 25 epochs, and early stopping (patience 3) on 10% of the training rows, with whole asteroids held out when the
split is grouped. On Sentry it uses batches of 64, at most 200 epochs and patience 15. On the 33-row table it trains
full-batch for 300 epochs. Imputation and scaling are fitted on the training fold, the threshold is 0.5, and PyTorch
runs with deterministic algorithms and 4 threads.

| Paper | What the code takes from it | Link (free copy) |
|---|---|---|
| LeCun, Bottou, Bengio and Haffner (1998), *Gradient-based learning applied to document recognition*, Proc. IEEE 86(11):2278–2324 | convolutional networks trained by gradient descent | [doi:10.1109/5.726791](https://doi.org/10.1109/5.726791) ([author's PDF](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)) |
| Kiranyaz et al. (2021), *1D convolutional neural networks and applications: A survey*, Mech. Syst. Signal Process. 151:107398 | 1-D convolutions over a feature vector | [doi:10.1016/j.ymssp.2020.107398](https://doi.org/10.1016/j.ymssp.2020.107398) ([arXiv:1905.03554](https://arxiv.org/abs/1905.03554)) |
| Kingma and Ba (2015), *Adam: A Method for Stochastic Optimization*, ICLR 2015 | the optimizer | [arXiv:1412.6980](https://arxiv.org/abs/1412.6980) |
| Srivastava, Hinton, Krizhevsky, Sutskever and Salakhutdinov (2014), *Dropout*, JMLR 15(56):1929–1958 | dropout in the dense head | [JMLR](https://jmlr.org/papers/v15/srivastava14a.html) |
| Paszke et al. (2019), *PyTorch*, NeurIPS 32 | the library | [arXiv:1912.01703](https://arxiv.org/abs/1912.01703) |

In asteroid science, published CNNs that we found classify survey images, not hazard tables. Examples are ATLAS's
detection classifier (Chyba Rabeendran and Denneau 2021, PASP 133:034501,
[doi:10.1088/1538-3873/abc900](https://doi.org/10.1088/1538-3873/abc900)) and Bacu et al. 2023 (Aerospace 10:752,
[doi:10.3390/aerospace10090752](https://doi.org/10.3390/aerospace10090752)). We did not find a published PHA study that
applies a CNN to close-approach tables (search of 2026-11-12).

### Model 2: GPBoost

GPBoost models the latent score as *F*(**x**) + *b*<sub>asteroid</sub>. Here *F* is an ensemble of boosted trees
(learning rate 0.05; 63 leaves, a minimum of 20 rows per leaf and 300 rounds on the 306,590 approaches). The term
*b* is a Gaussian random effect per asteroid, with a Bernoulli-probit likelihood, fitted jointly with the trees. An
asteroid that is absent from the training rows gets *b* = 0 at prediction time. This is why GPBoost reaches 99.90%
under the leaky row split but never predicts a PHA for new asteroids (F1 0.000 on the grouped folds). On the Sentry and
33-row tables no asteroid repeats, so the random effect is dropped and plain boosted trees with a logistic objective
are used (15 or 7 leaves, 200 rounds).

| Paper | What the code takes from it | Link (free copy) |
|---|---|---|
| Sigrist (2022), *Gaussian Process Boosting*, JMLR 23(232):1–46 | boosting combined with Gaussian-process and grouped random-effects models | [JMLR](https://jmlr.org/papers/v23/20-322.html) ([arXiv:2004.02653](https://arxiv.org/abs/2004.02653)) |
| Sigrist (2023), *Latent Gaussian Model Boosting*, IEEE TPAMI 45(2):1894–1905 | the extension to non-Gaussian likelihoods, including the Bernoulli-probit used here | [doi:10.1109/TPAMI.2022.3168152](https://doi.org/10.1109/TPAMI.2022.3168152) (open access, CC BY 4.0; [arXiv:2105.08966](https://arxiv.org/abs/2105.08966)) |
| Friedman (2001), *Greedy function approximation: A gradient boosting machine*, Ann. Stat. 29(5):1189–1232 | gradient boosting | [doi:10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451) |
| Ke et al. (2017), *LightGBM*, NeurIPS 30:3146–3154 | the tree learner that GPBoost builds on | [NeurIPS](https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) |

We did not find a published asteroid-hazard study that uses GPBoost (search of 2026-11-12).

### Model 3: random forest tuned by randomized search

A randomized search scored by F1 picks the forest's settings. It samples 8 candidates on the 306,590 approaches (on
40,000 training rows, whole asteroids when grouped) and 10 on Sentry. The search space is: number of trees {200, 400},
maximum depth {none, 8, 16, 32}, minimum leaf size {1, 2, 4, 8}, features per split {√d, 0.5, 1.0}, and class weight
{none, balanced, balanced per tree}. The inner 3-fold split lies inside the training fold only, grouped by asteroid
when the outer split is. The best setting is then refitted on the whole training fold. The 33-row table has too few
positives to tune, so it uses library defaults with 400 trees.

| Paper | What the code takes from it | Link (free copy) |
|---|---|---|
| Breiman (2001), *Random Forests*, Mach. Learn. 45(1):5–32 | the model | [doi:10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324) |
| Bergstra and Bengio (2012), *Random Search for Hyper-Parameter Optimization*, JMLR 13(10):281–305 | randomized search | [JMLR](https://jmlr.org/papers/v13/bergstra12a.html) |
| Probst, Wright and Boulesteix (2019), *Hyperparameters and tuning strategies for random forest*, WIREs DMKD 9(3):e1301 | which forest settings to tune | [doi:10.1002/widm.1301](https://doi.org/10.1002/widm.1301) ([arXiv:1804.03515](https://arxiv.org/abs/1804.03515)) |
| Pedregosa et al. (2011), *Scikit-learn*, JMLR 12:2825–2830 | the library | [JMLR](https://jmlr.org/papers/v12/pedregosa11a.html) |

Random forests are the most common model in PHA studies. Malakouti et al. 2023 (MethodsX 11:102337,
[doi:10.1016/j.mex.2023.102337](https://doi.org/10.1016/j.mex.2023.102337)), Nair and Babu 2025 (Procedia Comput. Sci.
259:1682–1692, [doi:10.1016/j.procs.2025.04.123](https://doi.org/10.1016/j.procs.2025.04.123)) and Chandra 2025
([arXiv:2508.15106](https://arxiv.org/abs/2508.15106)) all use them on NeoWs-derived tables, and none evaluates with
splits grouped by asteroid.

### The six models of prior work

These are logistic regression, *k*-NN (*k* = 5), a random forest (200 trees), XGBoost (400 trees, depth 8), LightGBM
(400 trees, 63 leaves) and an MLP (64-32, early stopping). They are implemented with scikit-learn, XGBoost and
LightGBM, with the settings fixed in advance as in `src/earth_view.py`. The model papers are Breiman 2001, Chen and
Guestrin 2016 (XGBoost, [doi:10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)), Ke et al. 2017 and
Pedregosa et al. 2011. The PHA studies that use them are in the paper's Table I.

## What the run shows

On the honest splits (grouped by asteroid, and the two small tables, where no object repeats), the rule beats every
model in every fold on accuracy, F1 and MCC. Only under the leaky row split does GPBoost score higher, because its
random effect recognises asteroids it has already seen.

The exact paired McNemar test uses the actual row-level predictions of the 306,590 grouped test rows. The rule makes
648 errors, against at least 22,602 for every model (p < 10⁻⁶²⁷⁸). It misses 468 of the 26,784 PHA approaches,
against at least 4,614, from the tuned random forest (p < 10⁻¹⁰⁸⁵). For every model, the confusion counts equal those
that the paper decoded from its stored metrics for Table X. Every p lies below the paper's least-favourable bound.

Run from the repository, the file reproduces the paper's stored results in every fold of all six units: every
prediction and every metric. The one exception is GPBoost's ROC and PR AUC, which vary between runs by at most
2 × 10⁻⁵ because its random-effect variance is estimated with multithreaded sums (see `proofs/REPRODUCTION.md`). The
report of this verification run is in [`output/report.txt`](output/report.txt), and `proofs/verify_all.sh` rechecks the
saved outputs against `results/` (check 12).
