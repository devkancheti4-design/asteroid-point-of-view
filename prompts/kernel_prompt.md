You are an expert in celestial mechanics and applied mathematics. Help me design the mathematical core of a research paper on classifying potentially hazardous asteroids (PHAs).

## Background
- Prior machine-learning papers classify hazard from Earth's point of view: one row per close approach, with features such as miss distance, relative velocity at that date, and estimated diameter.
- Our approach takes the asteroid's point of view. Each asteroid is described only by its own heliocentric path (osculating orbital elements) and its size. Hazard is judged from the geometry of that path relative to Earth's ORBIT, treated as a fixed curve. Earth's position at any particular time must never enter.
- Ground-truth label (NASA/JPL definition): an asteroid is a PHA if its Earth MOID (minimum distance between the two orbits) is <= 0.05 au AND its absolute magnitude H is <= 22.0.
- Orbits are uncertain: every element has a 1-sigma uncertainty, which can be large for recently discovered objects.

## Inputs per asteroid (NASA JPL SBDB field names; about 43,000 asteroids)
- Orbit: a (au), e, i (deg), om = node longitude Omega (deg), w = argument of perihelion omega (deg), ma = mean anomaly M (deg), epoch (JD TDB), q (au), ad (au)
- Uncertainty: sigma_a, sigma_e, sigma_i, sigma_om, sigma_w, sigma_ma; condition_code (0 = best, 9 = worst); data_arc (days); n_obs_used
- Size: H (mag); diameter (km) and albedo exist for only ~10% of objects
- Earth's orbit: standard heliocentric ecliptic J2000 elements, treated as a fixed curve

## Forbidden inputs (they leak the label or bring back Earth's viewpoint)
moid, pha, any close-approach distance/velocity/date, and Earth's position at a specific time.

## Kernels to design
K1. Path kernel: closed-form heliocentric ecliptic position on the orbit as a function of true anomaly, including the rotation by Omega, i and omega.
K2. Path-to-path proximity kernel: a smooth, differentiable measure of how close the asteroid's orbit comes to Earth's orbit, independent of where either body is along its orbit. Give two versions:
  (a) a cheap closed-form approximation, e.g. from the node distances r± = a(1 - e^2) / (1 ± e cos omega) compared with Earth's heliocentric distance at those longitudes, with a correction for low inclination where nodes are ill-defined;
  (b) an exact or near-exact formulation as a minimization over the two anomalies, with a robust numerical procedure (initialization, avoiding local minima).
  Explain how (b) relates to MOID, why (a) is a valid approximation, and where it fails.
K3. Soft size kernel: map H (and albedo when known) to diameter via D = 1329 km / sqrt(p) * 10^(-H/5), and to a smooth score for "larger than the H <= 22 threshold", using an albedo prior when albedo is unknown.
K4. Uncertainty kernel: propagate the element uncertainties into K2 and output P(hazard) = P(proximity <= 0.05 au AND H <= 22). Give (a) a linearized (delta-method) closed form using partial derivatives of K2(a) with respect to the elements, and (b) a sampling or unscented-transform version. State assumptions (Gaussian errors; independent sigmas when no covariance is given).
K5. Combination kernel: a final dimensionless hazard score in [0, 1] combining K2-K4, in which Earth's time-dependent position cancels out. Name every free parameter and its physical meaning, and explain how to fit it on training data only.

## For every kernel, give
- the formula in LaTeX, with every symbol and unit defined
- Python-style pseudocode using numpy and the field names above
- edge cases: inclination near 0, e near 0 (omega undefined), e near 1, orbits entirely inside or outside Earth's orbit (Atiras, distant Amors), angle wrap-around, degree/radian conversion
- computational cost per asteroid
- a short derivation and the physical meaning
- what is genuinely new versus standard celestial mechanics, citing standard MOID literature (e.g. Gronchi; Wisniowski & Rickman) so the paper does not overclaim

## Rules
- Do not invent numerical results, expected accuracies, or example values for specific asteroids; every number will be checked against NASA JPL data.
- Use only quantities computable from the inputs above.
- If part of this idea cannot work (for example because the label itself is defined by MOID), say so plainly.

## Output format
Markdown with LaTeX: one section per kernel, then a summary table of each kernel's inputs, outputs and free parameters.
