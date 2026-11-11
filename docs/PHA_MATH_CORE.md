# Mathematical core: classifying potentially hazardous asteroids from the asteroid's own orbit

Each asteroid enters only as its heliocentric path (osculating elements) and its absolute magnitude. Hazard is judged from the geometry of that path relative to Earth's **orbit**, treated as a fixed curve. No kernel reads the mean anomaly, the epoch, Earth's position at any time, a close-approach record, `moid` or `pha`.

Every formula below has a reference implementation (`pha_kernels.py`, `pha_pipeline.py`) and property tests (`tests/`). No number in this document is a result about a real asteroid. The numbers are of three kinds: published constants (cited), quantities derived from those constants (shown with their derivation), and checks of the implementation on synthetic orbits, labelled as such in Appendix A.

---

## 0. What this design can and cannot do (read first)

1. **The label is a deterministic function of the inputs.** The PHA label is $[\mathrm{MOID}\le 0.05\ \mathrm{au}]\wedge[H\le 22.0]$. The MOID is a function of the five shape-and-orientation elements $(a,e,i,\Omega,\omega)$ and Earth's orbit. Kernel K2(b) computes that function.
   - Up to the choice of Earth curve and numerical tolerance (item 7), K2(b) followed by the two thresholds *reproduces the label by construction*.
   - So "classifying PHAs from orbital elements" is not a learning problem. A machine-learning classifier trained on this label can at best approximate a known closed procedure, and its accuracy measures that approximation, not a discovery about hazard.
   - What this design can honestly contribute:
     - (a) cheap **certified** screening that settles most objects without the full MOID;
     - (b) a **probability** that the label holds, given the stated orbit uncertainty;
     - (c) an explicit **unresolved** class for objects whose label could flip within that uncertainty.
2. **Size cannot add information about the label.** The label's size test is $H\le 22.0$ on the catalogue $H$, which is an input. Diameter and albedo (K3) can describe physical size, but any size score that differs from $\mathbf 1[H\le22]$ can only lower agreement with the label.
3. **The probability cannot be validated against the label.** The label is computed from the nominal orbit and carries no uncertainty.
   - Checking P(hazard) needs either internal consistency (the delta method against sampling, Appendix A) or later, better orbits of the same objects. Later orbits are not among the inputs.
4. **Element sigmas without covariance are not enough at low inclination.**
   - For $i\to0$, $\Omega$ and $\omega$ are individually ill-determined. Only $\varpi=\Omega+\omega$ is well determined, so their errors are strongly anti-correlated.
   - Treating the listed sigmas as independent then overstates the uncertainty of the curve, often grossly. With the listed inputs this cannot be repaired. Such objects end up "unresolved", which is safe but uninformative.
   - The analogous case at $e\to0$ is benign for the MOID. There the ill-determined pair is $\omega$ and $M$; the MOID ignores $M$, and $\partial\,\mathrm{MOID}/\partial\omega=O(e)$, so the product $e\,\sigma_\omega$ is what enters (K4).
   - For short arcs, the uncertainty region is also strongly non-Gaussian in elements (Milani et al. 2005).
5. **`ma` and `epoch` carry no hazard information.** The MOID does not depend on the mean anomaly, so $\sigma_M$ is unused as well. In a learned model these fields could only pick up catalogue artefacts, such as the era of discovery.
6. **Observation metadata leaks the label.** `condition_code`, `data_arc` and `n_obs_used` correlate with PHA status through follow-up selection: objects that pass near Earth's orbit are prioritised for more observations.
   - As classifier features they would leak the label through human attention. Here they appear only as diagnostics of whether the Gaussian model is credible (item 4), never inside a score.
7. **The Earth curve differs slightly from the label's.**
   - We fix Earth's orbit to the J2000 Earth–Moon-barycentre elements below.
   - The catalogue MOID is computed from osculating orbits at each solution's epoch. Its exact Earth model (Earth or Earth–Moon barycentre, which ephemeris) must be taken from JPL's documentation.
   - The difference is a small systematic offset. It can flip labels only for objects whose MOID lies very close to 0.05 au. Its size must be **measured** on the data, not assumed; K5's parameter $\beta$ exists for this.

## Notation, units and constants

| symbol | meaning | unit | SBDB field |
|---|---|---|---|
| $a,\ e,\ q,\ Q$ | semi-major axis, eccentricity, perihelion and aphelion distance | au, –, au, au | `a`, `e`, `q`, `ad` |
| $i,\ \Omega,\ \omega$ | inclination, longitude of ascending node, argument of perihelion | rad (converted once from deg) | `i`, `om`, `w` |
| $p=q(1+e)$ | semi-latus rectum (from $q$: well conditioned as $e\to1$) | au | – |
| $f,\ u=\omega+f$ | true anomaly, argument of latitude | rad | – |
| $H$ | absolute magnitude | mag | `H` |
| $p_V,\ D$ | geometric albedo, diameter | –, km | `albedo`, `diameter` |
| $\sigma_x$ | 1-sigma of element $x$ | as $x$ | `sigma_a` … `sigma_w` |
| $d_0=0.05$ au, $H_0=22.0$ | PHA thresholds (definition; never fitted) | au, mag | – |
| $\Phi$ | standard normal CDF | – | – |

Earth's orbit, a fixed curve. These are the J2000 mean ecliptic elements of the Earth–Moon barycentre from Standish's "Approximate Positions of the Planets", Table 1, valid 1800–2050:
- $a_\oplus=1.00000261$ au, $e_\oplus=0.01671123$ and $\varpi_\oplus=102.93768193^\circ$;
- $i_\oplus=-0.00001531^\circ$, which we set to $0$, so the curve lies in the reference plane. The vertical offset dropped is below $3\times10^{-7}$ au.

Derived constants:
- $p_\oplus=a_\oplus(1-e_\oplus^2)=0.9997233$ au;
- $b_\oplus=a_\oplus\sqrt{1-e_\oplus^2}$;
- $q_\oplus=0.9832913$ au and $Q_\oplus=1.0167139$ au.

Earth's heliocentric distance at ecliptic longitude $\lambda$:

$$R_\oplus(\lambda)=\frac{p_\oplus}{1+e_\oplus\cos(\lambda-\varpi_\oplus)},\qquad R_\oplus'(\lambda)=\frac{p_\oplus e_\oplus\sin(\lambda-\varpi_\oplus)}{\bigl(1+e_\oplus\cos(\lambda-\varpi_\oplus)\bigr)^2},\qquad K_\oplus=\frac{p_\oplus e_\oplus}{(1-e_\oplus)^2}\ \ge\ \max_\lambda|R_\oplus'(\lambda)| .$$

Numerically, $K_\oplus=0.017279$ au/rad, against a true maximum of $0.016716$.

---

## K1. Path kernel

**Formula.**

$$\mathbf r(f)=\frac{p}{1+e\cos f}\,\bigl(\cos f\,\hat{\mathbf P}+\sin f\,\hat{\mathbf Q}\bigr),$$

$$\hat{\mathbf P}=\begin{pmatrix}\cos\Omega\cos\omega-\sin\Omega\sin\omega\cos i\\ \sin\Omega\cos\omega+\cos\Omega\sin\omega\cos i\\ \sin\omega\sin i\end{pmatrix},\quad
\hat{\mathbf Q}=\begin{pmatrix}-\cos\Omega\sin\omega-\sin\Omega\cos\omega\cos i\\ -\sin\Omega\sin\omega+\cos\Omega\cos\omega\cos i\\ \cos\omega\sin i\end{pmatrix},\quad
\hat{\mathbf h}=\begin{pmatrix}\sin\Omega\sin i\\ -\cos\Omega\sin i\\ \cos i\end{pmatrix}.$$

$[\hat{\mathbf P},\hat{\mathbf Q},\hat{\mathbf h}]=R_3(\Omega)R_1(i)R_3(\omega)$ is the perifocal-to-ecliptic rotation, and $\mathbf r$ is in au. K2 also uses the cylindrical form, with $u=\omega+f$ and $r=p/(1+e\cos f)$:

$$\rho=r\sqrt{\cos^2u+\sin^2u\cos^2 i},\qquad \lambda=\Omega+\operatorname{atan2}(\sin u\cos i,\ \cos u),\qquad z=r\sin u\sin i .$$

**Derivation and meaning.** The conic $r=p/(1+e\cos f)$ is written in the orbital plane and then rotated by $\omega$ about the orbit normal, by $i$ about the node line, and by $\Omega$ about the ecliptic pole (e.g. Murray & Dermott 1999). The curve $\{\mathbf r(f)\}$ is the asteroid's path. The mean anomaly would place the body *on* the path at a time; K1 deliberately has no time.

**Pseudocode.**
```python
import numpy as np
def elements(row):                                   # SBDB fields -> au, rad (the only deg->rad step)
    e, q = row['e'], row['q']
    return dict(a=row['a'], e=e, q=q, p=q*(1+e),
                i=np.deg2rad(row['i']), om=np.deg2rad(row['om'] % 360), w=np.deg2rad(row['w'] % 360))

def basis(i, om, w):
    ci, si, cO, sO, cw, sw = np.cos(i), np.sin(i), np.cos(om), np.sin(om), np.cos(w), np.sin(w)
    P = np.array([cO*cw - sO*sw*ci, sO*cw + cO*sw*ci, sw*si])
    Q = np.array([-cO*sw - sO*cw*ci, -sO*sw + cO*cw*ci, cw*si])
    return P, Q

def path(el, f):                                     # f: array of true anomalies (rad) -> (3, len(f)) au
    P, Q = basis(el['i'], el['om'], el['w'])
    r = el['p'] / (1 + el['e']*np.cos(f))
    return r * (np.cos(f)*P[:, None] + np.sin(f)*Q[:, None])
```

**Edge cases.**

| case | handling |
|---|---|
| $i\to0$ | Nothing is divided by $\sin i$. At $i=0$ the position depends only on $\Omega+\omega+f$, so an ill-defined $\Omega$ moves no point of the curve. |
| $e\to0$ | Nothing is divided by $e$. The position depends only on $u=\omega+f$. |
| $e\to1$ | $p=q(1+e)$ from the catalogue $q$; $a(1-e^2)$ loses precision as $a\to\infty$. |
| $e\ge1$ | The same formula, with $f$ restricted to $\lvert f\rvert<\arccos(-1/e)$. |
| retrograde $i>90^\circ$ | No special case. |
| angle wrap | Only trigonometric functions of the angles are used, so wrap-around is automatic. Angles are reduced mod $360^\circ$ only for display. |
| deg→rad | Exactly once, in `elements()`, for both the angles and their sigmas. |

**Cost.** 6 trigonometric calls and about 20 flops per asteroid for the basis, then 2 trigonometric calls and about 12 flops per point.

**New vs standard.** Nothing new: textbook two-body geometry.

---

## K2. Path-to-path proximity kernel

The target quantity is the minimum orbit intersection distance:

$$\mathrm{MOID}=\min_{f,\psi}\ \bigl\|\mathbf r(f)-\mathbf r_\oplus(\psi)\bigr\|.$$

It is a property of the two curves, so it is independent of where either body is along its orbit. K2 has three layers:
- (a) closed forms that bound or approximate it;
- (a′) a one-dimensional certified bracket;
- (b) the exact value.

### K2(a). Closed forms: radial gap, nodal distances, linearised node correction

**Radial gap (lower bound).**

$$L_0=\max\bigl(0,\ q-Q_\oplus,\ q_\oplus-Q\bigr)\ \le\ \mathrm{MOID},\qquad Q=q\frac{1+e}{1-e}\ (Q=\infty\ \text{for}\ e\ge1).$$

*Proof.* $\|\mathbf r-\mathbf r_\oplus\|\ge\bigl|\,|\mathbf r|-|\mathbf r_\oplus|\,\bigr|$, with $|\mathbf r|\in[q,Q]$ and $|\mathbf r_\oplus|\in[q_\oplus,Q_\oplus]$.

$L_0>d_0$ settles orbits wholly outside or wholly inside Earth's: Amors with $q>Q_\oplus+d_0=1.06671$ au and Atiras with $Q<q_\oplus-d_0=0.93329$ au.

**Nodal distances (upper bound).** The asteroid crosses the reference plane at $f=-\omega$ (ascending node, longitude $\Omega$) and at $f=\pi-\omega$ (descending node, longitude $\Omega+\pi$):

$$r_\pm=\frac{p}{1\pm e\cos\omega},\qquad \delta_\pm=r_\pm-R_\oplus(\lambda_\pm),\quad \lambda_+=\Omega,\ \lambda_-=\Omega+\pi,\qquad U_{\rm node}=\min\bigl(|\delta_+|,|\delta_-|\bigr)\ \ge\ \mathrm{MOID}.$$

*Proof.* At a node, the asteroid's point and Earth's orbit point at the same longitude are two actual points of the two curves, on the same ray from the Sun, a distance $|\delta_\pm|$ apart. This holds because Earth's curve lies in the reference plane. For $e\ge1$, a node exists only if $1\pm e\cos\omega>0$.

**Linearised node correction (approximation).** Write $\nu$ for the argument of latitude measured from a node. Near that node:
- the radial gap is $g(\nu)\approx\delta+s\,\nu$;
- the height is $|z(\nu)|\approx r\sin i\,|\nu|$.

Minimising $g^2+z^2$ gives

$$d_{\rm lin}=\frac{|\delta|\ r\sin i}{\sqrt{s^2+r^2\sin^2 i}},\qquad \nu^*=-\frac{s\,\delta}{s^2+r^2\sin^2 i},$$

$$s_+=-\frac{r_+^2\,e\sin\omega}{p}-\cos i\,R_\oplus'(\Omega),\qquad s_-=+\frac{r_-^2\,e\sin\omega}{p}-\cos i\,R_\oplus'(\Omega+\pi).$$

Here $s$ (au/rad) is $d(\rho-R_\oplus(\lambda))/d\nu$ at the node. It uses $dr/df=r^2e\sin f/p$ and $d\lambda/du=\cos i$ at a node. $d_{\rm lin}$ is taken only where $|\nu^*|\le\nu_{\max}$; the default $\nu_{\max}=0.2$ rad is a policy choice. Then $d_a=\min$ over valid nodes, and $d_a$ is undefined if neither node is valid.

**Smooth surrogate (for gradient-based use).** A smoothed absolute value and a soft minimum:

$$D_{\varepsilon,\tau}=-\tau\ln\!\Bigl(e^{-\sqrt{\delta_+^2+\varepsilon^2}/\tau}+e^{-\sqrt{\delta_-^2+\varepsilon^2}/\tau}\Bigr).$$

It is differentiable everywhere and lies within $\tau\ln2+\varepsilon$ of $U_{\rm node}$. Both $\varepsilon$ and $\tau$ are in au and are free parameters.

**Why (a) is valid.**
- $L_0$ and $U_{\rm node}$ are rigorous bounds.
- If the inclination is not small, the curves can come close only near the line of nodes, because away from it the height $|z|=r|\sin u|\sin i$ grows. The first-order expansion makes this quantitative:

$$\mathrm{MOID}\approx d_{\rm lin}=U_{\rm node}\cdot\frac{r\sin i}{\sqrt{s^2+r^2\sin^2 i}}\le U_{\rm node}.$$

  Equality holds when the radial gap is stationary at the node ($s=0$).
- The ratio $\epsilon=|s|/(r\sin i)$ is the failure parameter. For $\epsilon\ll1$, $U_{\rm node}\approx\mathrm{MOID}$.

**Where (a) fails.**
- (i) Low inclination, where $\epsilon\gtrsim1$. The closest approach moves away from the node, $U_{\rm node}$ can greatly overestimate the MOID, and $|\nu^*|$ grows until $d_{\rm lin}$ is invalid.
- (ii) Near-tangent geometry, where $s\approx0$ and $i$ is small. The curves run close together along an arc.
- (iii) A missing node, for $e\ge1$.
- (iv) A closest approach unrelated to either node.

In all these cases K2(a′)/(b) take over. The cheap screen uses $L_0$ and $U_{\rm node}$ only *as bounds*, so its failures cost time, never a wrong label.

**Pseudocode.**
```python
D0 = 0.05
A_E, E_E, VARPI_E = 1.00000261, 0.01671123, np.deg2rad(102.93768193)
P_E = A_E*(1 - E_E**2); qE, QE = A_E*(1 - E_E), A_E*(1 + E_E)
R_E  = lambda lam: P_E / (1 + E_E*np.cos(lam - VARPI_E))
dR_E = lambda lam: P_E*E_E*np.sin(lam - VARPI_E) / (1 + E_E*np.cos(lam - VARPI_E))**2

def radial_gap(el):                                  # vectorised over asteroids
    q, e = el['q'], el['e']
    Q = np.where(e < 1, q*(1 + e)/np.where(e < 1, 1 - e, 1), np.inf)
    return np.maximum(0, np.maximum(q - QE, qE - Q))

def nodal(el, nu_max=0.2):
    p, e, i, om, w = (el[k] for k in ('p', 'e', 'i', 'om', 'w'))
    out = {}
    for sgn, lam in ((+1, om), (-1, om + np.pi)):
        den = 1 + sgn*e*np.cos(w)
        r = np.where(den > 0, p/np.where(den > 0, den, 1), np.inf)
        delta = r - R_E(lam)
        s = -sgn*r**2*e*np.sin(w)/p - np.cos(i)*dR_E(lam)
        q2 = s**2 + (r*np.sin(i))**2
        q2 = np.where(q2 > 0, q2, np.nan)            # i = 0 and s = 0: no linearisation exists
        dlin = np.abs(delta)*r*np.sin(i)/np.sqrt(q2)
        nu = -s*delta/q2
        out[sgn] = (delta, np.where(np.abs(nu) <= nu_max, dlin, np.nan))
    U_node = np.minimum(np.abs(out[+1][0]), np.abs(out[-1][0]))
    d_a = np.fmin(out[+1][1], out[-1][1])
    return U_node, d_a, out[+1][0], out[-1][0]
```

**Cost.** $L_0$: about 6 flops. $U_{\rm node}$ and $d_{\rm lin}$: 6 trigonometric calls and about 40 flops. Vectorised over the whole catalogue, the closed forms take microseconds in total per thousand objects (Appendix A).

### K2(a′). A certified one-dimensional bracket (same-longitude bounds)

**Lemma.** Take any point with cylindrical coordinates $(\rho,\lambda,z)$, and put

$$g=\rho-R_\oplus(\lambda),\qquad c=\frac{4\rho\,q_\oplus}{\pi^2},\qquad \kappa=\frac{c}{c+K_\oplus^2}\in(0,1].$$

Its distance $d$ to Earth's orbit satisfies

$$z^2+\kappa\,g^2\ \le\ d^2\ \le\ z^2+g^2 .$$

*Proof.* Let Earth's point be at longitude $\lambda+\Delta$ with $|\Delta|\le\pi$ and radius $R'$. Then

$$d^2=z^2+(\rho-R')^2+2\rho R'(1-\cos\Delta).$$

- Taking $\Delta=0$ gives the upper bound.
- $|R'-R_\oplus(\lambda)|\le K_\oplus|\Delta|$, so $(\rho-R')^2\ge\bigl((|g|-K_\oplus|\Delta|)_+\bigr)^2$.
- $R'\ge q_\oplus$ and $\sin(|\Delta|/2)\ge|\Delta|/\pi$ give $2\rho R'(1-\cos\Delta)\ge c\,\Delta^2$.
- Minimising $(|g|-K_\oplus t)_+^2+c\,t^2$ over $t\ge0$ gives $g^2c/(c+K_\oplus^2)$. $\square$

**Bracket.** Taking the minimum over the asteroid's true anomaly:

$$L_{\rm lon}=\min_f\sqrt{z^2+\kappa g^2}\ \le\ \mathrm{MOID}\ \le\ U_{\rm lon}=\min_f\sqrt{z^2+g^2}.$$

- For a circular Earth orbit ($e_\oplus=0$), $K_\oplus=0$ and $\kappa=1$, and both sides equal the MOID. This reduction of the MOID to a 1-D minimisation is the classical circular-orbit case.
- For Earth's actual orbit, at $\rho\approx1$ au, $1-\sqrt\kappa\approx K_\oplus^2/(2c)=3.7\times10^{-4}$.
- So the bracket is narrower than $3.7\times10^{-4}\,|g|$. For $|g|\le d_0$, that is below $2\times10^{-5}$ au.
- $U_{\rm lon}\le U_{\rm node}$, because the nodes are points of the same minimisation.

**Numerical procedure.** Use a grid of $N$ true anomalies (default $N=720$) plus the two node anomalies, exactly. Refine the lowest few local minima of each function with bounded Brent.
- The upper bound is valid at any grid point, being a real distance.
- A strictly rigorous lower bound under sampling subtracts (Lipschitz constant × half the grid step). The reference implementation refines instead and checks the result against K2(b).

**Cost.** $N$ points × (4 trigonometric calls + about 30 flops), plus about 30 evaluations per refined basin.

### K2(b). Exact MOID: nested minimisation with a certificate

**Formula.** Parametrise Earth's orbit by its eccentric anomaly $\psi$:

$$\mathbf r_\oplus(\psi)=a_\oplus(\cos\psi-e_\oplus)\,\hat{\mathbf x}_\oplus+b_\oplus\sin\psi\,\hat{\mathbf y}_\oplus,$$

where $\hat{\mathbf x}_\oplus=(\cos\varpi_\oplus,\sin\varpi_\oplus,0)$ and $\hat{\mathbf y}_\oplus=(-\sin\varpi_\oplus,\cos\varpi_\oplus,0)$.

**Inner problem (point to Earth's ellipse).** For a point with projection $(X,Y)$ in Earth's perifocal frame and height $Z$, the stationarity condition is

$$F(\psi)=(a_\oplus X+a_\oplus^2e_\oplus)\sin\psi-b_\oplus Y\cos\psi-(a_\oplus^2-b_\oplus^2)\sin\psi\cos\psi=0 .$$

- Solve it by Newton from the central angle $\psi_0=\operatorname{atan2}\bigl(Y/b_\oplus,\ (X+a_\oplus e_\oplus)/a_\oplus\bigr)$; 7 iterations in the reference code.
- The minimum is unique for every point whose projection lies outside the evolute of Earth's ellipse. The evolute is an astroid of half-size $a_\oplus e_\oplus^2=2.8\times10^{-4}$ au about the ellipse's centre. Points that close to the centre are about 1 au from the curve, so they never matter for a 0.05 au threshold.

**Outer problem.**
- $G(f)=\min_\psi\|\mathbf r(f)-\mathbf r_\oplus(\psi)\|^2$ is smooth wherever the inner minimum is unique and non-degenerate. This includes orbit crossings, because $G$ is the *squared* distance.
- $\mathrm{MOID}^2=\min_f G(f)$.
- Grid: $N$ true anomalies plus the two node anomalies. Keep the `keep` lowest local minima (default 4), refine each by bounded Brent to $10^{-13}$ rad, and take the least.

**Certificate.** Accept only if

$$\max(L_0,L_{\rm lon})-\eta\ \le\ \mathrm{MOID}\ \le\ \min(U_{\rm node},U_{\rm lon})+\eta .$$

$\eta=2\times10^{-8}$ au is the resolution of a distance computed from squared differences of 1-au vectors, about $\sqrt{\epsilon_{\rm mach}}$ au. If the certificate fails, retry once with $4N$. If it still fails, the certificate is marked BAD and the decision law routes the object to a human.

**Signed MOID** (needed by K4). At the minimising pair $(f^*,\psi^*)$, take $\boldsymbol\Delta=\mathbf r(f^*)-\mathbf r_\oplus(\psi^*)$ and tangents $\boldsymbol\tau=d\mathbf r/df$, $\boldsymbol\tau_\oplus=d\mathbf r_\oplus/d\psi$:

$$\hat{\mathbf n}=\frac{\boldsymbol\tau_\oplus\times\boldsymbol\tau}{\|\boldsymbol\tau_\oplus\times\boldsymbol\tau\|},\qquad \tilde d=\hat{\mathbf n}\cdot\boldsymbol\Delta,\qquad |\tilde d|=\mathrm{MOID}.$$

At a stationary point $\boldsymbol\Delta$ is orthogonal to both tangents, hence parallel to $\hat{\mathbf n}$. $\tilde d$ changes sign smoothly when the curves cross, provided the tangents are not parallel (Gronchi & Tommei 2007). We require $\sin\angle(\boldsymbol\tau,\boldsymbol\tau_\oplus)\ge\sin\theta_{\min}$, with default $\sin\theta_{\min}=0.05$.

**How (b) relates to the MOID.** (b) *is* the MOID, with Earth's orbit replaced by the fixed J2000 curve. Its only differences from the catalogue value come from that replacement (§0, item 7) and from the tolerance $\eta$. This is why the label is reproducible from the inputs (§0, item 1).

**Robustness (initialisation, local minima).**
- (1) The squared distance between two confocal conics has finitely many critical points (Gronchi 2002, 2005). A grid that includes both node anomalies, refined in every low basin, finds them in practice.
- (2) The inner problem is unimodal for every relevant point, because Earth's orbit is nearly circular.
- (3) The independent bracket of K2(a)/(a′) certifies the global minimum. If the outer search had missed it, the computed value would exceed $U_{\rm lon}$ and the certificate would fail.
- The MOID as a function of the elements is continuous but has kinks where two local minima exchange. Where differentiability across such kinks is needed, use a soft minimum over the local minima, $-\tau\ln\sum_k e^{-d_k/\tau}$.

**Pseudocode.**
```python
XE = np.array([np.cos(VARPI_E), np.sin(VARPI_E), 0.]); YE = np.array([-np.sin(VARPI_E), np.cos(VARPI_E), 0.])
B_E = A_E*np.sqrt(1 - E_E**2)

def nearest_on_earth(X3, iters=7):                   # X3: (3, n) points -> eccentric anomaly, squared distance
    X, Y, Z = XE @ X3, YE @ X3, X3[2]
    psi = np.arctan2(Y/B_E, (X + A_E*E_E)/A_E)
    c1, c2, c3 = A_E*X + A_E**2*E_E, B_E*Y, A_E**2 - B_E**2
    for _ in range(iters):
        s, c = np.sin(psi), np.cos(psi)
        psi -= (c1*s - c2*c - c3*s*c) / (c1*c + c2*s - c3*(c*c - s*s))
    return psi, (X - A_E*(np.cos(psi) - E_E))**2 + (Y - B_E*np.sin(psi))**2 + Z**2

def moid(el, N=720, keep=4):
    w = el['w']
    f = np.unique(np.r_[np.linspace(-np.pi, np.pi, N, endpoint=False),
                        (np.array([-w, np.pi - w]) + np.pi) % (2*np.pi) - np.pi])
    G = lambda ff: nearest_on_earth(path(el, np.atleast_1d(ff)))[1]
    g = G(f)
    loc = np.nonzero((g <= np.roll(g, 1)) & (g <= np.roll(g, -1)))[0]
    loc = loc[np.argsort(g[loc])][:keep]
    def bracket(k):                                  # neighbouring grid points, wrapping around the circle
        return (f[k-1] if k > 0 else f[-1] - 2*np.pi), (f[k+1] if k < len(f) - 1 else f[0] + 2*np.pi)
    best = min(scipy.optimize.minimize_scalar(lambda t: G(t)[0], bounds=bracket(k), method='bounded',
                                              options=dict(xatol=1e-13)).fun for k in loc)
    return np.sqrt(best)                             # then the certificate, the signed MOID, the runner-up
```

**Edge cases.**

| case | handling |
|---|---|
| $i\to0$ | Coplanar orbits: the 2-D search does not rely on nodes, and the node anomalies are just two more grid points. |
| $e\to0$ | Fine; $\omega$ only shifts the grid. |
| $e\to1$, $e\ge1$ | $p=q(1+e)$; the grid is restricted to $\lvert f\rvert<\arccos(-1/e)$. |
| crossing orbits (MOID $\to0$) | $G$ stays smooth; the resolution floor is $\eta$. |
| Atiras, distant Amors | Settled by $L_0>d_0$ before K2(b) is ever run. |
| near-coincident orbits | Many shallow minima. `keep` can be raised; the certificate catches a miss. |
| angle wrap | Grids live on $(-\pi,\pi]$, and refinement brackets wrap. |

**Cost.** $N\times$ (one K1 point + 7 Newton steps of 2 trigonometric calls and about 12 flops), plus about 30 evaluations per refined basin. The certificate adds one K2(a′).

**New vs standard (K2).**
- Nodal distances, the radial gap and MOID computation are standard. MOID methods include Sitarski (1968), Kholshevnikov & Vassiliev (1999), Gronchi (2002, 2005, algebraic), Armellin et al. (2010, rigorous global optimisation), Wiśniowski & Rickman (2013, geometric), Hedo et al. (2018, 2020) and Baluev & Mikryukov (2019, error-controlled).
- $d_{\rm lin}$ is an elementary first-order expansion; analytical MOID approximations near the nodes go back at least to Bonanno (2000).
- The circular-orbit 1-D reduction is classical.
- The explicit two-sided bracket for a slightly eccentric planet (the lemma) follows from an elementary Lipschitz argument. We have not seen it stated in this form in the works cited, but it should be presented as an observation, not a contribution, unless a wider literature search confirms otherwise.
- Our K2(b) specialises the nested-minimisation idea to a near-circular target orbit and certifies it with the bracket. This is a convenience, not a new MOID algorithm.

---

## K3. Soft size kernel

**Formulas.**

$$D(H,p_V)=\frac{1329\ \mathrm{km}}{\sqrt{p_V}}\,10^{-H/5}\qquad\text{(Fowler \& Chillemi 1992; Pravec \& Harris 2007)}.$$

- **The label's own size test** is exactly $S_H=\mathbf 1[H\le H_0]$, with $H_0=22.0$ on the catalogue $H$. This is the only size term that can enter a reproduction of the label.
- **Equivalent diameter threshold** at albedo $p_V$: $D_0(p_V)=1329\ \mathrm{km}\cdot10^{-22/5}/\sqrt{p_V}$. For example, $D_0(0.14)=0.141$ km, which is the conventional "about 140 m" statement.
- **Physical-size probability** under an albedo prior. If $\ln p_V\sim\mathcal N(\mu,s^2)$ then $\ln D=\ln1329-\tfrac{H}{5}\ln10-\tfrac12\ln p_V$ is Gaussian, so

$$P(D\ge D^*\mid H)=\Phi\!\left(\frac{\ln(1329\,\mathrm{km}/D^*)-\tfrac{H}{5}\ln10-\mu/2}{s/2}\right).$$

  For a mixture prior $\sum_k w_k\mathcal N(\mu_k,s_k^2)$ on $\ln p_V$, this becomes $\sum_k w_k\Phi(z_k)$.
- **Measured values.** Where `albedo` is measured, $D$ follows directly. Where `diameter` is measured, use it.

**Parameters.**
- Prior parameters $\mu$, $s$ (or $w_k,\mu_k,s_k$) are fitted on the **training split's** measured albedos only.
- That subset is not a random sample of the catalogue: it reflects which objects thermal-infrared and other surveys measured. The prior inherits that selection, and the paper must say so.
- $D^*$ (km) is a physical choice, for example 0.14 km. It is not fitted.

**Pseudocode.**
```python
def diameter_km(H, pV): return 1329.0/np.sqrt(pV) * 10**(-H/5)
def size_label(H):      return H <= 22.0                              # the label's test, exactly
def p_large(H, mu, s, Dstar_km=0.14):                                  # ln pV ~ N(mu, s^2), fitted on train
    z = (np.log(1329.0/Dstar_km) - H*np.log(10)/5 - mu/2) / (s/2)
    return scipy.special.ndtr(z)
mu, s = np.log(train.albedo[train.albedo > 0]).mean(), np.log(train.albedo[train.albedo > 0]).std(ddof=1)
```

**Edge cases.**
- Albedo missing or zero: use the prior.
- Diameter given without albedo: use $D$.
- $H$ missing: the input is invalid and goes to a human.
- $H=22.0$ exactly: counts as $\le$, so the object is eligible.
- Angles do not enter K3.

**Cost.** $O(1)$.

**Meaning, and what cannot work.** $S_H$ *is* the label's size condition. $P(D\ge D^*)$ is a different, physical quantity. It can disagree with $S_H$ near $H=22$, and every disagreement is a disagreement with the label. So K3's soft score cannot improve label reproduction; it is for physical interpretation only.

**New vs standard.** Nothing new.

---

## K4. Uncertainty kernel

**Inputs.**
- Element vector $\mathbf x=(a,e,i,\Omega,\omega)$.
- $\Sigma=\mathrm{diag}(\sigma_a^2,\sigma_e^2,\sigma_i^2,\sigma_\Omega^2,\sigma_\omega^2)$, with the angular sigmas converted from degrees to radians.
- $\sigma_M$ is unused because the MOID does not depend on $M$.

**Target.**

$$P(\text{hazard})=\mathbf 1[H\le22]\cdot P(\mathrm{MOID}\le d_0).$$

The catalogue $H$ is part of the label's definition. If a photometric $\sigma_H$ were available, $\mathbf 1[H\le22]$ would become $\Phi((22-H)/\sigma_H)$, under the additional assumption that photometric and astrometric errors are independent. $\sigma_H$ is not among the listed inputs.

### K4(a). Delta method (closed form)

**On the nodal distances** (the closed form of K2(a)).

| gradient | $\partial/\partial a$ | $\partial/\partial e$ | $\partial/\partial i$ | $\partial/\partial\Omega$ | $\partial/\partial\omega$ |
|---|---|---|---|---|---|
| $\nabla\delta_+$ | $\dfrac{1-e^2}{1+e\cos\omega}$ | $-\dfrac{a\,[2e+(1+e^2)\cos\omega]}{(1+e\cos\omega)^2}$ | $0$ | $-R_\oplus'(\Omega)$ | $\dfrac{a(1-e^2)e\sin\omega}{(1+e\cos\omega)^2}$ |
| $\nabla\delta_-$ | $\dfrac{1-e^2}{1-e\cos\omega}$ | $\dfrac{a\,[(1+e^2)\cos\omega-2e]}{(1-e\cos\omega)^2}$ | $0$ | $-R_\oplus'(\Omega+\pi)$ | $-\dfrac{a(1-e^2)e\sin\omega}{(1-e\cos\omega)^2}$ |

With $C=\begin{pmatrix}\nabla\delta_+^\top\Sigma\nabla\delta_+ & \nabla\delta_+^\top\Sigma\nabla\delta_-\\ \cdot & \nabla\delta_-^\top\Sigma\nabla\delta_-\end{pmatrix}$ and $(\delta_+,\delta_-)\sim\mathcal N(\hat{\boldsymbol\delta},C)$:

$$P_{\rm node}=P_++P_--P\bigl(|\delta_+|\le d_0,\ |\delta_-|\le d_0\bigr),\qquad P_\pm=\Phi\!\Bigl(\tfrac{d_0-\hat\delta_\pm}{s_\pm}\Bigr)-\Phi\!\Bigl(\tfrac{-d_0-\hat\delta_\pm}{s_\pm}\Bigr).$$

The last term is a bivariate-normal rectangle probability. Since $U_{\rm node}\ge\mathrm{MOID}$, $P_{\rm node}\le P(\mathrm{MOID}\le d_0)$: a **lower** bound on the hazard probability. Folding through $|\delta|$ keeps it correct when a nodal distance changes sign.

**On the radial gap.** Take $g=q-Q_\oplus$, with $\nabla g=(1-e,\,-a,0,0,0)$, or $g=q_\oplus-Q$, with $\nabla g=(-(1+e),\,-a,0,0,0)$. Then $P_{\rm rad}=\Phi((d_0-g)/s_g)$ is an **upper** bound on the hazard probability.

**On the exact signed MOID.** By the envelope theorem, the derivative of a minimum is the partial derivative at the minimiser, with Earth's curve fixed:

$$\frac{\partial\tilde d}{\partial x_k}=\hat{\mathbf n}\cdot\frac{\partial\mathbf r}{\partial x_k}\Big|_{f^*},\qquad
\frac{\partial\mathbf r}{\partial a}=\frac{\mathbf r}{a},\quad
\frac{\partial\mathbf r}{\partial e}=-\frac{a\,[2e+(1+e^2)\cos f]}{(1+e\cos f)^2}\,\hat{\mathbf r},\quad
\frac{\partial\mathbf r}{\partial i}=\hat{\mathbf n}_\Omega\times\mathbf r,\quad
\frac{\partial\mathbf r}{\partial \Omega}=\hat{\mathbf z}\times\mathbf r,\quad
\frac{\partial\mathbf r}{\partial \omega}=\hat{\mathbf h}\times\mathbf r .$$

Here $\hat{\mathbf n}_\Omega=(\cos\Omega,\sin\Omega,0)$ is the node direction. Then $s^2=\nabla\tilde d^\top\Sigma\,\nabla\tilde d$ and

$$P_{\rm lin}=\Phi\!\Bigl(\frac{d_0-\tilde d}{s}\Bigr)-\Phi\!\Bigl(\frac{-d_0-\tilde d}{s}\Bigr).$$

Because $\hat{\mathbf n}$ is defined by the tangents, the gradient exists even at a crossing ($\tilde d=0$). This is exactly why the signed MOID is used (Gronchi & Tommei 2007).

**Validity.** The linearisation is trusted only if all three hold; otherwise K4(b) runs. The defaults are policy choices, not fitted:
- the runner-up local minimum lies at least $k_{\rm gap}\,s$ above the MOID (default $k_{\rm gap}=5$);
- the tangent angle is at least $\theta_{\min}$;
- $s\le s_{\rm lin}$ (default $s_{\rm lin}=0.01$ au, chosen from the synthetic checks in Appendix A).

### K4(b). Sampling and the unscented transform

**Monte Carlo.**
- Draw $\mathbf x^{(j)}\sim\mathcal N(\hat{\mathbf x},\Sigma)$, $j=1..N_{\rm MC}$ (default 4000).
- Fold draws with $e<0$ by $(e,\omega)\to(-e,\omega+\pi)$, and draws with $i\notin[0,\pi]$ by $(i,\Omega,\omega)\to(2\pi-i\ \text{or}\ -i,\ \Omega+\pi,\ \omega+\pi)$. These describe the same geometric orbit.
- Reject draws with $a\le0$ or $e\ge1$ for a bound nominal orbit, and **report the rejected count**.
- Compute $\mathrm{MOID}^{(j)}$ by K2(b), vectorised over draws.
- $\hat P=k/N_{\rm MC}$, with a 95% Wilson interval $[P_{\rm lo},P_{\rm hi}]$. A "sure" call requires $P_{\rm lo}\ge1-\alpha$ or $P_{\rm hi}\le\alpha$.

**Unscented transform** (Julier & Uhlmann 2004).
- $2n+1=11$ sigma points $\hat{\mathbf x}$ and $\hat{\mathbf x}\pm\sqrt{(n+\kappa)}\,\sigma_k\mathbf e_k$, with $\kappa=3-n$.
- Weights $W_0=\kappa/(n+\kappa)$ and $W_k=1/(2(n+\kappa))$.
- Signed MOID at each point; weighted mean and variance; then the folded normal above.
- The transform is flagged invalid if a sigma point lands in a different basin of K2(b), because the map is then not smooth.

**Pseudocode.**
```python
sig = np.array([row['sigma_a'], row['sigma_e'], np.deg2rad(row['sigma_i']),
                np.deg2rad(row['sigma_om']), np.deg2rad(row['sigma_w'])])            # sigma_ma unused
m = moid_certified(el)                                # MOID, signed MOID d~, f*, n_hat, runner-up
grad = moid_grad(el, m)                               # the five envelope-theorem terms above
s = np.sqrt(np.sum((grad*sig)**2))                    # independent sigmas (no covariance given)
P_lin = Phi((D0 - m['signed'])/s) - Phi((-D0 - m['signed'])/s)
lin_ok = m['signed_ok'] and m['second'] - m['moid'] >= 5*s and s <= 0.01
if not lin_ok:                                        # K4(b)
    draws = fold_and_reject(rng.normal(x_hat, sig, (4000, 5)))
    P_mc = np.mean(moid_batch(draws) <= D0)          # with its Wilson interval
P_hazard = (row['H'] <= 22.0) * (P_lin if lin_ok else P_mc)
```

**Edge cases.**
- $e\to0$: $\partial\tilde d/\partial\omega=O(e)$, because $\hat{\mathbf h}\times\mathbf r$ becomes tangent to the orbit. A huge $\sigma_\omega$ then multiplies a vanishing derivative; the delta method handles this correctly if $e\,\sigma_\omega$ is meaningful.
- $i\to0$: $\partial/\partial\Omega$ and $\partial/\partial\omega$ both tend to $\partial/\partial\varpi$. With independent sigmas the variance is overstated (§0, item 4). The object is then usually "unresolved", which is safe but uninformative.
- $e\to1$: draws with $e\ge1$ are rejected and counted.
- Angles: every sigma is converted to radians, and the folding keeps the same geometric orbit.

**Assumptions** (state them in the paper):
- osculating elements at one epoch, with Gaussian errors;
- independent sigmas, because no covariance is given;
- linearity, for (a) only;
- Earth's curve exact and fixed.

**Where it fails:**
- short arcs, which are non-Gaussian (Milani et al. 2005);
- low $i$ or low $e$ without covariance;
- nearly parallel tangents at a crossing;
- several near-equal local minima;
- large $s$.

These are exactly the conditions that route an object to sampling or to "unresolved".

**Cost.**
- (a): $O(1)$ after K2(b): five cross or dot products. The nodal version needs one bivariate-normal rectangle.
- (b): $N_{\rm MC}\times$ K2(b), vectorised.
- Unscented transform: $11\times$ K2(b).

**New vs standard.**
- Linearised MOID uncertainty and the signed MOID are Gronchi & Tommei (2007). Sampling and the unscented transform are standard.
- Reporting the PHA *label* itself as a probability under the catalogue uncertainty may be new as a classification target. This should be checked against the literature before it is claimed.

---

## K5. Combination kernel and decision law

### K5(a). The hazard score

$$S=\mathbf 1[H\le H_0]\cdot\Bigl[\Phi\Bigl(\frac{d_0+\beta-\tilde d}{s_{\rm eff}}\Bigr)-\Phi\Bigl(\frac{-d_0-\beta-\tilde d}{s_{\rm eff}}\Bigr)\Bigr],\qquad s_{\rm eff}=\sqrt{s^2+w^2},\qquad S\in[0,1].$$

| parameter | physical meaning | unit | how it is set |
|---|---|---|---|
| $d_0=0.05$, $H_0=22.0$ | the PHA definition | au, mag | fixed, never fitted |
| $a_\oplus,e_\oplus,\varpi_\oplus$ | Earth's fixed curve | au, –, rad | fixed (Standish) |
| $\beta$ | systematic offset between our fixed-curve MOID and the label's MOID (Earth model, epoch) | au | maximum likelihood on the **training** split |
| $w$ | residual scatter between the two MOIDs not explained by orbit uncertainty | au | maximum likelihood on the **training** split |
| $\alpha$ | tail beyond which a call is "sure" (default $\Phi(-3)$) | – | policy; chosen before any test label is seen |
| $s_{\rm lin},k_{\rm gap},\theta_{\min},\nu_{\max}$ | validity limits of K4(a) and K2(a) | au, –, rad, rad | policy; sensitivity reported |
| $N,\ \texttt{keep},\ N_{\rm MC}$ | grid size, basins refined, draws | – | numerical; increase until results stop changing |
| $\varepsilon,\tau$ | smoothing of the K2(a) surrogate (only if gradients are needed) | au | numerical |

**Fitting $\beta$ and $w$, on training data only.**
- Split **by object**, not by close-approach record, stratified on the label.
- Maximise $\sum_j\bigl[y_j\ln S_j+(1-y_j)\ln(1-S_j)\bigr]$ over $(\beta,\ln w)$ on the training split, where $y$ is the `pha` flag used **only as the training target**. Faint objects ($H>22$) carry no information about $\beta$ and $w$ and drop out.
- Never use the `moid` column, not even to fit.
- Report $\hat\beta$ and $\hat w$ with intervals. Evaluate once on the held-out split.
- A large $\hat w$ relative to the numerical tolerance $\eta$ means the Earth curve, not the classifier, needs attention.
- With $\beta=w=0$ and $s=0$, $S$ is exactly the label's definition evaluated with our MOID. Fitting can only absorb the Earth-curve mismatch; it cannot add physics.

**Earth's time-dependent position cancels.**
- **Proposition.** $S$ is invariant under $M\to M+c$ and under any shift of Earth's position along its orbit.
- **Proof.** $S$ depends on the inputs only through $\tilde d$, $s$ and $H$.
  - $\tilde d$ is a minimum over *both* anomalies $(f,\psi)$, so it is a function of the two curves alone.
  - $s$ is evaluated at the minimising anomalies, which the curves determine.
  - Neither $M$, the epoch, nor Earth's mean longitude appears. $\square$
- A close-approach distance depends on both phases. The MOID is its infimum over both phases, which is what makes the asteroid's point of view phase-free.

### K5(b). The decision law (classification with explicit abstention)

The kernels produce certificates and probabilities. A fixed law turns them into an act. The law reads only a 9-bit *situation word*: never an orbit, a magnitude or a time.

| bits | field | values |
|---|---|---|
| 0 | INV | 1 = input unusable (non-finite, $e<0$, $q\le0$, $i\notin[0,180^\circ]$, $H$ missing) |
| 1 | FAINT | 1 = $H>22.0$ |
| 2–4 | CERT | 0 OPEN ($L_0\le d_0<U_{\rm node}$, no exact MOID yet); 1 FAR ($L_0>d_0$); 2 NEAR ($U_{\rm node}\le d_0$); 3 EXACT_FAR; 4 EXACT_NEAR (certified K2(b)); 5–7 BAD |
| 5–6 | UNC | 0 none; 1 K4(a) valid; 2 linearisation invalid; 3 sampled |
| 7–8 | PB | 0 SURE_IN ($P\ge1-\alpha$); 1 MIDDLE; 2 SURE_OUT ($P\le\alpha$); 3 BAD |

**Acts, in priority order.** The lowest-numbered rung that holds wins, and the act is the rung number.

| act | rung condition | meaning |
|---|---|---|
| 0 HUMAN | outside the 9-bit domain, or INV, or CERT ≥ 5, or PB = 3 | invalid input or failed certificate |
| 1 NOT_PHA_SIZE | FAINT | not a PHA whatever the orbit; no orbit work is done |
| 2 EXACT | CERT = OPEN | the cheap screen is open: run K2(b) |
| 3 LINEAR | UNC = 0 | run K4(a) on the certificate held |
| 4 SAMPLE | UNC = 2 | run K4(b) |
| 5 PHA | CERT ∈ {NEAR, EXACT_NEAR} and PB = SURE_IN | PHA, robust at level $\alpha$ |
| 6 NOT_PHA_GEOMETRY | CERT ∈ {FAR, EXACT_FAR} and PB = SURE_OUT | not a PHA, robust at level $\alpha$ |
| 7 EXACT_FOR_P | CERT ∈ {FAR, NEAR} | a cheap bound whose probability is not decisive: run K2(b), recompute $P$ |
| 8 OBSERVE_PHA | CERT = EXACT_NEAR | nominal PHA, but the label could flip within the uncertainty: follow up |
| 9 OBSERVE_NOT | (floor) | nominal non-PHA that could flip: follow up |

**Terms used.**
- "PHA" and "not a PHA" are the label's two classes, made robust.
- OBSERVE_* is the explicit *unresolved* class. Its first word repeats the nominal label.
- HUMAN is the abstention for unusable inputs.
- Every non-terminal act changes the word:
  - EXACT sets CERT and resets the probability;
  - LINEAR sets UNC to 1 or 2;
  - SAMPLE sets UNC to 3.

  The procedure therefore terminates in at most six steps.

**The law as executed.** It is machine-authored by integer program synthesis. Each lane is a short branchless integer expression found by search from its truth table; the glue is masks and ORs; and the act is the lowest set bit.

```c
#define LANE static inline __attribute__((always_inline)) int32_t
LANE I_HI     (int32_t x){ return ((x >> 31) | ((-x) >> 31)); }
LANE INV_EQ1  (int32_t x){ return ((-x) >> 31); }
LANE CERT_GE5 (int32_t x){ return ((-(x - 16)) >> 31); }
LANE PB_EQ3   (int32_t x){ return (((-384) >> 31) ^ ((-(x ^ 384)) >> 31)); }
LANE FAINT_EQ1(int32_t x){ return ((-x) >> 31); }
LANE CERT_EQ0 (int32_t x){ return ((x - 1) >> 31); }
LANE UNC_EQ0  (int32_t x){ return ((x - 1) >> 31); }
LANE UNC_EQ2  (int32_t x){ return (((-64) >> 31) ^ ((-(x ^ 64)) >> 31)); }
LANE CERT_IN24(int32_t x){ return (((-(x & 12)) >> 31) ^ ((-(x & (x ^ 8))) >> 31)); }
LANE PB_EQ0   (int32_t x){ return ((x - 1) >> 31); }
LANE CERT_IN13(int32_t x){ return (((-(x ^ 4)) >> 31) ^ ((-(x ^ 12)) >> 31)); }
LANE PB_EQ2   (int32_t x){ return (((-256) >> 31) ^ ((-(x ^ 256)) >> 31)); }
LANE CERT_IN12(int32_t x){ return (((-(x ^ 4)) >> 31) ^ ((-(x ^ 8)) >> 31)); }
LANE CERT_EQ4 (int32_t x){ return (((-16) >> 31) ^ ((-(x ^ 16)) >> 31)); }
LANE AND2     (int32_t x){ return ((-(x - 2)) >> 31); }

int32_t pha_law(int32_t n)          /* build: cc -O2 -fwrapv (int32 wraps at every step) */
{   int32_t inv = n & 0x1, faint = n & 0x2, cert = n & 0x1C, unc = n & 0x60, pb = n & 0x180, hi = n & ~0x1FF;
    int32_t m = ((I_HI(hi) | INV_EQ1(inv) | CERT_GE5(cert) | PB_EQ3(pb)) & 1)
              | ((FAINT_EQ1(faint)) & 2)
              | ((CERT_EQ0(cert)) & 4)
              | ((UNC_EQ0(unc)) & 8)
              | ((UNC_EQ2(unc)) & 16)
              | ((AND2((CERT_IN24(cert) & 1) | (PB_EQ0(pb) & 2))) & 32)
              | ((AND2((CERT_IN13(cert) & 1) | (PB_EQ2(pb) & 2))) & 64)
              | ((CERT_IN12(cert)) & 128)
              | ((CERT_EQ4(cert)) & 256)
              | 512;
    return __builtin_ctz((uint32_t)m);
}
```

**Properties.**
- **Exhaustive equality.** It equals the written specification (the table above, transcribed independently as plain `if`/`else`) on **all $2^{32}$ int32 inputs**, with 0 mismatches.
- **Compiled form.** Under Apple clang 21 at `-O2 -fwrapv` on arm64, it compiles to 60 instructions with no branches.
- **Speed.** It takes about 2.2 ns per decision on an Apple M4 Pro.
- **What it cannot change.** The law only routes. It cannot change a label: a wrong certificate becomes BAD and goes to a human, never to a wrong class.

---

## Summary table

| kernel | inputs (SBDB fields) | outputs | free parameters |
|---|---|---|---|
| K1 path | `q`, `e`, `i`, `om`, `w` | $\mathbf r(f)$; $(\rho,\lambda,z)$; tangents | none |
| K2(a) closed forms | `q`, `e`, `i`, `om`, `w`; Earth constants | $L_0$ (lower bound), $U_{\rm node}$ (upper bound), $\delta_\pm$, $d_{\rm lin}$, $D_{\varepsilon,\tau}$ | $\nu_{\max}$ (policy); $\varepsilon,\tau$ (numerical) |
| K2(a′) bracket | as K2(a) | $L_{\rm lon}\le\mathrm{MOID}\le U_{\rm lon}$ | $N$ (numerical) |
| K2(b) exact | as K2(a) | MOID, signed MOID $\tilde d$, $f^*$, $\psi^*$, runner-up minimum, certificate | $N$, `keep`, $\eta$ (numerical) |
| K3 size | `H`; `albedo`, `diameter` where present | $\mathbf 1[H\le22]$ (label); $D$; $P(D\ge D^*)$ | $\mu,s$ or a mixture (fitted on train albedos); $D^*$ (physical choice) |
| K4(a) delta | K2 outputs; `sigma_a`, `sigma_e`, `sigma_i`, `sigma_om`, `sigma_w` | $s$; $P_{\rm node}$ (lower bound), $P_{\rm rad}$ (upper bound), $P_{\rm lin}$; validity flag | $s_{\rm lin},k_{\rm gap},\theta_{\min}$ (policy) |
| K4(b) sampling | as K4(a) | $\hat P$ with a Wilson interval and the rejected count; unscented $P$ with a validity flag | $N_{\rm MC}$, seed (numerical) |
| K5(a) score | $\tilde d$, $s$, `H` | $S\in[0,1]$ | $\beta,w$ (fitted on train); $d_0,H_0$ (fixed) |
| K5(b) decision law | the 9-bit situation word | one of ten acts | $\alpha$ (policy); the law itself has none |

Never used: `ma`, `epoch`, `sigma_ma`, `moid`, `pha` (except as the training target), close-approach data, Earth's position at a time. Diagnostics only, never in a score: `condition_code`, `data_arc`, `n_obs_used`.

---

## Appendix A. Implementation checks on synthetic orbits (not results)

These numbers describe the reference code on **synthetic** orbits, generated by `tests/`, against an independent brute-force MOID. That reference uses both orbits in eccentric anomaly, explicit rotation matrices, a full 1500 × 1500 grid and a 2-D BFGS polish. They say nothing about real asteroids and will change with grids and tolerances. Reproduce them with `sh run_all.sh`.

| check | synthetic set | outcome |
|---|---|---|
| K1: nodes on the reference plane; $\lvert\mathbf r\rvert$ law; basis against rotation matrices | 210 orbits | nodes: ≤ $1.1\times10^{-15}$ au; $\lvert\mathbf r\rvert$ law: $7.1\times10^{-15}$ au; orthonormality: $4.4\times10^{-16}$; matrices: $2.2\times10^{-16}$ |
| K2(b) against the independent brute force | 300 random + 10 edge cases (coplanar, $i=10^{-7\circ}$, $e=0$, $e=10^{-9}$, $e=0.9755$, Atira-like, distant Amor, retrograde, Earth's own orbit, angle wrap) | max $\lvert\Delta\rvert=2.6\times10^{-11}$ au, median $2.2\times10^{-16}$ au |
| bound order $L_0,L_{\rm lon}\le\mathrm{MOID}\le U_{\rm lon}\le U_{\rm node}$; certificate | same 310 | 0 violations, 0 uncertified |
| bracket width $U_{\rm lon}-L_{\rm lon}$ where MOID < 0.2 au | 110 of the 310 | ≤ $9.4\times10^{-5}$ au |
| vectorised MOID (sampling) against K2(b) | 200 | ≤ $1.2\times10^{-13}$ au |
| closed-form gradients against central differences | 60 | nodal ≤ $7.2\times10^{-9}$; signed MOID (envelope theorem) ≤ $1.5\times10^{-7}$ |
| K4(a) against Monte Carlo (4,000 draws) | 11 orbits, sigmas scaled so $P$ lies between 0.01 and 0.99 | 11/11 inside the 95% interval ± 0.01; max $\lvert P_{\rm lin}-P_{\rm MC}\rvert=0.0145$; unscented max $0.0038$ |
| bivariate nodal $P$ against Monte Carlo of $U_{\rm node}$ | 12 | max difference $0.0011$ |
| K5 fitter recovers known $(\beta,w)$ | 20,000 synthetic labels | true (0.00070, 0.00150), fitted (0.00059, 0.00138) |
| decision procedure end to end | 13 synthetic rows | all 10 acts reached; every run terminates; no state repeats |
| decision law against specification | all $2^{32}$ int32 words | 0 mismatches |

Reference timings on synthetic orbits (Apple M4 Pro, Python/numpy reference code; hardware- and implementation-specific):
- closed forms: 0.1 µs per orbit, vectorised;
- K2(b): 0.35 ms per orbit, or 0.6 ms with its certificate;
- vectorised MOID for sampling: 80 µs per draw;
- the decision law in C: about 2.2 ns per decision.

## References

(Bibliographic details are being verified against primary sources; see the note at the end of this list.)

- Armellin, R., Di Lizia, P., Berz, M., Makino, K. (2010). Computing the critical points of the distance function between two Keplerian orbits via rigorous global optimization. *Celest. Mech. Dyn. Astron.* 107, 377–395.
- Baluev, R. V., Mikryukov, D. V. (2019). Fast error-controlling MOID computation for confocal elliptic orbits. *Astronomy and Computing* 27, 11–22.
- Bonanno, C. (2000). An analytical approximation for the MOID and its consequences. *Astron. Astrophys.* 360, 411–416.
- Bowell, E., Muinonen, K. (1994). Earth-crossing asteroids and comets: groundbased search strategies. In *Hazards due to Comets and Asteroids* (T. Gehrels, ed.), Univ. Arizona Press, 149–197.
- Fowler, J. W., Chillemi, J. R. (1992). IRAS asteroid data processing. In *The IRAS Minor Planet Survey* (E. F. Tedesco, ed.), Phillips Laboratory Tech. Rep. PL-TR-92-2049, 17–43.
- Gronchi, G. F. (2002). On the stationary points of the squared distance between two ellipses with a common focus. *SIAM J. Sci. Comput.* 24(1), 61–80.
- Gronchi, G. F. (2005). An algebraic method to compute the critical points of the distance function between two Keplerian orbits. *Celest. Mech. Dyn. Astron.* 93, 295–329.
- Gronchi, G. F., Tommei, G. (2007). On the uncertainty of the minimal distance between two confocal Keplerian orbits. *Discrete Contin. Dyn. Syst. Ser. B* 7(4), 755–778.
- Hedo, J. M., Ruíz, M., Peláez, J. (2018). On the minimum orbital intersection distance computation: a new effective method. *Mon. Not. R. Astron. Soc.* 479, 3288–3299.
- Hedo, J. M., Fantino, E., Peláez, J., Urrutxua, H. (2020). Minimum orbital intersection distance: an asymptotic approach. *Astron. Astrophys.* 633, A22.
- Julier, S. J., Uhlmann, J. K. (2004). Unscented filtering and nonlinear estimation. *Proc. IEEE* 92(3), 401–422.
- Kholshevnikov, K. V., Vassiliev, N. N. (1999). On the distance function between two Keplerian elliptic orbits. *Celest. Mech. Dyn. Astron.* 75, 75–83.
- Mainzer, A., et al. (2011). NEOWISE observations of near-Earth objects: preliminary results. *Astrophys. J.* 743, 156.
- Milani, A., Sansaturio, M. E., Tommei, G., Arratia, O., Chesley, S. R. (2005). Multiple solutions for asteroid orbits: computational procedure and applications. *Astron. Astrophys.* 431, 729–746.
- Murray, C. D., Dermott, S. F. (1999). *Solar System Dynamics*. Cambridge University Press.
- Pravec, P., Harris, A. W. (2007). Binary asteroid population. 1. Angular momentum content. *Icarus* 190, 250–259.
- Sitarski, G. (1968). Approaches of the parabolic comets to the outer planets. *Acta Astronomica* 18, 171–195.
- Standish, E. M. Keplerian elements for approximate positions of the major planets. JPL Solar System Dynamics, https://ssd.jpl.nasa.gov/planets/approx_pos.html (Table 1, 1800–2050 AD).
- Wiśniowski, T., Rickman, H. (2013). A fast, geometric method for calculating accurate minimum orbit intersection distances (MOIDs). *Acta Astronomica* 63, 293–307.
- Wright, E. L., et al. (2016). The albedo distribution of near Earth asteroids. *Astron. J.* 152, 79.
- CNEOS, NASA/JPL. NEO groups and the definition of a Potentially Hazardous Asteroid. https://cneos.jpl.nasa.gov/about/neo_groups.html

*Verification note: entries are being checked against primary sources; corrections, if any, are applied in this list.*
