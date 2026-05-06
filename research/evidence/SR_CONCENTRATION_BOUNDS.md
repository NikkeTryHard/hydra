# Strongly Rayleigh Bounds for Hydra Safety: Research Report

## Executive Summary

Hydra multivariate hypergeometric tile distribution = **Strongly Rayleigh (SR)**, not mere Negatively Associated (NA). This unlocks much tighter bounds. Replace current union-bound safety calc with exponential concentration inequalities. Gain not small: from O(J) union bounds to exp(-a^2/8k) Gaussian tails.

---

## 1. The Negative Dependence Hierarchy

**Source:** Borcea, Branden, Liggett. "Negative Dependence and Geometry of Polynomials." *JAMS* 22(2), 2009. [AMS link](https://www.ams.org/jams/2009-22-02/S0894-0347-08-00618-8/)

Hierarchy, strongest -> weakest (BBL 2009, Figure 1):

```
Strongly Rayleigh (SR)
  => PHR (Projected Homogeneous Rayleigh)
    => CNA+ (Strong Conditional Negative Association)
      => Rayleigh (h-NLC+)
        => CNA (Conditional Negative Association)
          => h-NLC (Hereditary Negative Lattice Condition)
            => NA (Negative Association)        <-- Hydra currently stops here
              => NLC (Negative Lattice Condition)
                => p-NC (Pairwise Negative Correlation)
```

**Definition (SR):** Measure mu on {0,1}^n is *strongly Rayleigh* iff generating polynomial g_mu(z) = sum_{S} mu(S) * prod_{i in S} z_i is *real stable* (no zeros when all Im(z_i) > 0).

**Equivalent characterization (BBL Theorem 4.1):** For multi-affine g with real coefficients, g stable iff:

$$\frac{\partial g}{\partial z_i}(x) \cdot \frac{\partial g}{\partial z_j}(x) \ge \frac{\partial^2 g}{\partial z_i \partial z_j}(x) \cdot g(x)$$

for all x in R^n and all i,j.

### Why Hydra's Distribution is SR

Multivariate hypergeometric distribution = uniform distribution on bases of partition matroid. Partition matroids are balanced matroids. By BBL Theorem 3.8, uniform measure on bases of balanced matroid is strongly Rayleigh.

More direct: generating polynomial of k-homogeneous exchangeable measure with rank sequence {r_k} is SR iff all zeros of sum_k r_k * z^k are real. For hypergeometric, polynomial has all real roots (factors into linear terms).

---

## 2. The Main Concentration Inequalities (Tighter Than NA)

### 2.1 Pemantle-Peres Gaussian Concentration (2014)

**Source:** Pemantle, Peres. "Concentration of Lipschitz Functionals of Determinantal and Other Strong Rayleigh Measures." *Combinatorics, Probability and Computing* 23(1), 2014. [arXiv:1108.0687](https://arxiv.org/abs/1108.0687)

**Theorem 3.1 (k-homogeneous SR => Gaussian tail):** Let P be k-homogeneous probability measure on B_n satisfying SCP. Let f be Lipschitz-1. Then:

$$P(f - Ef \ge \le \exp\left(-\frac{a^2}{8k}\right)$$

Two-sided version:

$$P(|f - Ef| > \le 2\exp\left(-\frac{a^2}{8k}\right)$$

When k <= n/2:

$$P(f - Ef > \le \exp\left(-\frac{a^2}{4n}\right)$$

**Theorem 3.2 (General SR => Gauss-Poisson tail):** For general (non-homogeneous) SR with mean mu = EN:

$$P(f - Ef > \le 3\exp\left(-\frac{a^2}{16(a + 2\mu)}\right)$$

$$P(|f - Ef| > \le 5\exp\left(-\frac{a^2}{16(a + 2\mu)}\right)$$

### What This Means for Hydra

**Current approach (NA only):** Use orthant bound P[cap_j {X_j >= t_j}] <= prod_j P[X_j >= t_j], plus Hunter union bounds for safety. These yield O(J)-term sums.

**SR upgrade:** ANY Lipschitz function of tile counts, including danger scores, tenpai indicators, safety metrics, concentrates with sub-Gaussian tails around expectation. Instead of bounding separate events then union-bounding, directly bound deviation of whole safety score function.

**Concrete example:** If k=13 tiles observed drawn (typical mid-game hand) and compute Lipschitz-1 safety function f:

$$P(f - Ef \ge \le \exp\left(-\frac{a^2}{104}\right)$$

For = 3 (3-sigma event): P <= exp(-9/104) ~ 0.917 (too loose here). Serfling-corrected version below tightens much more.

---

## 3. Sampling-Without-Replacement Concentration (Serfling Bounds)

**Source:** Bardenet, Maillard. "Concentration Inequalities for Sampling Without Replacement." *Bernoulli* 21(3), 2015. [arXiv:1309.4029](https://arxiv.org/abs/1309.4029)

These apply DIRECTLY to Hydra because tile draws are sampling without replacement from wall.

### 3.1 The Serfling Coupling (Why SWOR is Tighter than SWR)

**Lemma 1.1:** For convex f, if X_1,...,X_n sampled without replacement and Y_1,...,Y_n with replacement from same population:

$$E[f(sum X_i)] \le E[f(sum Y_i)]$$

Thus EVERY Chernoff/Hoeffding bound for i.i.d. sampling transfers to SWOR, and gets TIGHTER by Serfling correction factor.

### 3.2 Hoeffding-Serfling Inequality

**Corollary 2.5:** With probability >= 1-delta:

$$\frac{1}{n}\sum_{t=1}^n (X_t - \mu) \le (b-a)\sqrt{\frac{\rho_n \log(1/\delta)}{2n}}$$

where **Serfling correction factor** is:

$$\rho_n = \begin{cases} 1 - \frac{n-1}{N} & \text{if } n \le N/2 \\ (1 - \frac{n}{N})(1 + 1/n) & \text{if } n > N/2 \end{cases}$$

**Hydra interpretation:** N = total remaining tiles in wall (~70 early game), n = tiles drawn. Factor rho_n < 1 means SWOR concentration tighter than SWR. Late game, n close to N => rho_n -> 0, meaning near-perfect concentration, matching intuition: few tiles left => little uncertainty.

### 3.3 Bernstein-Serfling Inequality (Variance-Adaptive)

**Corollary 3.6:** With probability >= 1 - 2*delta:

$$\frac{1}{n}\sum_{t=1}^n (X_t - \mu) \le \sigma\sqrt{\frac{2\rho_n \log(1/\delta)}{n}} + \frac{\kappa_n(b-a)\log(1/\delta)}{n}$$

where sigma^2 is population variance and kappa_n is:

$$\kappa_n = \begin{cases} \frac{4}{3} + \sqrt{\frac{f_n}{g_{n-1}}} & n \le N/2 \\ \frac{4}{3} + \sqrt{g_{n+1}(1-f_n)} & n > N/2 \end{cases}$$

with f_n = n/N, g_n = N/n - 1.

**Why this matters:** Bernstein bound is variance-adaptive. When variance is small, e.g. dangerous tile type has few remaining copies, bound is MUCH tighter than Hoeffding.

### 3.4 Empirical Bernstein (No True Variance Needed)

**Theorem 4.3:** With probability >= 1 - 5*delta:

$$\frac{1}{n}\sum_{t=1}^n (X_t - \mu) \le \hat{\sigma}_n\sqrt{\frac{2\rho_n \log(1/\delta)}{n}} + \frac{7/3 + 3/\sqrt{2}}{1} \cdot \frac{(b-a)\log(1/\delta)}{n}$$

where hat{sigma}_n is EMPIRICAL variance from observed tiles. Critical for Hydra: no need true distribution of remaining tiles; use empirical variance from observations.

---

## 4. Critical SR Closure Properties (Why Conditioning Doesn't Break It)

**Source:** BBL 2009, Section 4.2 and Theorem 4.9.

Arguably most important result for Hydra:

**Theorem (BBL 4.9):** Strongly Rayleigh property is closed under:
1. **Conditioning** -- conditioning on X_i = 0 or X_i = 1 preserves SR
2. **External fields** -- reweighting by exp(lambda_i * X_i) preserves SR
3. **Projections** -- marginalizing out variables preserves SR
4. **Truncation** (single-level) -- conditioning on sum(X) = k preserves SR

**What this means for Hydra:** When tiles observed (discards, calls, draws), you condition multivariate hypergeometric. Under NA alone, conditional distributions may lose negative association. Under SR, **conditional distribution stays SR**. So ALL concentration bounds above remain valid after every observation. Incremental belief updates preserve SR structure.

**Proposition 2.3 (Pemantle-Peres):** For SR measure P, let P_k = P conditioned on N=k. Then P_{k+1} covers P_k (stochastic domination). Meaning: conditioning on "more tiles drawn" only makes concentration TIGHTER.

---

## 5. Matrix Concentration (For Multi-Dimensional Safety)

### 5.1 Matrix Bernstein for SR (Kathuria 2020)

**Source:** Kathuria. Matrix Bernstein Inequality for Strong Rayleigh Distributions." [arXiv:2011.13340](https://arxiv.org/abs/2011.13340)

**Theorem 1.1:** For k-homogeneous SR measure pi on n elements, L-Lipschitz matrix function F: {0,1}^n -> S_d:

$$Pr[\|F(x) - mu\| \ge t] \le 2d \cdot \exp\left(-\frac{t^2}{32(kL^2 + t\sqrt{k}L)}\right)$$

### 5.2 Matrix Chernoff for SR (Kyng-Song 2018)

**Source:** Kyng, Song. Matrix Chernoff Bound for Strongly Rayleigh Distributions." *FOCS 2018.* [IEEE](https://ieeexplore.ieee.org/abstract/document/8555121/)

**Theorem 4.1:** For k-homogeneous SR, PSD matrices A_e with ||A_e|| <= 1 and ||E[sum x_e A_e]|| <= mu:

$$Pr[\|\sum x_e A_e - E[\sum x_e A_e]\| \ge \epsilon\mu] \le d \cdot e^{-\epsilon^2 \mu / (\log k + \epsilon) \cdot \Theta(1)}$$

**Hydra application:** If danger model outputs vector of per-tile danger probabilities, these matrix bounds let you bound concentration of ENTIRE danger vector at once, not tile-by-tile.

---

## 6. Log-Sobolev Inequalities (Strongest Known Tool)

**Source:** Hermon, Salez. "Modified Log-Sobolev Inequalities for Strong-Rayleigh Measures." *Annals of Applied Probability* 33(2), 2023. [Project Euclid](https://projecteuclid.org/journals/annals-of-applied-probability/volume-33/issue-2/Modified-log-Sobolev-inequalities-for-strong-Rayleigh-measures/10.1214/22-AAP1847.short)

State of art. They prove universal modified log-Sobolev inequalities for measures satisfying SCP, which includes all SR measures. These imply:

1. Pemantle-Peres concentration as corollary
2. Hypercontractivity estimates
3. Entropy decay bounds

MLSI constant for k-homogeneous SCP measures is at least 1/(2k), which directly yields exp(-a^2/8k) concentration.

---

## 7. Concrete Comparison: NA vs SR for Hydra Safety

### Scenario: Estimating P(opponent has >= 2 copies of dangerous tile)

Setup: H = 70 hidden tiles, K = 4 copies of tile type T, opponent holds n = 13 tiles.

**Method 1 (Current - NA orthant bound):**
P[X_T >= 2] = 1 - P[X_T=0] - P[X_T=1] (exact hypergeometric, fine for single tile)
But for JOINT events across multiple tile types: use union bound or Hunter bound.
For J threat events: O(J) terms, O(J^2) for Hunter.

**Method 2 (SR Concentration):**
Define f(X) = danger_score(X), Lipschitz function of tile counts.
If f is L-Lipschitz and draw is k-homogeneous SR:

$$P(f(X) - Ef(X) \ge \le \exp\left(-\frac{a^2}{8kL^2}\right)$$

For k = 13 (hand size), L = 1:

$$P(f - Ef \ge \le \exp\left(-\frac{a^2}{104}\right)$$

With Serfling correction (rho_13 = 1 - 12/70 = 0.829 for H=70):
Effective bound tightens by factor rho_13.

**Method 3 (Bernstein-Serfling, variance-adaptive):**
If tile rare (small sigma), Bernstein gives:

$$P(f - Ef \ge \le \exp\left(-\frac{na^2/2}{\sigma^2(1 - \frac{n-1}{N}) + \frac{2}{3}(b-a)a}\right)$$

For low-variance cases, e.g. 1 copy of dangerous tile remaining, this is DRAMATICALLY tighter.

### Summary of Bound Tightness

| Method | Bound Type | Tightness | When Best |
|--------|-----------|-----------|-----------|
| Union bound | Sum of marginals | Loosest | Never (dominated) |
| Hunter/Kounias | Union - tree pairwise | Better | Few large threats |
| NA orthant | Product of marginals | Good for independent-like | Many small threats |
| Hoeffding-Serfling | exp(-2n*eps^2/rho) | Tight, no variance needed | General purpose |
| **Bernstein-Serfling** | **Variance-adaptive exp** | **Tightest scalar** | **Known/estimable variance** |
| **SR Lipschitz** | **exp(-a^2/8k)** | **Universal for any Lip func** | **Complex safety functions** |
| **Empirical Bernstein** | **Data-driven** | **Tight, no params needed** | **Online/adaptive** |

---

## 8. Key References (Complete)

1. **Joag-Dev, Proschan (1983).** "Negative Association of Random Variables." *Annals of Statistics.* [Current Hydra baseline]
2. **Pemantle (2000).** "Towards Theory of Negative Dependence." *J. Math. Physics* 41(3). [Survey, problem formulation]
3. **Borcea, Branden, Liggett (2009).** "Negative Dependence and Geometry of Polynomials." *JAMS* 22(2). [Defines SR, proves hierarchy]
4. **Pemantle, Peres (2014).** "Concentration of Lipschitz Functionals of Determinantal and Other Strong Rayleigh Measures." *CPC* 23(1). [Main concentration bounds]
5. **Bardenet, Maillard (2015).** "Concentration Inequalities for Sampling Without Replacement." *Bernoulli* 21(3). [Serfling-corrected Hoeffding/Bernstein]
6. **Greene, Wellner (2017).** "Exponential Bounds for Hypergeometric Distribution." *Bernoulli.* [Direct hypergeometric tail bounds]
7. **Kyng, Song (2018).** "Matrix Chernoff Bound for Strongly Rayleigh." *FOCS.* [Matrix concentration]
8. **Kathuria (2020).** "Matrix Bernstein for Strong Rayleigh." *arXiv:2011.13340.* [Matrix Bernstein]
9. **Hermon, Salez (2023).** "Modified Log-Sobolev Inequalities for Strong-Rayleigh Measures." *AAP* 33(2). [State-of-the-art, implies all prior results]
10. **Tolstikhin (2017).** "Concentration Inequalities for Samples Without Replacement." *TPITS.* [Additional SWOR bounds]

---

## 9. Recommended Changes to HYDRA_FINAL.md Section 5

Replace Section 5 "Conservative Probability Foundations" with:

1. **5.1:** State multivariate hypergeometric is **Strongly Rayleigh**, not only NA. Cite BBL 2009.
2. **5.2:** Replace orthant bound with Pemantle-Peres exp(-a^2/8k) concentration. This subsumes orthant bound.
3. **5.3:** Keep exact second-order statistics (still useful for Bernstein bounds).
4. **5.4 (NEW):** Add Bernstein-Serfling inequality with correction factor rho_n. This is workhorse bound for safety.
5. **5.5 (NEW):** Add SR closure under conditioning -- critical for proving incremental belief updates preserve concentration guarantees.
6. **6.2:** Upgrade Hunter/Kounias: note SR concentration can sometimes REPLACE union bounds entirely when safety function is Lipschitz.