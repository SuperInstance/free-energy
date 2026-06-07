# free-energy

> **Minimize surprise. Maximize survival. The Free Energy Principle, computed.**

[![crates.io](https://img.shields.io/crates/v/free-energy.svg)](https://crates.io/crates/free-energy)
[![docs.rs](https://docs.rs/free-energy/badge.svg)](https://docs.rs/free-energy)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A Rust library implementing the computational core of the Free Energy Principle (FEP). Computes variational free energy, generative models with recognition densities, prediction error dynamics, and homeostatic regulation — the mathematical foundations for agents that survive by minimizing surprise.

---

## Table of Contents

- [What is the Free Energy Principle?](#what-is-the-free-energy-principle)
- [Why Does This Matter?](#why-does-this-matter)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Mathematical Background](#mathematical-background)
- [Installation](#installation)
- [Related Crates](#related-crates)
- [License](#license)

---

## What is the Free Energy Principle?

The Free Energy Principle, proposed by Karl Friston (2006), states that biological agents must minimize the difference between their internal model of the world and what they actually observe. This difference is called **variational free energy** — a mathematically tractable upper bound on surprise (negative log-evidence).

The core equation:

```
F = complexity - accuracy
  = KL[q(z) || p(z)] - E_q[ln p(o|z)]
```

Where:
- **F** = variational free energy (what the agent minimizes)
- **q(z)** = the agent's posterior belief about hidden causes
- **p(z)** = the agent's prior beliefs
- **p(o|z)** = the generative model (likelihood of observations given causes)
- **KL** = Kullback-Leibler divergence (how far beliefs are from priors)

The agent never directly minimizes surprise — it minimizes free energy, which is an upper bound. This is tractable because the agent only needs its own beliefs and observations, not the true state of the world.

## Why Does This Matter?

**For AI**: This is a unified theory of perception, action, and learning. Instead of separate modules for each, a single free energy minimization drive produces all three behaviors. Perception = updating beliefs to reduce prediction error. Action = changing the world to match predictions. Learning = updating the model itself.

**For robotics**: Homeostatic systems (implemented here) model how agents maintain internal variables within survivable bounds. A robot that monitors battery, temperature, and load and autonomously corrects deviations is implementing free energy minimization.

**For cognitive science**: The FEP provides a mathematical framework for understanding how brains work. Predictive coding, Bayesian brain, and active inference all flow from this single principle.

**For agent design**: Variational optimization (`optimize_variational`) shows how agents can learn by iteratively adjusting their beliefs to minimize free energy — converging toward accurate world models.

## Architecture

```
free-energy
│
├── variational_free_energy    ← Core FEP computations
│   ├── free_energy()              F = complexity - accuracy
│   ├── complexity()               KL[q || p]
│   ├── accuracy()                 Expected log-likelihood
│   ├── surprise()                 -ln p(o)
│   ├── elbo()                     Evidence lower bound
│   └── optimize_variational()     Gradient-based belief update
│
├── GenerativeModel            ← Agent's world model
│   ├── set_likelihood()           p(observation | state)
│   ├── set_transition()           p(state_t+1 | state_t)
│   ├── predict_observation()      Forward model prediction
│   ├── log_evidence()             ln p(observation | model)
│   └── model_entropy()            Uncertainty in the model
│
├── RecognitionDensity         ← Agent's approximate posterior
│   ├── with_params()              Gaussian parameterization
│   ├── entropy()                  Uncertainty in beliefs
│   ├── update()                   Belief revision from prediction error
│   ├── kl_to()                    KL divergence to another density
│   └── sample_deterministic()     MAP estimate
│
├── prediction_error           ← Error-driven learning
│   ├── simple_error()             observed - predicted
│   ├── precision_weighted_pe()    Error × precision (attention)
│   ├── saliency()                 Norm of precision-weighted errors
│   └── rmse() / mae()             Aggregate error metrics
│
└── HomeostaticSystem          ← Survival maintenance
    ├── is_homeostatic()           Are all variables in bounds?
    ├── perturb() / correct()      Perturbation and correction
    ├── drive()                    Urgency to return to set-point
    ├── viability()                Distance from boundary death
    └── regulate_step()            Single homeostatic regulation cycle
```

## Quick Start

```rust
use free_energy::{
    variational_free_energy::{free_energy, complexity, surprise, optimize_variational},
    GenerativeModel,
    RecognitionDensity,
    HomeostaticSystem,
};

// Compute variational free energy
let complexity = 0.5;
let accuracy = 2.3;
let f = free_energy(complexity, accuracy);
println!("Free energy: {:.4}", f); // -1.8 (lower is better)

// How surprised is the agent by an observation with probability 0.1?
let s = surprise(0.1);
println!("Surprise: {:.4}", s); // 2.3026 nats

// Optimize beliefs: start with uniform, converge toward target
let initial = vec![0.25, 0.25, 0.25, 0.25];
let target  = vec![0.1, 0.7, 0.1, 0.1];
let posterior = optimize_variational(&initial, &target, 0.1, 100);
println!("Optimized posterior: {:?}", posterior);

// Build a generative model for a 3-state, 2-observation system
let mut model = GenerativeModel::new(3, 2);
model.set_prior(vec![0.5, 0.3, 0.2]);
model.set_likelihood(0, vec![0.8, 0.2]); // State 0 → likely obs 0
model.set_likelihood(1, vec![0.1, 0.9]); // State 1 → likely obs 1
let evidence = model.log_evidence(0);
println!("Log evidence for obs 0: {:.4}", evidence);

// Homeostatic system: maintain temperature at 37°C
let mut body = HomeostaticSystem::new(
    vec![37.0],                          // set-points
    vec![(35.0, 40.0)],                  // survivable bounds
    1.0,                                 // tolerance
);
body.perturb(&[2.0]);                    // Fever! +2°C deviation
println!("Homeostatic? {}", body.is_homeostatic()); // false
println!("Viability: {:.3}", body.viability());
body.correct(0.5);                       // Thermostat kicks in
```

## API Reference

### Variational Free Energy

| Function | Signature | Description |
|----------|-----------|-------------|
| `free_energy` | `(f64, f64) → f64` | F = complexity − accuracy |
| `complexity` | `(&[f64], &[f64]) → f64` | KL[q(z) ‖ p(z)] |
| `accuracy` | `(&[f64], &[f64]) → f64` | E_q[ln p(o\|z)] |
| `surprise` | `(f64) → f64` | −ln p(o) |
| `elbo` | `(f64, f64) → f64` | Evidence lower bound = accuracy − complexity |
| `expected_free_energy` | `(f64, f64) → f64` | G = risk + ambiguity |
| `optimize_variational` | `(&[f64], &[f64], f64, usize) → Vec<f64>` | Gradient descent on F |

### GenerativeModel

| Method | Returns | Description |
|--------|---------|-------------|
| `new(state_dim, obs_dim)` | `Self` | Create empty generative model |
| `set_likelihood(state, dist)` | `()` | Set p(observation \| state) |
| `set_transition(state, dist)` | `()` | Set p(next_state \| state) |
| `set_prior(prior)` | `()` | Set p(state) |
| `predict_observation(state)` | `&[f64]` | Forward prediction |
| `log_evidence(obs)` | `f64` | ln p(observation \| model) |
| `model_entropy()` | `f64` | H[model] |

### RecognitionDensity

| Method | Returns | Description |
|--------|---------|-------------|
| `new(dim)` | `Self` | Uniform Gaussian density |
| `with_params(means, precisions)` | `Self` | Parameterized density |
| `density(&x)` | `f64` | Evaluate q(x) |
| `entropy()` | `f64` | H[q] |
| `update(pe, lr)` | `()` | Update from prediction error |
| `kl_to(other)` | `f64` | KL[self ‖ other] |

### Prediction Error

| Function | Returns | Description |
|----------|---------|-------------|
| `simple_error(obs, pred)` | `Vec<f64>` | Element-wise residual |
| `squared_error(obs, pred)` | `f64` | ‖obs − pred‖² |
| `precision_weighted_pe(err, prec)` | `Vec<f64>` | Error × precision (attention) |
| `saliency(err)` | `f64` | ‖precision-weighted error‖ |

### HomeostaticSystem

| Method | Returns | Description |
|--------|---------|-------------|
| `new(set_points, bounds, tol)` | `Self` | Initialize homeostatic system |
| `is_homeostatic()` | `bool` | All variables within tolerance? |
| `deviation()` | `f64` | Total deviation from set-points |
| `perturb(delta)` | `()` | Apply external perturbation |
| `correct(strength)` | `()` | Drive toward set-points |
| `drive()` | `Vec<f64>` | Urgency vector per variable |
| `viability()` | `f64` | Distance from boundary (0=dead, 1=optimal) |
| `regulate_step(pert, corr)` | `()` | Full perturb-then-correct cycle |

## Mathematical Background

### Variational Free Energy

The agent maintains a generative model p(o, z \| m) and an approximate posterior q(z). Free energy is:

```
F(q) = E_q[ln q(z)] − E_q[ln p(o, z|m)]
     = KL[q(z) || p(z|o,m)] − ln p(o|m)
     ≥ −ln p(o|m)
```

Since F ≥ −ln p(o), minimizing F also minimizes an upper bound on surprise. The agent doesn't need to know the true posterior — only its approximation q and its generative model p.

### Predictive Coding

Prediction errors drive learning and perception:

```
ε = o − ō          (prediction error)
δ = ε × π           (precision-weighted)
Δq ∝ −∂F/∂q ∝ δ    (belief update)
```

Precision π acts as attention: high precision on a channel means the agent trusts that sensory input and updates beliefs rapidly.

### Homeostasis and Allostasis

Homeostatic variables x_i must stay within bounds [lo_i, hi_i]:

```
viability = Π_i  1 − |x_i − x*_i| / (bound_i / 2)
drive_i   = (x_i − x*_i) / bound_i
```

Allostasis (anticipatory regulation) adds a cost for predictive mismatch:

```
C_allostatic = ||drive − anticipation||²
```

### ELBO and Variational Inference

The Evidence Lower Bound (ELBO) is the negative of free energy:

```
ELBO = −F = accuracy − complexity = ln p(o|m) − KL[q || p(z|o,m)]
```

Maximizing ELBO is equivalent to minimizing F. `optimize_variational` performs gradient ascent on ELBO by iteratively adjusting the variational parameters.

## Installation

```bash
cargo add free-energy
```

Or add to your `Cargo.toml`:

```toml
[dependencies]
free-energy = "0.1"
```

## Related Crates

Part of the **SuperInstance Exocortex** ecosystem:

- **[markov-blanket](https://github.com/SuperInstance/markov-blanket)** — Statistical boundary between agent and world
- **[active-inference](https://github.com/SuperInstance/active-inference)** — Action as surprise minimization
- **[signal-transduction](https://github.com/SuperInstance/signal-transduction)** — Biological signal cascading for agents
- **[morphogenesis](https://github.com/SuperInstance/morphogenesis)** — Turing pattern formation for agent development
- **[dream-cycle](https://github.com/SuperInstance/dream-cycle)** — Sleep consolidation for agent memory

## License

MIT © [SuperInstance](https://github.com/SuperInstance)

Part of the [Exocortex](https://github.com/SuperInstance/exocortex) project.
