# free-energy

> **Minimize surprise. Maximize survival. The Free Energy Principle, computed.**

[![crates.io](https://img.shields.io/crates/v/free-energy.svg)](https://crates.io/crates/free-energy)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Computational implementation of variational free energy from the Free Energy Principle (FEP). Models agents as systems that minimize the difference between their internal model and the true causes of sensory input.

## Variational Free Energy

F = -ln p(s|m) + KL[q(z) || p(z|s,m)]

Where:
- s = sensory input
- m = agent's model
- z = latent causes
- q(z) = agent's posterior belief
- KL = Kullback-Leibler divergence

The agent minimizes F by updating its beliefs to better predict sensory input.

## License

MIT © [SuperInstance](https://github.com/SuperInstance)

Part of the [Exocortex](https://github.com/SuperInstance/exocortex) project.
