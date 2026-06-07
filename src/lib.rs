//! Free energy principle for agent homeostasis.
//!
//! Implements variational free energy, generative models, recognition densities,
//! prediction error computation, and homeostasis maintenance.

// ── Module: variational_free_energy ──────────────────────────────────────────

pub mod variational_free_energy {
    /// Compute variational free energy: F = E_q[ln q(s)] - E_q[ln p(o,s)]
    /// Simplified as: F = KL[q(s) || p(s|o)] - ln p(o)
    /// Approximated by: F = complexity - accuracy
    pub fn free_energy(complexity: f64, accuracy: f64) -> f64 {
        complexity - accuracy
    }

    /// Compute complexity term: KL divergence between posterior and prior.
    pub fn complexity(posterior: &[f64], prior: &[f64]) -> f64 {
        posterior.iter().zip(prior.iter())
            .filter(|(&q, _)| q > 0.0)
            .map(|(q, p)| {
                let p_safe = if *p > 0.0 { *p } else { 1e-10 };
                q * (q / p_safe).ln()
            })
            .sum()
    }

    /// Compute accuracy term: expected log-likelihood.
    pub fn accuracy(log_likelihoods: &[f64], weights: &[f64]) -> f64 {
        log_likelihoods.iter().zip(weights.iter())
            .map(|(&ll, &w)| w * ll)
            .sum()
    }

    /// Compute expected free energy for a policy.
    pub fn expected_free_energy(
        risk: f64,       // expected divergence from preferred states
        ambiguity: f64,  // expected ambiguity of observations
    ) -> f64 {
        risk + ambiguity
    }

    /// Compute surprise (negative log probability).
    pub fn surprise(probability: f64) -> f64 {
        if probability > 0.0 { -probability.ln() } else { f64::MAX }
    }

    /// Compute variational bound (ELBO).
    pub fn elbo(complexity: f64, accuracy: f64) -> f64 {
        accuracy - complexity
    }

    /// Gradient of free energy w.r.t. variational parameters.
    pub fn free_energy_gradient(values: &[f64], targets: &[f64]) -> Vec<f64> {
        values.iter().zip(targets.iter())
            .map(|(&v, &t)| v - t)
            .collect()
    }

    /// Optimize variational parameters to minimize free energy.
    pub fn optimize_variational(
        initial: &[f64],
        target: &[f64],
        learning_rate: f64,
        iterations: usize,
    ) -> Vec<f64> {
        let mut params = initial.to_vec();
        for _ in 0..iterations {
            let grad = free_energy_gradient(&params, target);
            for i in 0..params.len() {
                params[i] -= learning_rate * grad[i];
                if params[i] < 0.0 { params[i] = 0.0; }
            }
        }
        // Normalize
        let sum: f64 = params.iter().sum();
        if sum > 0.0 {
            for p in &mut params { *p /= sum; }
        }
        params
    }
}

// ── Module: generative_model ─────────────────────────────────────────────────

pub mod generative_model {
    /// A simple generative model mapping hidden states to observations.
    #[derive(Clone, Debug)]
    pub struct GenerativeModel {
        pub state_dim: usize,
        pub obs_dim: usize,
        pub likelihood: Vec<Vec<f64>>,  // state -> observation distribution
        pub transition: Vec<Vec<f64>>,  // state -> next state distribution
        pub prior: Vec<f64>,            // prior over states
    }

    impl GenerativeModel {
        pub fn new(state_dim: usize, obs_dim: usize) -> Self {
            let uniform_state = 1.0 / state_dim as f64;
            let uniform_obs = 1.0 / obs_dim as f64;
            GenerativeModel {
                state_dim,
                obs_dim,
                likelihood: (0..state_dim).map(|_| vec![uniform_obs; obs_dim]).collect(),
                transition: (0..state_dim).map(|_| vec![uniform_state; state_dim]).collect(),
                prior: vec![uniform_state; state_dim],
            }
        }

        pub fn set_likelihood(&mut self, state: usize, dist: Vec<f64>) {
            if state < self.state_dim && dist.len() == self.obs_dim {
                self.likelihood[state] = dist;
            }
        }

        pub fn set_transition(&mut self, state: usize, dist: Vec<f64>) {
            if state < self.state_dim && dist.len() == self.state_dim {
                self.transition[state] = dist;
            }
        }

        pub fn set_prior(&mut self, prior: Vec<f64>) {
            if prior.len() == self.state_dim {
                self.prior = prior;
            }
        }

        /// Generate an observation distribution given a state.
        pub fn predict_observation(&self, state: usize) -> &[f64] {
            &self.likelihood[state]
        }

        /// Predict next state distribution.
        pub fn predict_next_state(&self, state: usize) -> &[f64] {
            &self.transition[state]
        }

        /// Compute joint probability p(s, o) = p(o|s) * p(s).
        pub fn joint_probability(&self, state: usize, obs: usize) -> f64 {
            let p_obs_given_s = self.likelihood.get(state)
                .and_then(|d| d.get(obs))
                .copied()
                .unwrap_or(0.0);
            let p_s = self.prior.get(state).copied().unwrap_or(0.0);
            p_obs_given_s * p_s
        }

        /// Compute log model evidence: ln p(o) = ln Σ_s p(o|s)p(s).
        pub fn log_evidence(&self, obs: usize) -> f64 {
            let p_obs: f64 = (0..self.state_dim)
                .map(|s| self.joint_probability(s, obs))
                .sum();
            if p_obs > 0.0 { p_obs.ln() } else { f64::NEG_INFINITY }
        }

        /// Model entropy H[p(s)] = -Σ p(s) ln p(s).
        pub fn model_entropy(&self) -> f64 {
            self.prior.iter()
                .filter(|&&p| p > 0.0)
                .map(|&p| -p * p.ln())
                .sum()
        }

        /// Update model with new parameters (online learning).
        pub fn update_prior(&mut self, posterior: &[f64], learning_rate: f64) {
            if posterior.len() != self.state_dim { return; }
            for i in 0..self.state_dim {
                self.prior[i] = (1.0 - learning_rate) * self.prior[i] + learning_rate * posterior[i];
            }
            // Normalize
            let sum: f64 = self.prior.iter().sum();
            if sum > 0.0 {
                for p in &mut self.prior { *p /= sum; }
            }
        }
    }
}

// ── Module: recognition_density ──────────────────────────────────────────────

pub mod recognition_density {
    /// A recognition density (approximate posterior) over hidden states.
    #[derive(Clone, Debug)]
    pub struct RecognitionDensity {
        pub means: Vec<f64>,
        pub precisions: Vec<f64>,  // inverse variances
        pub dim: usize,
    }

    impl RecognitionDensity {
        pub fn new(dim: usize) -> Self {
            RecognitionDensity {
                means: vec![0.0; dim],
                precisions: vec![1.0; dim],
                dim,
            }
        }

        pub fn with_params(means: Vec<f64>, precisions: Vec<f64>) -> Self {
            let dim = means.len();
            RecognitionDensity { means, precisions, dim }
        }

        /// Compute the density at a point (Gaussian approximation).
        pub fn density(&self, x: &[f64]) -> f64 {
            if x.len() != self.dim { return 0.0; }
            let mut log_p = 0.0;
            for i in 0..self.dim {
                let diff = x[i] - self.means[i];
                let var = 1.0 / self.precisions[i];
                log_p += -0.5 * (2.0 * std::f64::consts::PI * var).ln() - 0.5 * self.precisions[i] * diff * diff;
            }
            log_p.exp()
        }

        /// Compute entropy of the recognition density.
        pub fn entropy(&self) -> f64 {
            self.precisions.iter()
                .map(|&prec| {
                    let var = 1.0 / prec;
                    0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * var).ln()
                })
                .sum()
        }

        /// Update recognition density given prediction error.
        pub fn update(&mut self, prediction_error: &[f64], learning_rate: f64) {
            if prediction_error.len() != self.dim { return; }
            for i in 0..self.dim {
                self.means[i] += learning_rate * self.precisions[i] * prediction_error[i];
            }
        }

        /// Increase precision (reduce uncertainty).
        pub fn increase_precision(&mut self, factor: f64) {
            for p in &mut self.precisions {
                *p *= factor;
            }
        }

        /// Compute variance for each dimension.
        pub fn variances(&self) -> Vec<f64> {
            self.precisions.iter().map(|&p| 1.0 / p).collect()
        }

        /// Compute KL divergence to another recognition density.
        pub fn kl_to(&self, other: &RecognitionDensity) -> f64 {
            if self.dim != other.dim { return f64::MAX; }
            let mut kl = 0.0;
            for i in 0..self.dim {
                let var_self = 1.0 / self.precisions[i];
                let var_other = 1.0 / other.precisions[i];
                let diff = self.means[i] - other.means[i];
                kl += 0.5 * (var_other.ln() - var_self.ln() - 1.0 + var_self / var_other + other.precisions[i] * diff * diff);
            }
            kl
        }

        /// Sample from the density (deterministic approximation: return means).
        pub fn sample_deterministic(&self) -> &[f64] {
            &self.means
        }
    }
}

// ── Module: prediction_error ─────────────────────────────────────────────────

pub mod prediction_error {
    /// Compute simple prediction error: observed - predicted.
    pub fn simple_error(observed: &[f64], predicted: &[f64]) -> Vec<f64> {
        observed.iter().zip(predicted.iter())
            .map(|(&o, &p)| o - p)
            .collect()
    }

    /// Compute squared prediction error.
    pub fn squared_error(observed: &[f64], predicted: &[f64]) -> f64 {
        observed.iter().zip(predicted.iter())
            .map(|(&o, &p)| (o - p).powi(2))
            .sum()
    }

    /// Compute weighted prediction error.
    pub fn weighted_error(observed: &[f64], predicted: &[f64], precision: &[f64]) -> Vec<f64> {
        observed.iter().zip(predicted.iter()).zip(precision.iter())
            .map(|((&o, &p), &prec)| prec * (o - p))
            .collect()
    }

    /// Compute precision-weighted prediction error (PEPE).
    pub fn precision_weighted_pe(error: &[f64], precision: &[f64]) -> Vec<f64> {
        error.iter().zip(precision.iter())
            .map(|(&e, &p)| p * e)
            .collect()
    }

    /// Compute mean absolute error.
    pub fn mae(observed: &[f64], predicted: &[f64]) -> f64 {
        let n = observed.len() as f64;
        observed.iter().zip(predicted.iter())
            .map(|(&o, &p)| (o - p).abs())
            .sum::<f64>() / n
    }

    /// Compute root mean squared error.
    pub fn rmse(observed: &[f64], predicted: &[f64]) -> f64 {
        squared_error(observed, predicted).sqrt()
    }

    /// Compute prediction error gradient for learning.
    pub fn error_gradient(error: &[f64], learning_rate: f64) -> Vec<f64> {
        error.iter().map(|&e| -learning_rate * e).collect()
    }

    /// Compute explanation for prediction error (saliency).
    pub fn saliency(error: &[f64]) -> f64 {
        error.iter().map(|e| e * e).sum::<f64>().sqrt()
    }
}

// ── Module: homeostasis ──────────────────────────────────────────────────────

pub mod homeostasis {
    /// A homeostatic system with set points and current values.
    #[derive(Clone, Debug)]
    pub struct HomeostaticSystem {
        pub set_points: Vec<f64>,
        pub current: Vec<f64>,
        pub bounds: Vec<(f64, f64)>,  // (min, max) for each variable
        pub tolerance: f64,
    }

    impl HomeostaticSystem {
        pub fn new(set_points: Vec<f64>, bounds: Vec<(f64, f64)>, tolerance: f64) -> Self {
            let current = set_points.clone();
            HomeostaticSystem { set_points, current, bounds, tolerance }
        }

        /// Check if the system is in homeostasis (all variables within tolerance).
        pub fn is_homeostatic(&self) -> bool {
            self.deviation() <= self.tolerance
        }

        /// Compute total deviation from set points.
        pub fn deviation(&self) -> f64 {
            self.current.iter().zip(self.set_points.iter())
                .map(|(&c, &s)| (c - s).abs())
                .sum::<f64>() / self.current.len() as f64
        }

        /// Compute per-variable deviation.
        pub fn per_variable_deviation(&self) -> Vec<f64> {
            self.current.iter().zip(self.set_points.iter())
                .map(|(&c, &s)| (c - s).abs())
                .collect()
        }

        /// Apply a perturbation to the system.
        pub fn perturb(&mut self, perturbation: &[f64]) {
            for i in 0..self.current.len() {
                if i < perturbation.len() {
                    self.current[i] += perturbation[i];
                    // Clamp to bounds
                    self.current[i] = self.current[i].max(self.bounds[i].0).min(self.bounds[i].1);
                }
            }
        }

        /// Apply a corrective action toward set points.
        pub fn correct(&mut self, strength: f64) {
            for i in 0..self.current.len() {
                let error = self.set_points[i] - self.current[i];
                self.current[i] += strength * error;
                self.current[i] = self.current[i].max(self.bounds[i].0).min(self.bounds[i].1);
            }
        }

        /// Compute homeostatic drive (urgency to return to set points).
        pub fn drive(&self) -> Vec<f64> {
            self.current.iter().zip(self.set_points.iter())
                .map(|(&c, &s)| s - c)
                .collect()
        }

        /// Compute viability (how far from boundary limits).
        pub fn viability(&self) -> f64 {
            self.current.iter().zip(self.bounds.iter())
                .map(|(&c, &(lo, hi))| {
                    let range = hi - lo;
                    if range == 0.0 { return 1.0; }
                    let dist_to_boundary = (c - lo).min(hi - c);
                    (2.0 * dist_to_boundary / range).min(1.0)
                })
                .product()
        }

        /// Check if any variable is at a boundary.
        pub fn at_boundary(&self) -> bool {
            self.current.iter().zip(self.bounds.iter())
                .any(|(&c, &(lo, hi))| c <= lo || c >= hi)
        }

        /// Simulate one homeostatic regulation step.
        pub fn regulate_step(&mut self, perturbation: &[f64], correction_strength: f64) {
            self.perturb(perturbation);
            self.correct(correction_strength);
        }
    }

    /// Compute allostasis cost (anticipatory regulation energy).
    pub fn allostasis_cost(drive: &[f64], anticipation: &[f64]) -> f64 {
        drive.iter().zip(anticipation.iter())
            .map(|(&d, &a)| (d - a).powi(2))
            .sum()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Variational free energy tests ──

    #[test]
    fn test_free_energy_basic() {
        let f = variational_free_energy::free_energy(1.0, 0.5);
        assert!((f - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_complexity_same_dist() {
        let c = variational_free_energy::complexity(&[0.5, 0.5], &[0.5, 0.5]);
        assert!(c.abs() < 1e-10);
    }

    #[test]
    fn test_complexity_different_dist() {
        let c = variational_free_energy::complexity(&[0.9, 0.1], &[0.5, 0.5]);
        assert!(c > 0.0);
    }

    #[test]
    fn test_accuracy() {
        let a = variational_free_energy::accuracy(&[-1.0, -2.0], &[0.5, 0.5]);
        assert!((a - (-1.5)).abs() < 1e-10);
    }

    #[test]
    fn test_expected_free_energy() {
        let efe = variational_free_energy::expected_free_energy(0.5, 0.3);
        assert!((efe - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_surprise() {
        let s = variational_free_energy::surprise(0.1);
        assert!(s > 0.0);
        assert!((s - (-0.1_f64.ln())).abs() < 1e-10);
    }

    #[test]
    fn test_surprise_zero() {
        let s = variational_free_energy::surprise(0.0);
        assert_eq!(s, f64::MAX);
    }

    #[test]
    fn test_elbo() {
        let elbo = variational_free_energy::elbo(1.0, 3.0);
        assert!((elbo - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_gradient() {
        let g = variational_free_energy::free_energy_gradient(&[0.3, 0.7], &[0.5, 0.5]);
        assert!((g[0] - (-0.2)).abs() < 1e-10);
        assert!((g[1] - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_optimize_variational() {
        let result = variational_free_energy::optimize_variational(
            &[0.5, 0.5], &[0.9, 0.1], 0.1, 100
        );
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(result[0] > 0.5); // Should move toward 0.9
    }

    // ── Generative model tests ──

    #[test]
    fn test_generative_model_creation() {
        let m = generative_model::GenerativeModel::new(3, 2);
        assert_eq!(m.state_dim, 3);
        assert_eq!(m.obs_dim, 2);
    }

    #[test]
    fn test_set_likelihood() {
        let mut m = generative_model::GenerativeModel::new(2, 2);
        m.set_likelihood(0, vec![0.9, 0.1]);
        assert!((m.likelihood[0][0] - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_set_transition() {
        let mut m = generative_model::GenerativeModel::new(2, 2);
        m.set_transition(0, vec![0.8, 0.2]);
        assert!((m.transition[0][0] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_predict_observation() {
        let mut m = generative_model::GenerativeModel::new(2, 3);
        m.set_likelihood(0, vec![0.7, 0.2, 0.1]);
        let obs = m.predict_observation(0);
        assert!((obs[0] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_predict_next_state() {
        let mut m = generative_model::GenerativeModel::new(3, 2);
        m.set_transition(0, vec![0.6, 0.3, 0.1]);
        let next = m.predict_next_state(0);
        assert!((next[0] - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_joint_probability() {
        let mut m = generative_model::GenerativeModel::new(2, 2);
        m.set_likelihood(0, vec![0.8, 0.2]);
        m.set_prior(vec![0.5, 0.5]);
        let jp = m.joint_probability(0, 0);
        assert!((jp - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_log_evidence() {
        let mut m = generative_model::GenerativeModel::new(2, 2);
        m.set_likelihood(0, vec![0.9, 0.1]);
        m.set_likelihood(1, vec![0.1, 0.9]);
        m.set_prior(vec![0.5, 0.5]);
        let le = m.log_evidence(0);
        assert!(le > 0.0_f64.ln()); // Should be > ln(0.5)
    }

    #[test]
    fn test_model_entropy() {
        let mut m = generative_model::GenerativeModel::new(2, 2);
        m.set_prior(vec![0.5, 0.5]);
        let h = m.model_entropy();
        assert!((h - 0.6931_f64).abs() < 0.01);
    }

    #[test]
    fn test_update_prior() {
        let mut m = generative_model::GenerativeModel::new(2, 2);
        m.set_prior(vec![0.5, 0.5]);
        m.update_prior(&[0.8, 0.2], 0.5);
        assert!(m.prior[0] > 0.5);
    }

    // ── Recognition density tests ──

    #[test]
    fn test_recognition_density_creation() {
        let rd = recognition_density::RecognitionDensity::new(3);
        assert_eq!(rd.dim, 3);
        assert_eq!(rd.means.len(), 3);
    }

    #[test]
    fn test_recognition_density_with_params() {
        let rd = recognition_density::RecognitionDensity::with_params(
            vec![1.0, 2.0], vec![0.5, 1.0]
        );
        assert_eq!(rd.means[0], 1.0);
        assert_eq!(rd.precisions[1], 1.0);
    }

    #[test]
    fn test_density_at_mean() {
        let rd = recognition_density::RecognitionDensity::with_params(
            vec![0.0], vec![1.0]
        );
        let d = rd.density(&[0.0]);
        assert!(d > 0.0);
    }

    #[test]
    fn test_density_away_from_mean() {
        let rd = recognition_density::RecognitionDensity::with_params(
            vec![0.0], vec![1.0]
        );
        let d_near = rd.density(&[0.0]);
        let d_far = rd.density(&[5.0]);
        assert!(d_near > d_far);
    }

    #[test]
    fn test_entropy() {
        let rd = recognition_density::RecognitionDensity::with_params(
            vec![0.0], vec![1.0]
        );
        let h = rd.entropy();
        assert!(h > 0.0);
    }

    #[test]
    fn test_update_means() {
        let mut rd = recognition_density::RecognitionDensity::with_params(
            vec![0.0, 0.0], vec![1.0, 1.0]
        );
        rd.update(&[0.5, -0.3], 0.1);
        assert!(rd.means[0] > 0.0);
        assert!(rd.means[1] < 0.0);
    }

    #[test]
    fn test_increase_precision() {
        let mut rd = recognition_density::RecognitionDensity::with_params(
            vec![0.0], vec![1.0]
        );
        rd.increase_precision(2.0);
        assert_eq!(rd.precisions[0], 2.0);
    }

    #[test]
    fn test_variances() {
        let rd = recognition_density::RecognitionDensity::with_params(
            vec![0.0], vec![2.0, 0.5]
        );
        let v = rd.variances();
        assert!((v[0] - 0.5).abs() < 1e-10);
        assert!((v[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_kl_same() {
        let rd1 = recognition_density::RecognitionDensity::with_params(
            vec![0.0], vec![1.0]
        );
        let rd2 = recognition_density::RecognitionDensity::with_params(
            vec![0.0], vec![1.0]
        );
        let kl = rd1.kl_to(&rd2);
        assert!(kl.abs() < 1e-10);
    }

    #[test]
    fn test_kl_different_means() {
        let rd1 = recognition_density::RecognitionDensity::with_params(
            vec![0.0], vec![1.0]
        );
        let rd2 = recognition_density::RecognitionDensity::with_params(
            vec![5.0], vec![1.0]
        );
        let kl = rd1.kl_to(&rd2);
        assert!(kl > 0.0);
    }

    #[test]
    fn test_sample_deterministic() {
        let rd = recognition_density::RecognitionDensity::with_params(
            vec![1.0, 2.0], vec![1.0, 1.0]
        );
        assert_eq!(rd.sample_deterministic(), &[1.0, 2.0]);
    }

    // ── Prediction error tests ──

    #[test]
    fn test_simple_error() {
        let e = prediction_error::simple_error(&[1.0, 2.0], &[0.8, 2.3]);
        assert!((e[0] - 0.2).abs() < 1e-10);
        assert!((e[1] - (-0.3)).abs() < 1e-10);
    }

    #[test]
    fn test_squared_error() {
        let se = prediction_error::squared_error(&[1.0], &[0.5]);
        assert!((se - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_error() {
        let e = prediction_error::weighted_error(&[1.0], &[0.5], &[2.0]);
        assert!((e[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_precision_weighted_pe() {
        let pe = prediction_error::precision_weighted_pe(&[0.5], &[2.0]);
        assert!((pe[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mae() {
        let mae = prediction_error::mae(&[1.0, 2.0], &[1.5, 1.5]);
        assert!((mae - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_rmse() {
        let rmse = prediction_error::rmse(&[1.0, 2.0], &[1.0, 2.0]);
        assert!(rmse.abs() < 1e-10);
    }

    #[test]
    fn test_error_gradient() {
        let g = prediction_error::error_gradient(&[0.5], 0.1);
        assert!((g[0] - (-0.05)).abs() < 1e-10);
    }

    #[test]
    fn test_saliency() {
        let s = prediction_error::saliency(&[3.0, 4.0]);
        assert!((s - 5.0).abs() < 1e-10);
    }

    // ── Homeostasis tests ──

    #[test]
    fn test_homeostatic_system_creation() {
        let h = homeostasis::HomeostaticSystem::new(
            vec![37.0], vec![(35.0, 42.0)], 1.0
        );
        assert_eq!(h.current[0], 37.0);
    }

    #[test]
    fn test_is_homeostatic() {
        let h = homeostasis::HomeostaticSystem::new(
            vec![37.0], vec![(35.0, 42.0)], 1.0
        );
        assert!(h.is_homeostatic());
    }

    #[test]
    fn test_deviation_zero() {
        let h = homeostasis::HomeostaticSystem::new(
            vec![37.0], vec![(35.0, 42.0)], 1.0
        );
        assert!(h.deviation().abs() < 1e-10);
    }

    #[test]
    fn test_perturb_and_correct() {
        let mut h = homeostasis::HomeostaticSystem::new(
            vec![37.0], vec![(35.0, 42.0)], 1.0
        );
        h.perturb(&[2.0]);
        assert!((h.current[0] - 39.0).abs() < 1e-10);
        h.correct(0.5);
        assert!(h.current[0] < 39.0);
    }

    #[test]
    fn test_drive() {
        let mut h = homeostasis::HomeostaticSystem::new(
            vec![37.0], vec![(35.0, 42.0)], 1.0
        );
        h.perturb(&[2.0]);
        let d = h.drive();
        assert!((d[0] - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_viability() {
        let h = homeostasis::HomeostaticSystem::new(
            vec![38.0], vec![(36.0, 40.0)], 1.0
        );
        assert!(h.viability() > 0.0);
        assert!(h.viability() <= 1.0);
    }

    #[test]
    fn test_at_boundary() {
        let mut h = homeostasis::HomeostaticSystem::new(
            vec![37.0], vec![(35.0, 42.0)], 1.0
        );
        assert!(!h.at_boundary());
        h.perturb(&[10.0]); // Should clamp to 42
        assert!(h.at_boundary());
    }

    #[test]
    fn test_per_variable_deviation() {
        let mut h = homeostasis::HomeostaticSystem::new(
            vec![37.0, 100.0], vec![(35.0, 42.0), (90.0, 110.0)], 1.0
        );
        h.perturb(&[1.0, -2.0]);
        let dev = h.per_variable_deviation();
        assert!((dev[0] - 1.0).abs() < 1e-10);
        assert!((dev[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_regulate_step() {
        let mut h = homeostasis::HomeostaticSystem::new(
            vec![37.0], vec![(35.0, 42.0)], 1.0
        );
        h.regulate_step(&[3.0], 0.5);
        assert!(h.deviation() < 3.0);
    }

    #[test]
    fn test_allostasis_cost() {
        let cost = homeostasis::allostasis_cost(&[1.0, 2.0], &[0.5, 1.5]);
        assert!((cost - 0.5).abs() < 1e-10);
    }
}
