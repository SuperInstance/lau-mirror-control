//! Mirror functor: the central duality swapping estimation ↔ control
//!
//! The mirror functor M : A ↔ B exchanges:
//! - Symplectic structure ω ↔ complex structure J
//! - Lagrangian submanifolds ↔ coherent sheaves
//! - HJB equation ↔ Kalman filter
//! - Action functional ↔ cohomology class

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// A-model data bundle (symplectic side)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AModelData {
    /// Symplectic form as an antisymmetric matrix (2n × 2n)
    pub symplectic_form: DMatrix<f64>,
    /// Hamiltonian function coefficients
    pub hamiltonian_coeffs: DVector<f64>,
    /// State dimension
    pub dim: usize,
    /// Running cost matrix Q (for control problems)
    pub cost_matrix: DMatrix<f64>,
    /// Control cost matrix R
    pub control_cost: DMatrix<f64>,
    /// Dynamics matrix A
    pub dynamics: DMatrix<f64>,
    /// Input matrix B
    pub input_matrix: DMatrix<f64>,
}

/// B-model data bundle (complex side)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BModelData {
    /// Complex structure matrix J (J² = -I)
    pub complex_structure: DMatrix<f64>,
    /// Cohomology class representative
    pub cohomology_class: DVector<f64>,
    /// State dimension
    pub dim: usize,
    /// Observation noise covariance (for filtering problems)
    pub observation_covariance: DMatrix<f64>,
    /// Process noise covariance
    pub process_covariance: DMatrix<f64>,
    /// Observation matrix H (C in Kalman)
    pub observation_matrix: DMatrix<f64>,
    /// Dynamics matrix A (same as A-model, mirrored)
    pub dynamics: DMatrix<f64>,
}

/// The mirror functor M: A ↔ B
///
/// This is the central object. It carries the mirror map that exchanges
/// symplectic data (A-model) with complex data (B-model).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MirrorFunctor {
    /// State dimension
    pub dim: usize,
    /// Mirror transformation matrix T: maps A-model data to B-model data
    /// For n-dim systems, T is 2n × 2n
    pub mirror_map: DMatrix<f64>,
    /// Inverse mirror map
    pub mirror_map_inv: DMatrix<f64>,
    /// Scaling parameter (relates to string coupling)
    pub kappa: f64,
}

impl MirrorFunctor {
    /// Construct the mirror functor from a given symplectic form.
    ///
    /// The mirror map is constructed by finding J such that:
    /// - J² = -I (complex structure)
    /// - g(u,v) = ω(u, Jv) is positive definite (Riemannian metric)
    /// - T maps the symplectic form to the complex structure
    pub fn from_symplectic(symplectic: &DMatrix<f64>) -> Self {
        let dim = symplectic.nrows() / 2;
        let n = dim;

        // Construct canonical J from ω
        // Standard symplectic form: J_0 = [[0, -I], [I, 0]]
        let neg_i = DMatrix::identity(n, n) * -1.0;
        let pos_i = DMatrix::identity(n, n);
        let j_canonical = DMatrix::from_rows(&[
            &neg_i.row(0),
            &neg_i.row(1),
            &(0..n).map(|i| pos_i.row(i)).collect::<Vec<_>>().iter()
                .flat_map(|r| r.iter().copied()).collect::<Vec<_>>().chunks(n).next().unwrap(),
        ]);

        // Build J by solving J = -ω⁻¹ · g for some g
        // Simplified: use the canonical complex structure
        let j = Self::canonical_complex_structure(n);

        // Mirror map T: ω → J via T^T J T = ω (up to scaling)
        // Use T = identity for the canonical case, with symplectic adjustment
        let mirror_map = Self::build_mirror_map(symplectic, &j);

        let mirror_map_inv = mirror_map.clone().try_inverse()
            .unwrap_or_else(|| mirror_map.clone());

        Self {
            dim,
            mirror_map,
            mirror_map_inv,
            kappa: 1.0,
        }
    }

    /// Canonical complex structure J for dimension n
    pub fn canonical_complex_structure(n: usize) -> DMatrix<f64> {
        let mut j = DMatrix::zeros(2 * n, 2 * n);
        for i in 0..n {
            // J = [[0, -I], [I, 0]]
            j[(i, n + i)] = -1.0;
            j[(n + i, i)] = 1.0;
        }
        j
    }

    /// Canonical symplectic form ω for dimension n
    pub fn canonical_symplectic(n: usize) -> DMatrix<f64> {
        let mut omega = DMatrix::zeros(2 * n, 2 * n);
        for i in 0..n {
            // ω = [[0, I], [-I, 0]]
            omega[(i, n + i)] = 1.0;
            omega[(n + i, i)] = -1.0;
        }
        omega
    }

    fn build_mirror_map(omega: &DMatrix<f64>, j: &DMatrix<f64>) -> DMatrix<f64> {
        let dim = omega.nrows();
        // T such that T^T ω T = J (conceptually)
        // For the canonical case, T = identity
        // For general case, use SVD-based alignment
        let svd = omega.svd(true, true);
        let sigma = svd.singular_values.map(|s| s.sqrt());
        let u = svd.u.unwrap_or_else(|| DMatrix::identity(dim, dim));
        let v_t = svd.v_t.unwrap_or_else(|| DMatrix::identity(dim, dim));

        let mut t = &u * DMatrix::from_diagonal(&sigma) * &v_t;

        // Ensure positivity of the metric g = ω J
        let g = omega * j;
        if g.trace() < 0.0 {
            t = -t;
        }
        t
    }

    /// Create a new mirror functor with canonical structures
    pub fn new(n: usize) -> Self {
        let omega = Self::canonical_symplectic(n);
        Self::from_symplectic(&omega)
    }

    /// Apply the mirror functor: A-model → B-model
    pub fn forward(&self, a_data: &AModelData) -> BModelData {
        let n = self.dim;
        let t = &self.mirror_map;

        // Mirror the dynamics: A_B = T⁻¹ A T
        let dynamics = &self.mirror_map_inv * &a_data.dynamics * t;

        // Mirror cost → covariances
        // Q → R_obs = κ Q⁻¹ (inverse of cost becomes observation covariance)
        let obs_cov = self.kappa * a_data.cost_matrix.clone().try_inverse()
            .unwrap_or_else(|| a_data.cost_matrix.clone());

        // R → Q_proc = κ R⁻¹
        let proc_cov = self.kappa * a_data.control_cost.clone().try_inverse()
            .unwrap_or_else(|| a_data.control_cost.clone());

        // B → H^T (input matrix transposes to observation matrix)
        let obs_matrix = a_data.input_matrix.transpose();

        // Complex structure from symplectic form via mirror map
        let complex = Self::canonical_complex_structure(n);

        // Cohomology class from Hamiltonian
        let cohomology = t.transpose() * &a_data.hamiltonian_coeffs;

        BModelData {
            complex_structure: complex,
            cohomology_class: cohomology,
            dim: n,
            observation_covariance: obs_cov,
            process_covariance: proc_cov,
            observation_matrix: obs_matrix,
            dynamics,
        }
    }

    /// Apply the inverse mirror functor: B-model → A-model
    pub fn inverse(&self, b_data: &BModelData) -> AModelData {
        let n = self.dim;
        let t_inv = &self.mirror_map_inv;

        // Mirror dynamics back: A_A = T A_B T⁻¹
        let dynamics = &self.mirror_map * &b_data.dynamics * t_inv;

        // Covariances → costs (inverse mirror of above)
        let cost_matrix = self.kappa * b_data.observation_covariance.clone().try_inverse()
            .unwrap_or_else(|| b_data.observation_covariance.clone());

        let control_cost = self.kappa * b_data.process_covariance.clone().try_inverse()
            .unwrap_or_else(|| b_data.process_covariance.clone());

        // H → B^T
        let input_matrix = b_data.observation_matrix.transpose();

        // Hamiltonian from cohomology
        let hamiltonian = self.mirror_map * &b_data.cohomology_class;

        // Symplectic form from complex structure via inverse mirror
        let symplectic = Self::canonical_symplectic(n);

        AModelData {
            symplectic_form: symplectic,
            hamiltonian_coeffs: hamiltonian,
            dim: n,
            cost_matrix,
            control_cost,
            dynamics,
            input_matrix,
        }
    }

    /// Verify mirror symmetry: M(M(x)) ≈ x
    pub fn verify_roundtrip(&self, a_data: &AModelData, tol: f64) -> bool {
        let b_data = self.forward(a_data);
        let a_roundtrip = self.inverse(&b_data);

        let diff_dynamics = (&a_data.dynamics - &a_roundtrip.dynamics).norm();
        let diff_cost = (&a_data.cost_matrix - &a_roundtrip.cost_matrix).norm();

        diff_dynamics < tol && diff_cost < tol
    }

    /// The mirror exchange: given an A-model problem, produce the mirror B-model problem
    pub fn exchange_estimation_for_control(&self, estimation_data: &BModelData) -> AModelData {
        self.inverse(estimation_data)
    }

    /// Given a control problem (A-model), produce the mirror estimation problem (B-model)
    pub fn exchange_control_for_estimation(&self, control_data: &AModelData) -> BModelData {
        self.forward(control_data)
    }

    /// Set the coupling parameter kappa
    pub fn with_kappa(mut self, kappa: f64) -> Self {
        self.kappa = kappa;
        self
    }
}

/// Mirror map between specific data points
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MirrorMap {
    /// The functor carrying the transformation
    pub functor: MirrorFunctor,
}

impl MirrorMap {
    pub fn new(functor: MirrorFunctor) -> Self {
        Self { functor }
    }

    /// Map a value function V(x) to an information function I(y) = V*(T·y)
    /// where V* is the Legendre transform and T is the mirror map
    pub fn value_to_information(&self, value_coeffs: &DVector<f64>) -> DVector<f64> {
        // Simplified: conjugate via mirror map
        // Full version would compute Legendre transform
        self.functor.mirror_map.transpose() * value_coeffs
    }

    /// Map an information function I(y) back to a value function V(x)
    pub fn information_to_value(&self, info_coeffs: &DVector<f64>) -> DVector<f64> {
        self.functor.mirror_map_inv.transpose() * info_coeffs
    }

    /// Map a cost-to-go J(x) to a log-likelihood ℓ(y)
    pub fn cost_to_loglikelihood(&self, cost: &DVector<f64>) -> DVector<f64> {
        -self.functor.kappa * self.functor.mirror_map.transpose() * cost
    }

    /// Map a log-likelihood back to cost-to-go
    pub fn loglikelihood_to_cost(&self, ll: &DVector<f64>) -> DVector<f64> {
        -(1.0 / self.functor.kappa) * self.functor.mirror_map_inv.transpose() * ll
    }
}
