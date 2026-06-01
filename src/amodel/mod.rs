//! A-model side: symplectic topology, HJB, Lagrangian submanifolds, action functional
//!
//! The A-model lives on the symplectic side. Key objects:
//! - Hamilton-Jacobi-Bellman equation: ∂V/∂t + H(x, ∇V) = 0
//! - Symplectic manifold (M, ω)
//! - Lagrangian submanifolds L ⊂ M (dim L = n in 2n-dim M)
//! - Action functional S[γ] = ∫(p dq - H dt)

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// A symplectic manifold (M, ω) of dimension 2n, represented discretely
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymplecticManifold {
    /// Dimension n (manifold is 2n-dimensional)
    pub half_dim: usize,
    /// Symplectic form ω as antisymmetric 2n×2n matrix
    pub omega: DMatrix<f64>,
    /// Riemannian metric g (compatible: g(u,v) = ω(u, Jv))
    pub metric: DMatrix<f64>,
    /// Complex structure J (compatible: ω(u,v) = g(Ju, v))
    pub complex_structure: DMatrix<f64>,
}

impl SymplecticManifold {
    /// Create the canonical symplectic manifold R^{2n} with standard structures
    pub fn canonical(n: usize) -> Self {
        let mut omega = DMatrix::zeros(2 * n, 2 * n);
        for i in 0..n {
            omega[(i, n + i)] = 1.0;
            omega[(n + i, i)] = -1.0;
        }

        let mut j = DMatrix::zeros(2 * n, 2 * n);
        for i in 0..n {
            j[(i, n + i)] = -1.0;
            j[(n + i, i)] = 1.0;
        }

        // g = ω · J should be positive definite (identity for canonical)
        let g = &omega * &j;

        Self {
            half_dim: n,
            omega,
            metric: g,
            complex_structure: j,
        }
    }

    /// Create from a given symplectic form
    pub fn from_symplectic(omega: DMatrix<f64>) -> Self {
        let dim = omega.nrows();
        let n = dim / 2;

        let j = {
            let mut j = DMatrix::zeros(dim, dim);
            for i in 0..n {
                j[(i, n + i)] = -1.0;
                j[(n + i, i)] = 1.0;
            }
            j
        };

        let g = &omega * &j;

        Self {
            half_dim: n,
            omega,
            metric: g,
            complex_structure: j,
        }
    }

    /// Verify ω is antisymmetric: ω + ω^T = 0
    pub fn verify_antisymmetry(&self, tol: f64) -> bool {
        let sum = &self.omega + &self.omega.transpose();
        sum.norm() < tol
    }

    /// Verify ω is non-degenerate: det(ω) ≠ 0
    pub fn verify_nondegenerate(&self) -> bool {
        self omega.determinant().abs() > 1e-10
    }

    /// Verify compatibility: g(u,v) = ω(u, Jv) for all u,v
    pub fn verify_compatibility(&self, tol: f64) -> bool {
        let g_from_omega_j = &self.omega * &self.complex_structure;
        (&self.metric - &g_from_omega_j).norm() < tol
    }

    /// Symplectic area of a 2-chain given by vertices
    pub fn symplectic_area(&self, u: &DVector<f64>, v: &DVector<f64>) -> f64 {
        u.transpose() * &self.omega * v
    }

    /// Poisson bracket {f, g} = ω^{-1}(df, dg) at a point
    pub fn poisson_bracket(&self, df: &DVector<f64>, dg: &DVector<f64>) -> f64 {
        let omega_inv = self.omega.clone().try_inverse()
            .unwrap_or_else(|| DMatrix::identity(2 * self.half_dim, 2 * self.half_dim));
        df.transpose() * &omega_inv * dg
    }
}

/// Lagrangian submanifold L ⊂ (M, ω): dim L = n, ω|_L = 0
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LagrangianSubmanifold {
    /// Base point in R^{2n}
    pub base_point: DVector<f64>,
    /// Tangent space basis as 2n × n matrix
    pub tangent_basis: DMatrix<f64>,
    /// Ambient symplectic manifold dimension
    pub ambient_dim: usize,
}

impl LagrangianSubmanifold {
    /// Create a Lagrangian submanifold from a base point and tangent vectors
    pub fn new(base_point: DVector<f64>, tangent_basis: DMatrix<f64>) -> Self {
        let ambient_dim = base_point.len();
        Self { base_point, tangent_basis, ambient_dim }
    }

    /// Verify Lagrangian condition: ω(u, v) = 0 for all u, v ∈ TL
    pub fn verify_lagrangian(&self, omega: &DMatrix<f64>, tol: f64) -> bool {
        let n = self.tangent_basis.ncols();
        for i in 0..n {
            for j in 0..n {
                let u = self.tangent_basis.column(i);
                let v = self.tangent_basis.column(j);
                let area = u.transpose() * omega * v;
                if area[0].abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Graph-type Lagrangian: L = {(q, ∇S(q))} for generating function S
    pub fn from_generating_function(grad_s: &DVector<f64>, q: &DVector<f64>) -> Self {
        let n = q.len();
        let mut base = DVector::zeros(2 * n);
        base.rows_mut(0, n).copy_from(q);
        base.rows_mut(n, n).copy_from(grad_s);

        // Tangent: for each basis direction e_i in q, the tangent is (e_i, ∂²S/∂q_i∂q_j e_j)
        // Simplified: assume linear generating function, Hessian = 0
        let mut tangent = DMatrix::zeros(2 * n, n);
        for i in 0..n {
            tangent[(i, i)] = 1.0;
        }

        Self::new(base, tangent)
    }

    /// Graph-type Lagrangian with Hessian of generating function
    pub fn from_generating_function_with_hessian(
        grad_s: &DVector<f64>,
        q: &DVector<f64>,
        hessian_s: &DMatrix<f64>,
    ) -> Self {
        let n = q.len();
        let mut base = DVector::zeros(2 * n);
        base.rows_mut(0, n).copy_from(q);
        base.rows_mut(n, n).copy_from(grad_s);

        let mut tangent = DMatrix::zeros(2 * n, n);
        // Upper block: identity
        for i in 0..n {
            tangent[(i, i)] = 1.0;
        }
        // Lower block: Hessian
        tangent.rows_mut(n, n).copy_from(hessian_s);

        Self::new(base, tangent)
    }

    /// Project a point onto the Lagrangian submanifold
    pub fn project(&self, point: &DVector<f64>) -> DVector<f64> {
        let diff = point - &self.base_point;
        // Project onto tangent space
        let n = self.tangent_basis.ncols();
        let mut coords = DVector::zeros(n);
        for i in 0..n {
            let col = self.tangent_basis.column(i);
            coords[i] = col.dot(&diff) / col.dot(&col);
        }
        &self.base_point + &self.tangent_basis * &coords
    }
}

/// Action functional S[γ] = ∫(p dq - H dt) along a path
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionFunctional {
    /// Hamiltonian coefficients
    pub hamiltonian: DVector<f64>,
    /// Time step for discretization
    pub dt: f64,
    /// Number of time steps
    pub steps: usize,
}

impl ActionFunctional {
    pub fn new(hamiltonian: DVector<f64>, dt: f64, steps: usize) -> Self {
        Self { hamiltonian, dt, steps }
    }

    /// Evaluate S[γ] for a discrete path γ = (q_0, p_0, q_1, p_1, ..., q_T, p_T)
    /// Path is 2n × (T+1) matrix
    pub fn evaluate(&self, path: &DMatrix<f64>) -> f64 {
        let n = self.hamiltonian.len() / 2;
        let t = path.ncols();
        let mut action = 0.0;

        for k in 0..t.saturating_sub(1) {
            let q_k = path.rows(0, n).column(k);
            let p_k = path.rows(n, n).column(k);
            let q_next = path.rows(0, n).column(k + 1);
            let p_next = path.rows(n, n).column(k + 1);

            // p · Δq term
            let p_avg = (&p_k + &p_next) * 0.5;
            let dq = q_next - q_k;
            action += p_avg.dot(&dq);

            // -H dt term
            let mut state = DVector::zeros(2 * n);
            state.rows_mut(0, n).copy_from(&q_k);
            state.rows_mut(n, n).copy_from(&p_k);
            let h = self.hamiltonian.dot(&state);
            action -= h * self.dt;
        }

        action
    }

    /// Compute ∂S/∂γ (functional derivative) for Hamilton's equations
    pub fn gradient(&self, path: &DMatrix<f64>) -> DMatrix<f64> {
        let n = self.hamiltonian.len() / 2;
        let t = path.ncols();
        let mut grad = DMatrix::zeros(2 * n, t);

        for k in 0..t {
            let mut g = DVector::zeros(2 * n);

            if k > 0 {
                let q_prev = path.rows(0, n).column(k - 1);
                let p_prev = path.rows(n, n).column(k - 1);
                // Contribution from p·Δq
                g.rows_mut(n, n).axpy(0.5, &q_prev, 1.0);
                g.rows_mut(n, n).axpy(-0.5, &path.rows(0, n).column(k), 1.0);
            }

            if k < t - 1 {
                let q_next = path.rows(0, n).column(k + 1);
                let p_next = path.rows(n, n).column(k + 1);
                g.rows_mut(n, n).axpy(0.5, &q_next, 1.0);
                g.rows_mut(n, n).axpy(-0.5, &path.rows(0, n).column(k), 1.0);
            }

            // -∂H/∂x contribution
            g -= &self.hamiltonian * self.dt;

            grad.column_mut(k).copy_from(&g);
        }

        grad
    }

    /// Find a stationary path (action is extremized) via gradient descent
    pub fn stationary_path(&self, initial_path: &DMatrix<f64>, lr: f64, iterations: usize) -> DMatrix<f64> {
        let mut path = initial_path.clone();
        for _ in 0..iterations {
            let grad = self.gradient(&path);
            path -= grad * lr;
        }
        path
    }
}

/// Hamilton-Jacobi-Bellman solver
///
/// HJB equation: ∂V/∂t + min_u {L(x, u) + (∂V/∂x)·f(x, u)} = 0
/// Discretized as: V_k = min_u { L(x, u)·Δt + V_{k+1}(x + f(x,u)·Δt) }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HJBSolver {
    /// Dynamics matrix A (ẋ = Ax + Bu)
    pub dynamics: DMatrix<f64>,
    /// Input matrix B
    pub input_matrix: DMatrix<f64>,
    /// State cost Q
    pub cost_q: DMatrix<f64>,
    /// Control cost R
    pub cost_r: DMatrix<f64>,
    /// Time horizon
    pub horizon: usize,
    /// Time step
    pub dt: f64,
}

impl HJBSolver {
    pub fn new(
        dynamics: DMatrix<f64>,
        input_matrix: DMatrix<f64>,
        cost_q: DMatrix<f64>,
        cost_r: DMatrix<f64>,
        horizon: usize,
        dt: f64,
    ) -> Self {
        Self { dynamics, input_matrix, cost_q, cost_r, horizon, dt }
    }

    /// Solve the LQR (linear quadratic regulator) — the HJB solution for linear systems
    ///
    /// Returns the sequence of P matrices: V(x) = x^T P x
    /// P_k = Q + A^T P_{k+1} A - A^T P_{k+1} B (R + B^T P_{k+1} B)^{-1} B^T P_{k+1} A
    pub fn solve_lqr(&self) -> Vec<DMatrix<f64>> {
        let n = self.dynamics.nrows();
        let mut p_matrices = Vec::with_capacity(self.horizon + 1);

        // Terminal cost: P_T = Q
        let p_final = self.cost_q.clone();
        p_matrices.push(p_final);

        for _ in 0..self.horizon {
            let p_next = p_matrices.last().unwrap();
            let a = &self.dynamics;
            let b = &self.input_matrix;
            let q = &self.cost_q;
            let r = &self.cost_r;

            // S = R + B^T P B
            let s = r + b.transpose() * p_next * b;
            let s_inv = s.clone().try_inverse().unwrap_or_else(|| s.clone());

            // K = S^{-1} B^T P A (optimal gain)
            // P = Q + A^T P A - A^T P B S^{-1} B^T P A
            let at_p_a = a.transpose() * p_next * a;
            let at_p_b = a.transpose() * p_next * b;
            let p_new = q + at_p_a - &at_p_b * &s_inv * b.transpose() * p_next * a;

            p_matrices.push(p_new);
        }

        p_matrices.reverse();
        p_matrices
    }

    /// Get the optimal control gain K at each time step
    pub fn optimal_gains(&self) -> Vec<DMatrix<f64>> {
        let p_matrices = self.solve_lqr();
        let mut gains = Vec::new();

        for p in &p_matrices {
            let a = &self.dynamics;
            let b = &self.input_matrix;
            let r = &self.cost_r;

            let s = r + b.transpose() * p * b;
            let s_inv = s.clone().try_inverse().unwrap_or_else(|| s.clone());
            let k = s_inv * b.transpose() * p * a;

            gains.push(k);
        }

        gains
    }

    /// Compute the optimal control at time k for state x
    pub fn optimal_control(&self, k: usize, x: &DVector<f64>) -> DVector<f64> {
        let gains = self.optimal_gains();
        let k_matrix = gains.get(k).unwrap_or_else(|| gains.last().unwrap());
        let u = -k_matrix * x;
        u
    }

    /// Compute the value function V_k(x) = x^T P_k x
    pub fn value_function(&self, k: usize, x: &DVector<f64>) -> f64 {
        let p_matrices = self.solve_lqr();
        let p = p_matrices.get(k).unwrap_or_else(|| p_matrices.last().unwrap());
        let val = x.transpose() * p * x;
        val[0]
    }

    /// Verify the discrete HJB equation: V_k ≈ min_u { L(x,u)dt + V_{k+1}(x') }
    pub fn verify_hjb(&self, k: usize, x: &DVector<f64>, tol: f64) -> bool {
        let u = self.optimal_control(k, x);
        let dx = &self.dynamics * x + &self.input_matrix * &u;
        let x_next = x + dx * self.dt;

        let v_k = self.value_function(k, x);
        let v_next = self.value_function(k + 1, &x_next);
        let stage_cost = (x.transpose() * &self.cost_q * x)[0]
            + (u.transpose() * &self.cost_r * &u)[0];

        let hjb_residual = (v_k - (stage_cost * self.dt + v_next)).abs();
        hjb_residual < tol
    }

    /// Full rollout: compute optimal trajectory from initial state
    pub fn rollout(&self, x0: &DVector<f64>) -> (Vec<DVector<f64>>, Vec<DVector<f64>>) {
        let mut states = vec![x0.clone()];
        let mut controls = Vec::new();

        let mut x = x0.clone();
        for k in 0..self.horizon {
            let u = self.optimal_control(k, &x);
            let dx = &self.dynamics * &x + &self.input_matrix * &u;
            x = x + dx * self.dt;
            states.push(x.clone());
            controls.push(u);
        }

        (states, controls)
    }
}
