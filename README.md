# lau-mirror-control

**Mirror symmetry between estimation and control: the Opus Pass 2.3 discovery.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Rust · `nalgebra` + `serde` + `num-complex`

---

## What This Does

This crate implements a **mirror symmetry** between two sides of agent theory:

- **A-model** (symplectic/HJB): Hamilton-Jacobi-Bellman optimal control, Lagrangian submanifolds, action functionals
- **B-model** (complex/Kalman-Hodge): Kalman filtering, Hodge decomposition, cohomology rings, harmonic forms

The mirror functor **M: A ↔ B** swaps estimation and control. Under this duality:
- Solving a Kalman filter problem *is* solving an HJB control problem (and vice versa)
- Lagrangian submanifolds on the A-side correspond to coherent sheaves on the B-side
- The symplectic form ω maps to the complex structure J
- The Obs ⊣ Ctrl adjunction is the decategorified shadow of the mirror functor

### Current Status

The A-model and mirror functor modules are implemented. The B-model, adjunction, and application modules are scaffolded (empty `mod.rs`) — awaiting the full mirror map implementation.

## Key Idea

In homological mirror symmetry (Kontsevich), the derived Fukaya category of a symplectic manifold is equivalent to the derived category of coherent sheaves on its mirror complex manifold. This crate applies that idea to agent systems:

- The **A-model** side is optimal control: HJB equation, symplectic geometry, Lagrangian submanifolds, action functionals
- The **B-model** side is state estimation: Kalman filtering, Hodge theory, cohomology, harmonic forms
- The **mirror functor** M maps data from one side to the other

The payoff: you can solve an estimation problem by solving its mirror control problem. If the HJB is intractable but the Kalman filter is easy, mirror-solve the Kalman and pull the answer back.

## Install

```toml
[dependencies]
lau-mirror-control = { git = "https://github.com/SuperInstance/lau-mirror-control" }
```

```bash
git clone https://github.com/SuperInstance/lau-mirror-control.git
cd lau-mirror-control
cargo build
```

## Quick Start

### Construct the Mirror Functor

```rust
use lau_mirror_control::{MirrorFunctor, SymplecticManifold};

// Create the canonical symplectic manifold R^{2n}
let symplectic = SymplecticManifold::canonical(3); // 6-D phase space

// Build the mirror functor from the symplectic form
let mirror = MirrorFunctor::from_symplectic(&symplectic.omega);

// Verify the mirror map: T · ω · T^T = J (complex structure)
assert!(mirror.verify_mirror_map(1e-10));
```

### A-Model: HJB Solver

```rust
use lau_mirror_control::{HJBSolver, SymplecticManifold, LagrangianSubmanifold};

let manifold = SymplecticManifold::canonical(2);
let mut solver = HJBSolver::new(&manifold);

// Solve the HJB equation for a quadratic cost
let cost_matrix = nalgebra::DMatrix::identity(4, 4);
let value_function = solver.solve_quadratic(&cost_matrix, 1.0);

// Extract Lagrangian submanifold from the solution
let lagrangian = solver.extract_lagrangian(&value_function);
assert!(lagrangian.verify_lagrangian(1e-10)); // ω|_L = 0
```

### Mirror Transformation

```rust
use lau_mirror_control::mirror::{AModelData, BModelData, MirrorFunctor};

let a_data = AModelData {
    symplectic_form: /* 2n × 2n matrix */,
    hamiltonian_coeffs: /* n-vector */,
    dim: 3,
    cost_matrix: /* Q */,
    control_cost: /* R */,
    dynamics: /* A */,
    input_matrix: /* B */,
};

let mirror = MirrorFunctor::from_symplectic(&a_data.symplectic_form);

// Transform A-model data to B-model data
let b_data = mirror.a_to_b(&a_data);

// Solve on the B-side (estimation) and pull back
// (full implementation pending in applications module)
```

## API Reference

| Module | Key Types | Status | Purpose |
|--------|-----------|--------|---------|
| `mirror` | `MirrorFunctor`, `AModelData`, `BModelData` | ✅ Implemented | Mirror map and data bundles |
| `amodel` | `SymplecticManifold`, `HJBSolver`, `LagrangianSubmanifold`, `ActionFunctional` | ✅ Implemented | Symplectic / HJB side |
| `bmodel` | `KalmanFilter`, `HodgeDecomposer`, `CohomologyRing`, `HarmonicForm` | 🔧 Scaffolded | Complex / Kalman-Hodge side |
| `adjunction` | `ObsCtrlAdjunction` | 🔧 Scaffolded | Decategorified mirror trace |
| `applications` | `MirrorSolver` | 🔧 Scaffolded | Solve estimation via control (and vice versa) |

## How It Works

### The Mirror Functor

Given a symplectic form ω on a 2n-dimensional manifold, the mirror functor constructs:

1. A **complex structure** J such that J² = −I and g(u,v) = ω(u, Jv) is positive definite
2. A **mirror map** T that converts A-model data (ω, H, Q, R, A, B) to B-model data (J, cohomology class, R_obs, Q_proc, C, A)
3. An **inverse mirror map** T⁻¹ for the reverse direction

The mirror map satisfies:
- Tᵀ · ω · T = J (symplectic → complex)
- T · J · Tᵀ = −ω (complex → symplectic, up to sign)

### A-Model: HJB + Symplectic

The Hamilton-Jacobi-Bellman equation for optimal control:

```
∂V/∂t + min_u { L(x, u) + ∇V · f(x, u) } = 0
```

For LQR (linear dynamics, quadratic cost), this reduces to the algebraic Riccati equation:

```
AᵀP + PA − PBR⁻¹BᵀP + Q = 0
```

The value function V(x) = xᵀPx is quadratic, and the optimal policy is u = −R⁻¹BᵀPx.

A **Lagrangian submanifold** L of (M, ω) satisfies dim L = n and ω|_L = 0. The optimal trajectory in phase space (x, p) lies on a Lagrangian submanifold defined by p = ∇V(x).

### B-Model: Kalman-Hodge (Planned)

The Kalman filter on the B-side is the mirror of the HJB controller. The observation update:

```
x̂_{k|k} = x̂_{k|k-1} + K_k(y_k − H x̂_{k|k-1})
```

Under the mirror map, the Kalman gain K corresponds to the LQR gain, the observation covariance R corresponds to the control cost R, and the process covariance Q corresponds to the state cost Q.

Hodge decomposition of the observation stream:

```
ω = dα + δβ + γ
```

gives exact (signal), co-exact (innovation), and harmonic (persistent bias) components.

## The Math

### Homological Mirror Symmetry (Kontsevich)

For a symplectic manifold (M, ω) with mirror (X, J), Kontsevich's conjecture states:

```
D^π Fuk(M, ω) ≅ D^b Coh(X, J)
```

The derived Fukaya category (A-model: Lagrangian submanifolds + Floer homology) is equivalent to the derived category of coherent sheaves (B-model: holomorphic bundles + Ext groups).

### The Mirror Map

For the standard symplectic form ω₀ on R^{2n}:

```
ω₀ = Σᵢ dpᵢ ∧ dqᵢ    (canonical symplectic form)
J₀ = [[0, −I], [I, 0]]  (canonical complex structure)
```

The mirror map T is constructed so that:

```
Tᵀ · ω₀ · T = J₀
T · J₀ · Tᵀ = −ω₀
```

This is essentially a change of basis that swaps the symplectic and complex structures. For the canonical case, T is the identity (or a simple permutation).

### HJB ↔ Kalman Duality

The key correspondence under the mirror map:

| A-Model (Control) | B-Model (Estimation) |
|-------------------|----------------------|
| State cost Q | Process noise Q |
| Control cost R | Observation noise R |
| Dynamics A | Dynamics A |
| Input matrix B | Observation matrix H |
| Riccati gain K = R⁻¹BᵀP | Kalman gain K = PHᵀR⁻¹ |
| Value function V(x) = xᵀPx | Estimation error P = Cov(x − x̂) |

Solving the HJB (control) gives you the Kalman filter (estimation) for free, via the mirror map.

### Lagrangian Submanifolds

A submanifold L ⊂ (M, ω) is Lagrangian if dim L = ½ dim M and ω|_L = 0. For the optimal control problem, the graph of the value function gradient:

```
L = { (x, ∇V(x)) : x ∈ R^n } ⊂ R^{2n}
```

is Lagrangian because ω(d∇V · v, w) = 0 when restricted to the graph (by symmetry of the Hessian ∂²V/∂xᵢ∂xⱼ).

### Action Functional

The action functional on the A-model side:

```
S[γ] = ∫₀ᵀ (p · q̇ − H(q, p)) dt = ∫₀ᵀ L(q, q̇) dt
```

Under the mirror map, this corresponds to the log-likelihood of the observation sequence on the B-model side:

```
log P(y₀:T | x₀:T) = −½ Σₖ (yₖ − Hxₖ)ᵀ R⁻¹ (yₖ − Hxₖ) + const
```

Minimizing the action (A-model) = maximizing the likelihood (B-model).

## License

MIT
