//! # lau-mirror-control
//!
//! Mirror symmetry between estimation and control: Opus Pass 2.3 discovery.
//!
//! The adjunction Obs ⊣ Ctrl is the decategorified shadow of a mirror functor
//! that swaps the A-model (symplectic/HJB) and B-model (complex/Kalman-Hodge) sides.
//!
//! ## Architecture
//!
//! - **A-model side**: Hamilton-Jacobi-Bellman, symplectic topology, Lagrangian submanifolds
//! - **B-model side**: Kalman filtering, Hodge theory, harmonic forms, cohomology
//! - **Mirror functor**: swaps estimation ↔ control
//! - **Mirror map**: explicit transformation taking A-model data to B-model data and back
//! - **Obs ⊣ Ctrl**: decategorified trace of the mirror functor
//! - **Applications**: solve estimation by solving mirror control (and vice versa)

pub mod amodel;
pub mod bmodel;
pub mod mirror;
pub mod adjunction;
pub mod applications;

pub use mirror::MirrorFunctor;
pub use amodel::{HJBSolver, SymplecticManifold, LagrangianSubmanifold, ActionFunctional};
pub use bmodel::{KalmanFilter, HodgeDecomposer, CohomologyRing, HarmonicForm};
pub use adjunction::ObsCtrlAdjunction;
pub use applications::MirrorSolver;

/// Re-export nalgebra types for convenience
pub use nalgebra::{DMatrix, DVector, DMatrix as Matrix, DVector as Vector};
