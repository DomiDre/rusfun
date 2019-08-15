#![allow(non_snake_case)]

///Includes the Minimizer to optimize parameters of a model using the LM algorithm
pub mod curve_fit;

/// A module to handle a function with its directive, domain and parameters
pub mod func1d;

/// Size Distribution functions, for now only Gaussian
pub mod size_distribution;

/// Standard functions such as linear, parabola, exponential,... using ndarray
pub mod standard;

/// Helper Functions such as the LU decomposition and interface functions
pub mod utils;

/// Interface module to enable the call of Rust functions using WASM
pub mod wasm;

// Add Small-Angle Scattering Module
pub mod sas;
