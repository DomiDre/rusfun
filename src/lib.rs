#![allow(dead_code)]
#![allow(non_snake_case)]
pub mod curve_fit;
pub mod func1d;
pub mod size_distribution;
pub mod standard;
pub mod utils;
pub mod wasm;

pub use crate::curve_fit::Minimizer;
pub use crate::func1d::Func1D;
pub use crate::standard::parabola;
pub use crate::utils::{matrix_solve, LU_decomp};
pub use crate::wasm::*;
