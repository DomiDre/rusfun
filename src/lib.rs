#![allow(dead_code)]
#![allow(non_snake_case)]
pub mod standard;
pub mod utils;
pub mod func1d;
pub mod curve_fit;
pub mod wasm;

pub use crate::wasm::*;

pub use crate::standard::parabola;
pub use crate::func1d::Func1D;
pub use crate::curve_fit::{Minimizer};
pub use crate::utils::{LU_decomp, matrix_solve};
