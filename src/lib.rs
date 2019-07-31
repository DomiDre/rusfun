#![allow(dead_code)]

mod standard;
mod utils;
mod func1d;
mod curve_fit;

pub use crate::standard::parabola;
pub use crate::func1d::Func1D;
pub use crate::curve_fit::{Residuum, std_residuum, lm};