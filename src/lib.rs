#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_variables)]
mod standard;
mod utils;
mod func1d;
mod curve_fit;
mod wasm;

pub use crate::wasm::*;

pub use crate::standard::parabola;
pub use crate::func1d::Func1D;
pub use crate::curve_fit::{Minimizer};
pub use crate::utils::{LU_decomp, matrix_solve};

#[cfg(test)]
mod tests {
	use super::*;
	use ndarray::{array, Array1, Array2};
    #[test]
    fn calculate_parabola() {
			let p = array![2.0, 1.0, 0.5];
			let x: Array1<f64> = Array1::range(0.0, 10.0, 1.0);
			let y: Array1<f64> = x.map(|x| 2.0*x.powi(2) + 1.0*x + 0.5);
			let parab = Func1D::new(&p, &x, parabola);
			// let sy: Array1<f64> = x.map(|x| 1.0+4.0*x.abs());
			assert_eq!(y, parab.output());
    }

		#[test]
		fn solve_system_of_linear_equations() {
			let A: Array2<f64> = array![
        [1.0, 3.0, 5.0],
        [2.0, 4.0, 7.0],
        [1.0, 1.0, 0.0],
    	];
			let b: Array1<f64> = array![1.0, 2.0, 3.0];
			let x = matrix_solve(&A, &b);
			assert_eq!(A.dot(&x), b);
    }

}