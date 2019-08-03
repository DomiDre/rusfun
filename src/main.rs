#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_variables)]
mod standard;
mod utils;
mod func1d;
mod curve_fit;

pub use crate::standard::parabola;
pub use crate::func1d::Func1D;
pub use crate::curve_fit::{Minimizer};
pub use crate::utils::matrix_solve;

use ndarray::{array, Array1};

fn main() {
	// define the model
	let p = array![2.5, 1.0, 0.5];
	let x: Array1<f64> = Array1::range(0.0, 10.0, 1.0);
	let parab = Func1D::new(&p, &x, parabola);

	// define the data
	let y: Array1<f64> = x.map(|x| x.powi(2));
	let sy: Array1<f64> = x.map(|x| 2.0);

	// fit data
	let mut minimizer = Minimizer::init(&parab, &y, &sy, 1.0);
	minimizer.minimize(10*p.len());
	minimizer.report();
}
