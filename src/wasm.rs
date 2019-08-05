use ndarray::Array1;

use crate::utils::array1_to_vec;
use crate::standard;
use crate::func1d;
use crate::curve_fit;

use wasm_bindgen::prelude::*;

pub fn get_function(function_name: &str) -> fn(&Array1<f64>, &Array1<f64>) -> Array1<f64> {
	match function_name {
		"linear" => standard::linear,
		"parabola" => standard::parabola,
		_ => standard::zero
	}
}

pub fn calculate_model(p: Vec<f64>, x: Vec<f64>, model: fn(&Array1<f64>, &Array1<f64>) -> Array1<f64>) -> Vec<f64> {
  // interface function between input from wasm as Vec<f64> to crate with Array1<f64>
  array1_to_vec((model)(&Array1::from(p), &Array1::from(x)))
}

#[wasm_bindgen]
pub fn linear(p: Vec<f64>, x: Vec<f64>) -> Vec<f64> {
  calculate_model(p, x, standard::linear)
}

#[wasm_bindgen]
pub fn parabola(p: Vec<f64>, x: Vec<f64>) -> Vec<f64> {
  calculate_model(p, x, standard::parabola)
}

#[wasm_bindgen]
pub struct FitResult {
	parameters: Vec<f64>,
	parameter_std_errors: Vec<f64>,
	num_func_evaluation: usize,
	chi2: f64,
	redchi2: f64,
	convergence_message: String
}

#[wasm_bindgen]
impl FitResult {
  pub fn parameters(&self) -> Vec<f64> {
    self.parameters.clone()
  }

  pub fn parameter_std_errors(&self) -> Vec<f64> {
    self.parameter_std_errors.clone()
  }

  pub fn num_func_evaluation(&self) -> usize {
    self.num_func_evaluation
  }
  pub fn chi2(&self) -> f64 {
    self.chi2
  }
  pub fn redchi2(&self) -> f64 {
    self.redchi2
  }
  pub fn convergence_message(&self) -> String {
    self.convergence_message.clone()
  }
}

#[wasm_bindgen]
pub fn fit(model_name: &str, p: Vec<f64>, x: Vec<f64>, y: Vec<f64>, sy: Vec<f64>) -> FitResult {
  let arr_p = Array1::from(p);
  let arr_x = Array1::from(x);
  let arr_y = Array1::from(y);
  let arr_sy = Array1::from(sy);

  let func = func1d::Func1D::new(&arr_p, &arr_x, get_function(model_name));
  let mut minimizer = curve_fit::Minimizer::init(&func, &arr_y, &arr_sy, 0.01);
  minimizer.minimize(10*arr_p.len());
  FitResult {
    parameters: array1_to_vec(minimizer.minimizer_parameters),
    parameter_std_errors: array1_to_vec(minimizer.parameter_errors),
    num_func_evaluation: minimizer.num_func_evaluation,
    chi2: minimizer.chi2,
    redchi2: minimizer.redchi2,
    convergence_message: String::from(minimizer.convergence_message)
  }
}