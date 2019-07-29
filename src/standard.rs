use ndarray::Array1;

use crate::utils::array1_to_vec;

extern crate wasm_bindgen;
use wasm_bindgen::prelude::*;

pub fn linear(p: &Array1<f64>, x: &Array1<f64>) -> Array1<f64> {
    x.map( |xval| p[0] * xval + p[1])
}

#[wasm_bindgen]
pub fn wasm_linear(p: Vec<f64>, x: Vec<f64>) -> Vec<f64> {
    array1_to_vec(linear(&Array1::from(p), &Array1::from(x)))
}

// #[wasm_bindgen]
pub fn parabola(p: &Array1<f64>, x: &Array1<f64>) -> Array1<f64> {
    x.map( |xval| p[0] * xval * xval + p[1] * xval + p[2])
}

#[wasm_bindgen]
pub fn wasm_parabola(p: Vec<f64>, x: Vec<f64>) -> Vec<f64> {
    array1_to_vec(parabola(&Array1::from(p), &Array1::from(x)))
}
