use crate::utils::array1_to_vec;
use crate::standard;
extern crate wasm_bindgen;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn linear(p: Vec<f64>, x: Vec<f64>) -> Vec<f64> {
    array1_to_vec(standard::linear(&Array1::from(p), &Array1::from(x)))
}

#[wasm_bindgen]
pub fn parabola(p: Vec<f64>, x: Vec<f64>) -> Vec<f64> {
    array1_to_vec(standard::parabola(&Array1::from(p), &Array1::from(x)))
}
