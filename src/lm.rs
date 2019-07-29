use ndarray::{Array1};
use crate::func1d::Func1D;

pub fn residuum(y: &Array1<f64>, sy: &Array1<f64>, model: &Func1D) -> Array1<f64> {
    (y - &model.output())/sy
}
