use ndarray::Array1;

mod func1d;
pub use crate::func1d::Func1D;

pub fn parabola(p: &Array1<f64>, x: &Array1<f64>) -> Array1<f64> {
    let mut result: Vec<f64> = Vec::new();
    for xval in x.iter() {
        result.push(p[0] * xval * xval + p[1] * xval + p[2]);
    }
    Array1::from(result)
}
