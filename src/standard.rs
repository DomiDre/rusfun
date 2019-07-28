use ndarray::Array1;

pub fn linear(p: &Array1<f64>, x: &Array1<f64>) -> Array1<f64> {
    x.map( |xval| p[0] * xval + p[1])
}

pub fn parabola(p: &Array1<f64>, x: &Array1<f64>) -> Array1<f64> {
    x.map( |xval| p[0] * xval * xval + p[1] * xval + p[2])
}
