use ndarray::Array1;

pub fn zero(p: &Array1<f64>, x: &Array1<f64>) -> Array1<f64> {
  x.map( |xval| 0.0)
}

pub fn linear(p: &Array1<f64>, x: &Array1<f64>) -> Array1<f64> {
  x.map( |xval| p[0] * xval + p[1])
}

pub fn parabola(p: &Array1<f64>, x: &Array1<f64>) -> Array1<f64> {
  x.map( |xval| p[0] * xval * xval + p[1] * xval + p[2])
}

pub fn sqrt(p: &Array1<f64>, x: &Array1<f64>) -> Array1<f64> {
  x.map( |xval| p[0] * xval.sqrt())
}

pub fn cos(p: &Array1<f64>, x: &Array1<f64>) -> Array1<f64> {
  x.map( |xval| p[0] * (p[1]*xval - p[2]).cos())
}

pub fn sin(p: &Array1<f64>, x: &Array1<f64>) -> Array1<f64> {
  x.map( |xval| p[0] * (p[1]*xval - p[2]).sin())
}

pub fn tan(p: &Array1<f64>, x: &Array1<f64>) -> Array1<f64> {
  x.map( |xval| p[0] * (p[1]*xval-p[2]).tan())
}

pub fn exp(p: &Array1<f64>, x: &Array1<f64>) -> Array1<f64> {
  x.map( |xval| p[0] * (p[1]*xval).exp())
}

