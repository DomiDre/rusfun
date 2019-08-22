use ndarray::Array1;

/// Zero function
/// p = []
/// f(x) = 0
pub fn zero(_p: &Array1<f64>, x: &Array1<f64>) -> Array1<f64> {
    x.map(|_xval| 0.0)
}


/// Linear function
/// p = [a, b]
/// f(x) = a*x + b
pub fn linear(p: &Array1<f64>, x: &Array1<f64>) -> Array1<f64> {
    x.map(|xval| p[0] * xval + p[1])
}


/// Quadratic function
/// p = [a, b, c]
/// f(x) = a*x^2 + b*x + c
pub fn parabola(p: &Array1<f64>, x: &Array1<f64>) -> Array1<f64> {
    x.map(|xval| p[0] * xval * xval + p[1] * xval + p[2])
}

/// Square-root function
/// p = [A, b, c]
/// f(x) = A*sqrt(b*x - c)
pub fn sqrt(p: &Array1<f64>, x: &Array1<f64>) -> Array1<f64> {
    x.map(|xval| p[0] * (p[1]*xval - p[2]).sqrt())
}


/// Cosine function
/// p = [A, b, c]
/// f(x) = A*cos(b*x - c)
pub fn cos(p: &Array1<f64>, x: &Array1<f64>) -> Array1<f64> {
    x.map(|xval| p[0] * (p[1] * xval - p[2]).cos())
}

/// Sine function
/// p = [A, b, c]
/// f(x) = A*sin(b*x - c)
pub fn sin(p: &Array1<f64>, x: &Array1<f64>) -> Array1<f64> {
    x.map(|xval| p[0] * (p[1] * xval - p[2]).sin())
}

/// Tan function
/// p = [A, b, c]
/// f(x) = A*tan(b*x - c)
pub fn tan(p: &Array1<f64>, x: &Array1<f64>) -> Array1<f64> {
    x.map(|xval| p[0] * (p[1] * xval - p[2]).tan())
}

/// Exponential function
/// p = [A, b]
/// f(x) = A*exp(b*x)
pub fn exp(p: &Array1<f64>, x: &Array1<f64>) -> Array1<f64> {
    x.map(|xval| p[0] * (p[1] * xval).exp())
}
