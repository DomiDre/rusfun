use ndarray::Array1;

pub fn gaussian(p: &Array1<f64>, x: &Array1<f64>) -> Array1<f64> {
    // Gaussian Size-Distribution
    // p = [A, mu, sig, c]
    // f(x) = A * exp(-0.5* ((x-mu)/sig)**2) + c
    x.map(|x| p[0] * (-0.5 * ((x - p[1]) / p[2]).powi(2)).exp() + p[3])
}
