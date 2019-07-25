use ndarray::Array1;

pub struct Func1D {
    pub parameters: Array1<f64>,
    pub domain: Array1<f64>,
    pub function: fn(&Array1<f64>, &Array1<f64>) -> Array1<f64>
}

impl Func1D {
    pub fn new(parameters: Vec<f64>, domain: Vec<f64>, function: fn(&Array1<f64>, &Array1<f64>) -> Array1<f64>) -> Func1D {
        Func1D {
            parameters: Array1::from(parameters),
            domain: Array1::from(domain),
            function
        }
    }

    pub fn output(&self) -> Array1<f64> {
        (self.function)(&self.parameters, &self.domain)
    }
}