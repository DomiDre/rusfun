use ndarray::{Array1, array};

struct Func1D {
    parameters: Array1<f64>,
    domain: Array1<f64>,
    function: fn(&Array1<f64>, &Array1<f64>) -> Array1<f64>
}

impl Func1D {
    fn output(&self) -> Array1<f64> {
        (self.function)(&self.parameters, &self.domain)
    }
}

fn parabola(p: &Array1<f64>, x: &Array1<f64>) -> Array1<f64> {
    let mut result: Vec<f64> = Vec::new();
    for xval in x.iter() {
        result.push(p[0] * xval * xval + p[1] * xval + p[2]);
    }
    Array1::from(result)
}

fn main() {
    let parab = Func1D {
        parameters: Array1::from(vec![1.0, 0.0, 0.0]),
        domain: array![1.0, 2.0, 3.0, 4.0, 5.0],
        function: parabola
    };
    
    println!("{}", parab.output());
}
