use ndarray::{Array1, Array2, s};

pub struct Func1D<'a> {
    pub parameters: &'a Array1<f64>,
    pub domain: &'a Array1<f64>,
    pub function: fn(&Array1<f64>, &Array1<f64>) -> Array1<f64>
}

impl<'a> Func1D<'a> {
    pub fn new(
    parameters: &'a Array1<f64>,
    domain: &'a Array1<f64>,
    function: fn(&Array1<f64>, &Array1<f64>) -> Array1<f64>) -> Func1D<'a> {
        Func1D {
            parameters: &parameters,
            domain: &domain,
            function
        }
    }

    pub fn output(&self) -> Array1<f64> {
        (self.function)(&self.parameters, &self.domain)
    }

    pub fn for_parameters(&self, parameters: &Array1<f64>) -> Array1<f64> {
        (self.function)(&parameters, &self.domain)
    }

    pub fn parameter_gradient(&self, parameters: &Array1<f64>, func_values: &Array1<f64>) -> Array2<f64> {
        let epsilon = std::f64::EPSILON.sqrt();
        let mut jacobian: Array2<f64> = Array2::zeros((self.parameters.len(), self.domain.len()));

        // let func_values = self.for_parameters(&parameters);
        for i in 0..parameters.len() {
            let mut shifted_parameters = parameters.clone();
            let mut shift = epsilon*shifted_parameters[i].abs();
            if shift == 0.0 { shift = epsilon };
            shifted_parameters[i] += shift;
            let shifted_func_values = self.for_parameters(&shifted_parameters);
            let mut row_slice = jacobian.slice_mut(s![i, ..]);
            let derivative: Array1<f64> = (shifted_func_values - func_values.clone()) / shift;
            row_slice.assign(&derivative);
        }
        jacobian.reversed_axes()
    }
}