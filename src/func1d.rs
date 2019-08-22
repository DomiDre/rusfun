use ndarray::{s, Array1, Array2};

/// Contains reference to initial parameters p, the domain x and the function that maps both to f(p, x) = y
pub struct Func1D<'a> {
    pub parameters: &'a Array1<f64>,
    pub domain: &'a Array1<f64>,
    pub function: fn(&Array1<f64>, &Array1<f64>) -> Array1<f64>,
}

impl<'a> Func1D<'a> {
    /// Initialize a Func1D by a reference to the parameters p, the domain x and a function to maps fn(p, x) -> y
    pub fn new(
        parameters: &'a Array1<f64>,
        domain: &'a Array1<f64>,
        function: fn(&Array1<f64>, &Array1<f64>) -> Array1<f64>,
    ) -> Func1D<'a> {
        Func1D {
            parameters: &parameters,
            domain: &domain,
            function,
        }
    }

    /// Performs calculation of f(p, x) using the initial parameters p
    pub fn output(&self) -> Array1<f64> {
        (self.function)(&self.parameters, &self.domain)
    }

    /// Performs calculation of f(p, x) using the given set of parameters
    pub fn for_parameters(&self, parameters: &Array1<f64>) -> Array1<f64> {
        (self.function)(&parameters, &self.domain)
    }

    /// Calculates the gradient of the function with respect to its parameters
    pub fn parameter_gradient(
        &self,
        parameters: &Array1<f64>,         // parameter values of the model
        include_parameter: &Array1<bool>, // which parameters to evaluate
        func_values: &Array1<f64>,        // function values for given parameters
    ) -> Array2<f64> {
        let epsilon = std::f64::EPSILON.sqrt();

        // calculate number of parameters that are being varied
        let num_varying_params = include_parameter
            .iter()
            .fold(0, |sum, val| if *val { sum + 1 } else { sum });

        // initialize the jacobian as zero matrix Np x Nx
        let mut jacobian: Array2<f64> = Array2::zeros((num_varying_params, self.domain.len()));

        let mut idx_param = 0;
        for i in 0..parameters.len() {
            if include_parameter[i] {
                // shift parameter by a small value to evaluate derivative
                let mut shifted_parameters = parameters.clone();
                let mut shift = epsilon * shifted_parameters[i].abs();
                if shift == 0.0 {
                    shift = epsilon
                };
                shifted_parameters[i] += shift;

                // calculate function values for shifted parameter f(x; p + delta)
                let shifted_func_values = self.for_parameters(&shifted_parameters);
                // derivative is eval as [f(x; p + delta) - f(x; p)]/delta
                let derivative: Array1<f64> = (shifted_func_values - func_values.clone()) / shift;

                // set derivative to row of jacobian
                let mut row_slice = jacobian.slice_mut(s![idx_param, ..]);
                row_slice.assign(&derivative);
                idx_param += 1;
            }
        }

        // return jacobian with derivatives on the columns
        jacobian.reversed_axes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::standard::parabola;
    use ndarray::array;
    #[test]
    fn calculate_parabola() {
        let p = array![2.0, 1.0, 0.5];
        let x: Array1<f64> = Array1::range(0.0, 10.0, 1.0);
        let y: Array1<f64> = x.map(|x| 2.0 * x.powi(2) + 1.0 * x + 0.5);
        let parab = Func1D::new(&p, &x, parabola);
        // let sy: Array1<f64> = x.map(|x| 1.0+4.0*x.abs());
        assert_eq!(y, parab.output());
    }
}
