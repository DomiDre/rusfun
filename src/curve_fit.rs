use crate::func1d::Func1D;
use crate::utils::{matrix_solve, LU_decomp, LU_matrix_solve};
use ndarray::{s, Array1, Array2};

pub fn chi2(y: &Array1<f64>, ymodel: &Array1<f64>, sy: &Array1<f64>) -> f64 {
    ((y - ymodel) / sy).map(|x| x.powi(2)).sum()
}

pub struct MinimizationStep {
    parameters: Array1<f64>,
    delta: Array1<f64>,
    ymodel: Array1<f64>,
    chi2: f64,
    redchi2: f64,
    metric: f64,
    metric_gradient: f64,
    metric_parameters: f64,
    JT_W_J: Array2<f64>,
}

pub struct Minimizer<'a> {
    pub model: &'a Func1D<'a>,
    pub y: &'a Array1<f64>,
    pub sy: &'a Array1<f64>,
    pub weighting_matrix: Array1<f64>,
    pub minimizer_parameters: Array1<f64>,
    pub minimizer_ymodel: Array1<f64>,
    pub jacobian: Array2<f64>,
    pub parameter_cov_matrix: Array2<f64>,
    pub parameter_errors: Array1<f64>,
    pub lambda: f64,
    pub num_func_evaluation: usize,
    pub num_params: usize,
    pub num_data: usize,
    pub chi2: f64,
    pub dof: usize,
    pub redchi2: f64,
    pub convergence_message: &'a str,
    pub epsilon1: f64,
    pub epsilon2: f64,
    pub epsilon3: f64,
    pub epsilon4: f64,
    pub lambda_UP_fac: f64,
    pub lambda_DOWN_fac: f64,
}

impl<'a> Minimizer<'a> {
    pub fn init<'b>(
        model: &'b Func1D,
        y: &'b Array1<f64>,
        sy: &'b Array1<f64>,
        lambda: f64,
    ) -> Minimizer<'b> {
        // at initialization
        let initial_parameters = model.parameters.clone();
        let minimizer_ymodel = model.for_parameters(&initial_parameters);
        let num_params = initial_parameters.len();
        let num_data = model.domain.len();
        let chi2 = chi2(&y, &minimizer_ymodel, &sy);
        let dof = num_data - num_params;
        let redchi2 = chi2 / (dof as f64);

        // initialize jacobian
        // J is the parameter gradient of f at the current values
        let j = model.parameter_gradient(&initial_parameters, &minimizer_ymodel);

        // W = 1 / sy^2, only diagonal is considered
        let weighting_matrix: Array1<f64> = sy.map(|x| 1.0 / x.powi(2));

        Minimizer {
            model: &model,
            y: &y,
            sy: &sy,
            weighting_matrix: weighting_matrix,
            minimizer_parameters: initial_parameters,
            minimizer_ymodel: minimizer_ymodel,
            jacobian: j,
            parameter_cov_matrix: Array2::zeros((num_params, num_params)),
            parameter_errors: Array1::zeros(num_params),
            lambda: lambda,
            num_func_evaluation: 0,
            num_data: num_data,
            num_params: num_params,
            chi2: chi2,
            dof: dof,
            redchi2: redchi2,
            convergence_message: "",
            epsilon1: 1e-3,
            epsilon2: 1e-3,
            epsilon3: 1e-1,
            epsilon4: 1e-1,
            lambda_UP_fac: 11.0,
            lambda_DOWN_fac: 9.0,
        }
    }

    pub fn lm(&mut self) -> MinimizationStep {
        //performs a Levenberg Marquardt step

        // determine change to parameters by solving the equation
        // [J^T W J + lambda diag(J^T W J)] delta = J^T W (y - f)
        // for delta

        // J^T is cloned to be multiplied by weighting_matrix later
        let mut jt = self.jacobian.clone().reversed_axes();

        // multiply J^T with W to obtain J^T W
        for i in 0..self.num_data {
            let mut col = jt.column_mut(i);
            col *= self.weighting_matrix[i];
        }

        // calculate J^T W (y - f) (rhs of LM step)
        let b = jt.dot(&(self.y - &self.minimizer_ymodel));

        // calculate J^T W J + lambda*diag(J^T W J)  [lhs of LM step]
        // first J^T W J
        let JT_W_J = jt.dot(&self.jacobian);

        let lambdaDiagJT_W_J = self.lambda * &JT_W_J.diag();
        let mut A = JT_W_J.clone();
        for i in 0..self.num_params {
            A[[i, i]] += lambdaDiagJT_W_J[i];
        }

        // solve LM step for delta
        let delta: Array1<f64> = matrix_solve(&A, &b);

        // calculate metrics to determine convergence
        let mut metric = delta.dot(&b);

        for i in 0..self.num_params {
            metric += delta[i].powi(2) * lambdaDiagJT_W_J[i];
        }

        // take maximum of the absolute value in the respective arrays as metric for the
        // convergence of either the gradient or the parameters
        let metric_gradient = b
            .map(|x| x.abs())
            .to_vec()
            .iter()
            .cloned()
            .fold(0. / 0., f64::max);
        let metric_parameters = (&delta / &self.minimizer_parameters)
            .map(|x| x.abs())
            .to_vec()
            .iter()
            .cloned()
            .fold(0. / 0., f64::max);;

        let updated_parameters = &self.minimizer_parameters + &delta;
        let updated_model = self.model.for_parameters(&updated_parameters);
        let updated_chi2 = chi2(&self.y, &updated_model, &self.sy);
        let redchi2 = updated_chi2 / (self.dof as f64);

        MinimizationStep {
            parameters: updated_parameters,
            delta: delta,
            ymodel: updated_model,
            chi2: updated_chi2,
            redchi2: redchi2,
            metric: metric,
            metric_gradient: metric_gradient,
            metric_parameters: metric_parameters,
            JT_W_J: JT_W_J,
        }
    }

    pub fn minimize(&mut self, max_iterations: usize) {
        let mut iterations = 0;
        let inverse_parameter_cov_matrix: Array2<f64>;

        loop {
            let update_step = self.lm();
            iterations += 1;

            // compare chi2 before and after with respect to metric to decide if step is accepted
            let rho = (self.chi2 - update_step.chi2) / update_step.metric;

            if rho > self.epsilon4 {
                //new parameters are better, update lambda
                self.lambda = (self.lambda / self.lambda_DOWN_fac).max(1e-7);

                // update jacobian
                if iterations % 2 * self.num_params == 0 {
                    // at every 2*n steps update jacobian by explicit calculation
                    // requires #params function evaluations
                    self.jacobian = self
                        .model
                        .parameter_gradient(&self.minimizer_parameters, &self.minimizer_ymodel);
                    self.num_func_evaluation += self.num_params;
                } else {
                    // otherwise update jacobian with Broyden rank-1 update formula
                    let norm_delta = update_step.delta.dot(&update_step.delta);
                    let diff = &update_step.ymodel
                        - &self.minimizer_ymodel
                        - self.jacobian.dot(&update_step.delta);
                    let mut jacobian_change: Array2<f64> =
                        Array2::zeros((self.num_data, self.num_params));

                    for i in 0..self.num_params {
                        let mut col_slice = jacobian_change.slice_mut(s![.., i]);
                        col_slice.assign(&(&diff * update_step.delta[i] / norm_delta));
                    }

                    self.jacobian = &self.jacobian + &jacobian_change;
                }

                // store new state in Minimizer
                self.minimizer_parameters = update_step.parameters;
                self.minimizer_ymodel = update_step.ymodel;
                self.chi2 = update_step.chi2;
                self.redchi2 = update_step.redchi2;

                // check convergence criteria
                // gradient converged
                if update_step.metric_gradient < self.epsilon1 {
                    self.convergence_message = "Gradient converged";
                    inverse_parameter_cov_matrix = update_step.JT_W_J;
                    break;
                };

                // parameters converged
                if update_step.metric_parameters < self.epsilon2 {
                    self.convergence_message = "Parameters converged";
                    inverse_parameter_cov_matrix = update_step.JT_W_J;
                    break;
                };

                // chi2 converged
                if update_step.redchi2 < self.epsilon3 {
                    self.convergence_message = "Chi2 converged";
                    inverse_parameter_cov_matrix = update_step.JT_W_J;
                    break;
                };
                if iterations >= max_iterations {
                    self.convergence_message = "Reached max. number of iterations";
                    inverse_parameter_cov_matrix = update_step.JT_W_J;
                    break;
                }
            } else {
                // new chi2 not good enough, increasing lambda
                self.lambda = (self.lambda * self.lambda_UP_fac).min(1e7);
                // step is rejected, update jacobian by explicit calculation
                self.jacobian = self
                    .model
                    .parameter_gradient(&self.minimizer_parameters, &self.minimizer_ymodel);
            }
        }

        // calculate parameter covariance matrix using the LU decomposition
        let (L, U, P) = LU_decomp(&inverse_parameter_cov_matrix);
        for i in 0..self.num_params {
            let mut unit_vector = Array1::zeros(self.num_params);
            unit_vector[i] = 1.0;
            let mut col_slice = self.parameter_cov_matrix.slice_mut(s![.., i]);
            col_slice.assign(&LU_matrix_solve(&L, &U, &P, &unit_vector));
        }
        // parameter fit errors are the sqrt of the diagonal
        self.parameter_errors = self.parameter_cov_matrix.diag().map(|x| x.sqrt());
    }

    pub fn report(&self) {
        println!("\t #Chi2:\t{:.6}", self.chi2);
        println!("\t #Red. Chi2:\t{:.6}", self.redchi2);
        println!("\t #Func. Evaluations:\t{}", self.num_func_evaluation);
        println!("\t #Converged by:\t{}", self.convergence_message);
        println!("---- Parameters ----");
        for i in 0..self.minimizer_parameters.len() {
            println!(
                "{:.8} +/- {:.8} ({:.2} %)\t(init: {})",
                self.minimizer_parameters[i],
                self.parameter_errors[i],
                (self.parameter_errors[i] / self.minimizer_parameters[i]).abs() * 100.0,
                self.model.parameters[i]
            );
        }
    }
}
