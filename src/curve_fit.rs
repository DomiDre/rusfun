use crate::func1d::Func1D;
use ndarray::Array1;
use crate::utils::matrix_solve;

pub fn std_residuum(model: &Func1D, parameters: &Array1<f64>,
y: &Array1<f64>, sy: &Array1<f64>) -> Array1<f64> {
	(y - &model.for_parameters(&parameters))/sy
}

pub fn chi2(residuum: &Array1<f64>) -> f64 {
	residuum.map(|x| x.powi(2)).sum()
}

pub struct MinimizationStep {
	parameters: Array1<f64>,
	residuum:  Array1<f64>,
	chi2: f64,
	metric: f64
}

pub struct Minimizer<'a> {
	pub model: &'a Func1D<'a>,
	pub y: &'a Array1<f64>,
	pub sy: &'a Array1<f64>,
	pub minimizer_parameters: Array1<f64>,
	pub lambda: f64,
	pub num_func_evaluation: usize,
	pub residuum: Array1<f64>,
	pub chi2: f64,
	pub epsilon1: f64,
	pub epsilon2: f64,
	pub epsilon3: f64,
	pub epsilon4: f64,
	pub lambda_UP_fac: f64,
	pub lambda_DOWN_fac: f64,
}

impl<'a> Minimizer<'a> {

	pub fn init<'b>(model: &'b Func1D, y: &'b Array1<f64>, sy: &'b Array1<f64>,
		lambda: f64) -> Minimizer<'b> {
		let initial_parameters = model.parameters.clone();
		let initial_residuum = std_residuum(&model, &initial_parameters, &y, &sy);
		let chi2 = chi2(&initial_residuum);
		Minimizer {
			model: &model,
			y: &y,
			sy: &sy,
			minimizer_parameters: initial_parameters,
			lambda: lambda,
			num_func_evaluation: 0,
			residuum: initial_residuum,
			chi2: chi2,
			epsilon1: 1e-3,
			epsilon2: 1e-3,
			epsilon3: 1e-1,
			epsilon4: 1e-1,
			lambda_UP_fac: 11.0,
			lambda_DOWN_fac: 9.0
		}
	}

	pub fn residuum(&self, parameters: &Array1<f64>) -> Array1<f64> {
		std_residuum(&self.model, &parameters, &self.y, &self.sy)
	}

	pub fn dof(&self) -> usize {
		let n_params = self.model.parameters.len();
		let n_points = self.model.domain.len();
		n_points - n_params
	}

	pub fn redchi2(&self, residuum: &Array1<f64>) -> f64 {
		let chi2 = chi2(residuum);
		let dof = self.dof();
		chi2 / (dof as f64)
	}

	pub fn lm(&mut self) -> MinimizationStep {
		//performs a Levenberg Marquardt step

		// determine change to parameters by solving the equation
		// [J^T W J + lambda diag(J^T W J)] delta = J^T W (y - f)
		// for delta

		// J is the parameter gradient of f at the current values
		let j = self.model.parameter_gradient(&self.minimizer_parameters);
		
		// println!("Parameter Gradient:\n{}", j);
		self.num_func_evaluation += self.model.parameters.len() + 1;

		// J^T is cloned to be multiplied by W later
		let mut jt = j.clone().reversed_axes();
		// calculate J^T W (y - f) (rhs), where W (y - f) is just the residuum
		
		let b = jt.dot(&self.residuum);

		// multiply J^T with W to obtain J^T W
		for i in 0..jt.cols() {
			let mut col = jt.column_mut(i);
			col *= 1.0/self.sy[i];
		}

		// calculate J^T W J + lambda*diag(J^T W J)  [lhs]
		// first J^T W J
		let mut A = jt.dot(&j);

		// lambda*diag(J^T W J) is needed for the calculation of the metric
		let mut gaussNewtonDiagonal: Array1<f64> = Array1::zeros(A.rows());
		for i in 0..A.rows() {
			gaussNewtonDiagonal[i] = self.lambda*A[[i, i]];
			// add to A to obtain lhs of LM step
			A[[i, i]] += gaussNewtonDiagonal[i];
		}

		let delta: Array1<f64> = matrix_solve(&A, &b);
		let mut metric = delta.dot(&b);
		for i in 0..A.rows() {
			metric +=  delta[i].powi(2)*gaussNewtonDiagonal[i];
		}

		let updated_parameters = self.minimizer_parameters.clone() + delta;
		let updated_residuum = self.residuum(&updated_parameters);
		let updated_chi2 = chi2(&updated_residuum);

		MinimizationStep {
			parameters: updated_parameters,
			residuum: updated_residuum,
			chi2: updated_chi2,
			metric: metric
		}
		
	}

	pub fn minimize(&mut self, max_iterations: u32) {
		let mut iterations = 0;

		while iterations < max_iterations {
			let update_step = self.lm();
			iterations += 1;
			// compare chi2 before and after
			let rho = (self.chi2 - update_step.chi2)/update_step.metric;

			if rho > self.epsilon4 {
				//new parameters are better
				self.lambda = (self.lambda/self.lambda_DOWN_fac).max(1e-7);
				// if update step is better, store new state
				self.minimizer_parameters = update_step.parameters; 
				self.residuum = update_step.residuum;
				self.chi2 = update_step.chi2;

			} else {
				// new chi2 not good enough, increasing lambda
				self.lambda = (self.lambda*self.lambda_UP_fac).min(1e7);
			}
		}
	}

	pub fn report(&self) {
		println!("\t #Func. Evaluations:\t{}", self.num_func_evaluation);
		println!("---- Parameters ----");
		for i in 0..self.minimizer_parameters.len() {
			println!("{}\t(init: {:?})", self.minimizer_parameters[i], self.model.parameters[i]);
		}
	}
}
