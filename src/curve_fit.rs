use crate::func1d::Func1D;
use ndarray::Array1;

pub struct Residuum<'a> {
	pub model: &'a Func1D<'a>,
	pub y: &'a Array1<f64>,
	pub sy: &'a Array1<f64>,
	pub function: fn(resi: &Residuum) -> Array1<f64>
}

impl<'a> Residuum<'a> {
	pub fn output(&self) -> Array1<f64> {
		(self.function)(&self)
	}

	pub fn chi2(&self) -> f64 {
		self.output().map(|x| x.powi(2)).sum()
	}

	pub fn dof(&self) -> usize {
		let n_params = self.model.parameters.len();
		let n_points = self.model.domain.len();
		n_points - n_params
	}

	pub fn redchi2(&self) -> f64 {
		let chi2 = self.chi2();
		let dof = self.dof();
		chi2 / (dof as f64)
	}
}


pub fn std_residuum(resi: &Residuum) -> Array1<f64> {
	(resi.y - &resi.model.output())/resi.sy
}

pub struct Minimizer<'a> {
	pub residuum: Residuum<'a>,
	pub num_func_evaluation: usize,
}

impl<'a> Minimizer<'a> {
	pub fn lm(&mut self) -> Array1<f64> {
		
		let j = self.residuum.model.parameter_gradient();
		let jt = j.t();
		let A = j.dot(&jt);
		let b = jt.dot(&self.residuum.output());

		println!("{:?}", j);
		self.num_func_evaluation += self.residuum.model.parameters.len();
		//JT

		self.residuum.model.domain.clone()
	
	}
}
