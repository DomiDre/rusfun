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

pub fn lm<'a>(
	function: Residuum, // residuum fcn
  model: &'a Func1D, y: &Array1<f64>, sy: &Array1<f64>, // model and data
  ) -> Func1D<'a> {
  
  


  Func1D {
    domain: model.domain,
    parameters: model.parameters,
    function: model.function
  }
}
