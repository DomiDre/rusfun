use rusfun::{parabola, Func1D, Residuum, std_residuum};
use ndarray::{array, Array1};

fn main() {
    let p = array![1.0, 0.0, 0.0];
    let x: Array1<f64> = Array1::range(0.0, 100.0, 1.0);
    let y: Array1<f64> = x.map(|x| x.powi(2));
    let sy: Array1<f64> = 1.5*Array1::ones(100);
    let parab = Func1D::new(&p, &x, parabola);

    let residuum = Residuum {
        model: &parab,
        y: &y,
        sy: &sy,
        function: std_residuum
    };
    // println!("{:?}", parab.output());
    // println!("{:?}", residuum.redchi2());
    println!("{:?}", parab.parameter_gradient());
}
