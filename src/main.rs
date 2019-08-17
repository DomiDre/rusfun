#![allow(dead_code)]
#![allow(non_snake_case)]
mod curve_fit;
mod func1d;
mod sas;
mod size_distribution;
mod standard;
mod utils;

use ndarray::{array, Array1};
use utils::read_column_file;

fn main() {
    // To do: Pass filepath & initial parameters from command line

    // define the model
    // let p = array![1.0, 1.0];
    // let p = array![4.0, 1.0, 1.0];
    // let p = array![2.5, 3.5, 1.0];
    let p = array![50.0, 0.1, 40e-6, 10e-6, 1.0, 20.0];
    let vary_p = array![true, true, false, false, true, false];

    // read data
    // let (x, y, sy) = read_column_file("./linearData.xye").unwrap();
    // let (x, y, sy) = read_column_file("./parabolaData.xye").unwrap();
    // let (x, y, sy) = read_column_file("./cosData.xye").unwrap();
    // let (x, y, sy) = read_column_file("./sample_data/gaussianData.xye").unwrap();
    let (x, y, sy) = read_column_file("./sample_data/AH11.xye").unwrap();

    let x = Array1::from_vec(x);
    let y = Array1::from_vec(y);
    let sy = Array1::from_vec(sy);

    let model = func1d::Func1D::new(&p, &x, sas::sphere::formfactor);

    // fit data
    let t0 = std::time::Instant::now();
    let mut minimizer = curve_fit::Minimizer::init(&model, &y, &sy, &vary_p, 0.01);
    minimizer.minimize();
    println!("Execution time: {} microsecs", t0.elapsed().as_micros());
    minimizer.report();
}
