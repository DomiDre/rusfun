#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_variables)]
mod curve_fit;
mod func1d;
mod standard;
mod utils;

pub use crate::curve_fit::Minimizer;
pub use crate::func1d::Func1D;
pub use crate::standard::{cos, linear, parabola};
pub use crate::utils::matrix_solve;

use ndarray::{array, Array1};
use std::fs::File;
use std::io::{BufRead, BufReader, Result};

fn main() {
    // define the model
    // let p = array![1.0, 1.0];
    let p = array![4.0, 1.0, 1.0];
    // let p = array![2.5, 3.5, 1.0];

    // read data
    // let (x, y, sy) = read_column_file("./examples/linearData.xye").unwrap();
    let (x, y, sy) = read_column_file("./examples/parabolaData.xye").unwrap();
    // let (x, y, sy) = read_column_file("./examples/cosData.xye").unwrap();

    let x = Array1::from_vec(x);
    let y = Array1::from_vec(y);
    let sy = Array1::from_vec(sy);

    let model = Func1D::new(&p, &x, parabola);

    // fit data
    let mut minimizer = Minimizer::init(&model, &y, &sy, 1.0);
    let t0 = std::time::Instant::now();
    minimizer.minimize(10 * p.len());
    println!("Execution time: {} microsecs", t0.elapsed().as_micros());
    minimizer.report();
}

fn read_column_file(filename: &str) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let mut x: Vec<f64> = Vec::new();
    let mut y: Vec<f64> = Vec::new();
    let mut sy: Vec<f64> = Vec::new();

    let reader = BufReader::new(File::open(filename).expect("Cannot open file"));

    for line in reader.lines() {
        let unwrapped_line = line.unwrap();
        let splitted_line = unwrapped_line.split_whitespace();
        for (i, number) in splitted_line.enumerate() {
            match i {
                0 => x.push(number.parse().unwrap()),
                1 => y.push(number.parse().unwrap()),
                2 => sy.push(number.parse().unwrap()),
                _ => {}
            }
        }
    }
    Ok((x, y, sy))
}
