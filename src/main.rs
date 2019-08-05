#![allow(dead_code)]
#![allow(non_snake_case)]
mod curve_fit;
mod func1d;
mod standard;
mod utils;
mod size_distribution;

use ndarray::{array, Array1};
use std::fs::File;
use std::io::{BufRead, BufReader, Result};

fn main() {
    // To do: Pass filepath & initial parameters from command line

    // define the model
    // let p = array![1.0, 1.0];
    // let p = array![4.0, 1.0, 1.0];
    // let p = array![2.5, 3.5, 1.0];
    let p = array![300.0, 3.0, 0.2, 0.0];

    // read data
    // let (x, y, sy) = read_column_file("./linearData.xye").unwrap();
    // let (x, y, sy) = read_column_file("./parabolaData.xye").unwrap();
    // let (x, y, sy) = read_column_file("./cosData.xye").unwrap();
    let (x, y, sy) = read_column_file("./sample_data/gaussianData.xye").unwrap();

    let x = Array1::from_vec(x);
    let y = Array1::from_vec(y);
    let sy = Array1::from_vec(sy);

    let model = func1d::Func1D::new(&p, &x, size_distribution::gaussian);

    // fit data
    let mut minimizer = curve_fit::Minimizer::init(&model, &y, &sy, 1.0);
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
