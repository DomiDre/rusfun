#[macro_use]
extern crate criterion;

use criterion::Criterion;
use ndarray::{array, Array1};

fn calc_with_array() -> f64 {
    let mut a: [f64; 100] = [0.0; 100];
    let mut b: [f64; 100] = [0.0; 100];
    for i in 0..100 {
        a[i] = i as f64;
        b[i] = 2.0*(i as f64);
    }
    a.iter().zip(b.iter()).map(|(x, y)| x*y).sum()
}

fn calc_with_vec() -> f64 {
    let mut a: Vec<f64> = vec![0.0; 100];
    let mut b: Vec<f64> = vec![0.0; 100];
    for i in 0..100 {
        a[i] = i as f64;
        b[i] = 2.0*(i as f64);
    }
    a.iter().zip(b.iter()).map(|(x, y)| x*y).sum()
}

fn calc_with_array1() -> f64 {
    let a: Array1<f64> = Array1::range(0.0, 100.0, 1.0);
    let b: Array1<f64> = Array1::range(0.0, 200.0, 2.0);
    a.dot(&b)
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("calc with array", |b| b.iter(|| calc_with_array()));
    c.bench_function("calc with vec", |b| b.iter(|| calc_with_vec()));
    c.bench_function("calc with array1", |b| b.iter(|| calc_with_array1()));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);