use rusfun::{parabola, Func1D};

fn main() {
    let p = vec![1.0, 0.0, 0.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let parab = Func1D::new(p, x, parabola);
    println!("{}", parab.output());
}
