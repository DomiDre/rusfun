use ndarray::Array1;

pub fn array1_to_vec(array: Array1<f64>) -> Vec<f64> {
    let mut result: Vec<f64> = Vec::new();
    for i in 0..array.len() {
        result.push(array[i]);
    }
    result
}