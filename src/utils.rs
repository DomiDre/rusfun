use ndarray::{s, Array, Array1, Array2, Axis, NdFloat, Zip};

use std::fs::File;
use std::io::{BufRead, BufReader, Result};

/// Transforms a ndarray Array1 to a vec
pub fn array1_to_vec<T>(array: Array1<T>) -> Vec<T>
where
    T: NdFloat,
{
    let mut result: Vec<T> = Vec::new();
    for i in 0..array.len() {
        result.push(array[i]);
    }
    result
}

/// Solves linear equation A*x = b by LU decomposition
pub fn matrix_solve<T>(A: &Array2<T>, b: &Array1<T>) -> Array1<T>
where
    T: NdFloat,
{
    let matrix_dimension = A.nrows();
    // solve Ax = b for x, where A is a square matrix
    let (L, U, P) = LU_decomp(A);
    // first solve Ly = Pb
    let pivotized_b = P.dot(b);
    let mut y: Array1<T> = pivotized_b.clone();
    for i in 1..matrix_dimension {
        y[i] = y[i] - L.slice(s![i, 0..i]).dot(&y.slice(s![0..i]));
    }
    // then solve Ux = y
    let mut x: Array1<T> = y.clone();
    x[matrix_dimension - 1] =
        x[matrix_dimension - 1] / U[[matrix_dimension - 1, matrix_dimension - 1]];
    for i in (0..matrix_dimension - 1).rev() {
        x[i] = (x[i]
            - U.slice(s![i, i + 1..matrix_dimension])
                .dot(&x.slice(s![i + 1..matrix_dimension])))
            / U[[i, i]];
    }
    x
}

/// Solves linear equation A*x = b where the partial pivoted LU decomposition of PA = LU is given
pub fn LU_matrix_solve<T>(L: &Array2<T>, U: &Array2<T>, P: &Array2<T>, b: &Array1<T>) -> Array1<T>
where
    T: NdFloat,
{
    let matrix_dimension = L.nrows();
    // first solve Ly = Pb
    let pivotized_b = P.dot(b);
    let mut y: Array1<T> = pivotized_b.clone();
    for i in 1..matrix_dimension {
        y[i] = y[i] - L.slice(s![i, 0..i]).dot(&y.slice(s![0..i]));
    }
    // then solve Ux = y
    let mut x: Array1<T> = y.clone();
    x[matrix_dimension - 1] =
        x[matrix_dimension - 1] / U[[matrix_dimension - 1, matrix_dimension - 1]];
    for i in (0..matrix_dimension - 1).rev() {
        x[i] = (x[i]
            - U.slice(s![i, i + 1..matrix_dimension])
                .dot(&x.slice(s![i + 1..matrix_dimension])))
            / U[[i, i]];
    }
    x
}

/// Performs partial pivoted LU decomposition of A such that P A = LU
/// with L a lower triangular matrix and U an upper triangular matrix
/// A needs to be a square matrix
pub fn LU_decomp<T>(A: &Array2<T>) -> (Array2<T>, Array2<T>, Array2<T>)
where
    T: NdFloat,
{
    let matrix_dimension = A.nrows();
    assert_eq!(
        matrix_dimension,
        A.ncols(),
        "Tried LU decomposition with a non-square matrix."
    );
    let P = pivot(&A);
    let pivotized_A = P.dot(A);

    let mut L: Array2<T> = Array::eye(matrix_dimension);
    let mut U: Array2<T> = Array::zeros((matrix_dimension, matrix_dimension));
    for idx_col in 0..matrix_dimension {
        // fill U
        for idx_row in 0..idx_col + 1 {
            U[[idx_row, idx_col]] = pivotized_A[[idx_row, idx_col]]
                - U.slice(s![0..idx_row, idx_col])
                    .dot(&L.slice(s![idx_row, 0..idx_row]));
        }
        // fill L
        for idx_row in idx_col + 1..matrix_dimension {
            L[[idx_row, idx_col]] = (pivotized_A[[idx_row, idx_col]]
                - U.slice(s![0..idx_col, idx_col])
                    .dot(&L.slice(s![idx_row, 0..idx_col])))
                / U[[idx_col, idx_col]];
        }
    }
    (L, U, P)
}

/// Pivot matrix A
fn pivot<T>(A: &Array2<T>) -> Array2<T>
where
    T: NdFloat,
{
    let matrix_dimension = A.nrows();
    let mut P: Array2<T> = Array::eye(matrix_dimension);
    for (i, column) in A.axis_iter(Axis(1)).enumerate() {
        // find idx of maximum value in column i
        let mut max_pos = i;
        for j in i..matrix_dimension {
            if column[max_pos].abs() < column[j].abs() {
                max_pos = j;
            }
        }
        // swap rows of P if necessary
        if max_pos != i {
            swap_rows(&mut P, i, max_pos);
        }
    }
    P
}

/// Swaps two rows of a matrix
fn swap_rows<T>(A: &mut Array2<T>, idx_row1: usize, idx_row2: usize)
where
    T: NdFloat,
{
    // to swap rows, get two ArrayViewMuts for the corresponding rows
    // and apply swap elementwise using ndarray::Zip
    let (.., mut matrix_rest) = A.view_mut().split_at(Axis(0), idx_row1);
    let (row0, mut matrix_rest) = matrix_rest.view_mut().split_at(Axis(0), 1);
    let (_matrix_helper, mut matrix_rest) = matrix_rest
        .view_mut()
        .split_at(Axis(0), idx_row2 - idx_row1 - 1);
    let (row1, ..) = matrix_rest.view_mut().split_at(Axis(0), 1);
    Zip::from(row0).and(row1).apply(std::mem::swap);
}

/// Reads three-column file, typically used to read x, y, σ data
pub fn read_column_file(filename: &str) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let mut x: Vec<f64> = Vec::new();
    let mut y: Vec<f64> = Vec::new();
    let mut sy: Vec<f64> = Vec::new();

    let reader = BufReader::new(File::open(filename).expect("Cannot open file"));

    for line in reader.lines() {
        let unwrapped_line = line.unwrap();
        if unwrapped_line.starts_with('#') {
            continue;
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};

    #[test]
    fn solve_system_of_linear_equations() {
        let A: Array2<f64> = array![[1.0, 3.0, 5.0], [2.0, 4.0, 7.0], [1.0, 1.0, 0.0],];
        let b: Array1<f64> = array![1.0, 2.0, 3.0];
        let x = matrix_solve(&A, &b);
        assert_eq!(A.dot(&x), b);
    }
}
