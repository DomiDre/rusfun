#![allow(dead_code)]

use gauss_quad::GaussHermite;
use ndarray::Array1;
use std::f64::consts::{FRAC_2_SQRT_PI, SQRT_2, PI};

/// Formfactor Amplitude F of a Spherical Particle
///
/// See for example Jan Skov Pedersen, Advances in Colloid and Interface Science 1997, 70, 171. doi: 10.1016/S0001-8686(97)00312-6
fn amplitude(q: f64, R: f64) -> f64 {
    let qR = q * R;
    3.0 * (qR.sin() - qR * qR.cos()) / qR.powi(3)
}

/// Size distribution integral.
/// 
/// The problem is trivially mapped on an integral over exp(-x^2) by a variable
/// transformation, which is solved by a Gauss-Hermite quadrature
fn size_distributed_formfactor(q: f64, R: f64, sigR: f64, gh_quad: &GaussHermite) -> f64 {
    let integral = gh_quad.integrate(|r_value| amplitude(q, R * (SQRT_2 * r_value * sigR).exp()).powi(2));
    integral * 0.5 * FRAC_2_SQRT_PI
}

/// Formfactor of a spherical particle
/// 
/// P = N/V * V_p^2 * DeltaSLD^2 * F^2 
/// Additionally a size distribution average is performed
pub fn formfactor(p: &Array1<f64>, q: &Array1<f64>) -> Array1<f64> {
    let R = p[0];
    let sigR = p[1];
    let SLDsphere = p[2];
    let SLDmatrix = p[3];
    let I0 = p[4];
    let deg = p[5] as usize;

    let V = 4.0/3.0*PI*R.powi(3);
    let quad = GaussHermite::init(deg);

    let I = q.map(|qval| size_distributed_formfactor(*qval, R, sigR, &quad));
    I0*((SLDsphere - SLDmatrix)*V).powi(2) * I
}
