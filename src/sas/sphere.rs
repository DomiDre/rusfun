use gauss_quad::GaussHermite;
use ndarray::Array1;

const PI: f64 = 3.1415926535897931;
const SQ_2: f64 = 1.4142135623730951;
const FRAC_SQ_PI: f64 = 0.56418958354775628;

/// Formfactor Amplitude F of a Spherical Particle
///
/// See for example Jan Skov Pedersen, Advances in Colloid and Interface Science 1997, 70, 171. doi: 10.1016/S0001-8686(97)00312-6
fn amplitude(q: f64, R: f64) -> f64 {
    let qR = q * R;
    if qR != 0.0 {
        4.0 * PI * R.powi(3) * (qR.sin() - qR * qR.cos()) / qR.powi(3)
    } else {
        4.0 / 3.0 * PI * R.powi(3)
    }
}

/// Size distribution integral.
///
/// The problem is trivially mapped on an integral over exp(-x^2) by a variable
/// transformation, which is solved by a Gauss-Hermite quadrature
fn size_distributed_formfactor(q: f64, R: f64, sigR: f64, gh_quad: &GaussHermite) -> f64 {
    let integral =
        gh_quad.integrate(|r_value| amplitude(q, R * (SQ_2 * r_value * sigR).exp()).powi(2));
    integral * FRAC_SQ_PI
}

/// Formfactor of a spherical particle
///
/// P = N/V * V_p^2 * DeltaSLD^2 * F^2
/// F = 3 (qRsin(qR) - cos(qR))/(qR)^3
/// Additionally a size distribution average is performed
pub fn formfactor(p: &Array1<f64>, q: &Array1<f64>) -> Array1<f64> {
    let I0 = p[0];
    let R = p[1];
    let sigR = p[2];
    let SLDsphere = p[3];
    let SLDmatrix = p[4];
    let deg = p[5] as usize;

    let I: Array1<f64>;
    if sigR > 0.0 && deg > 1 {
        let quad = GaussHermite::init(deg);
        I = q.map(|qval| size_distributed_formfactor(*qval, R, sigR, &quad));
    } else {
        I = q.map(|qval| amplitude(*qval, R).powi(2));
    }
    I0 * (SLDsphere - SLDmatrix).powi(2) * I
}
