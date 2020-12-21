use gauss_quad::{GaussHermite, GaussLegendre};
use ndarray::Array1;

const PI_2: f64 = 1.5707963267948966;
const SQ_2: f64 = 1.4142135623730951;
const FRAC_SQ_PI: f64 = 0.56418958354775628;

/// Formfactor Amplitude F of a Cube Particle
///
/// See for example http://gisaxs.com/index.php/Form_Factor:Cube
fn amplitude(qx: f64, qy: f64, qz: f64, a: f64) -> f64 {
    a.powi(3) * sinc(0.5 * qx * a) * sinc(0.5 * qy * a) * sinc(0.5 * qz * a)
}

/// Simple sinc(x) function to avoid division by zero
fn sinc(x: f64) -> f64 {
    if x != 0.0 {
        x.sin() / x
    } else {
        1.0
    }
}

/// Orientation integral.
///
/// Integrate in spherical coordinates over all possible angles. Cube symmetry
/// reduces the integral to the range 0..pi/2
fn orientation_averaged_formfactor(q: f64, a: f64, gl_quad: &GaussLegendre) -> f64 {
    gl_quad.integrate(0.0, PI_2, |theta| theta_integral(q, a, theta, &gl_quad))
}

/// Inner integral of orientation integral
///
/// Calculation of cosine and sine of theta is put before the inner integral
fn theta_integral(q: f64, a: f64, theta: f64, gl_quad: &GaussLegendre) -> f64 {
    let cos_theta = theta.cos();
    let sin_theta = theta.sin();
    gl_quad.integrate(0.0, PI_2, |phi| {
        amplitude(
            q * phi.cos() * sin_theta,
            q * phi.sin() * sin_theta,
            q * cos_theta,
            a,
        )
        .powi(2)
    }) * sin_theta
}

/// Size distribution integral.
///
/// The problem is trivially mapped on an integral over exp(-x^2) by a variable
/// transformation, which is solved by a Gauss-Hermite quadrature
fn size_distributed_formfactor(
    q: f64,
    a: f64,
    sigA: f64,
    gh_quad: &GaussHermite,
    gl_quad: &GaussLegendre,
) -> f64 {
    let integral = gh_quad.integrate(|a_value| {
        orientation_averaged_formfactor(q, a * (SQ_2 * a_value * sigA).exp(), &gl_quad)
    });
    integral * FRAC_SQ_PI
}

/// Formfactor of a cubically shaped particle
///
/// P = N/V * V_p^2 * DeltaSLD^2 * F^2
/// F = sinc(q_x*a/2)*sinc(q_y*a/2)*sinc(q_z*a/2)
/// Additionally a orientation & size distribution average is performed
pub fn formfactor(p: &Array1<f64>, q: &Array1<f64>) -> Array1<f64> {
    let I0 = p[0];
    let a = p[1];
    let sigA = p[2];
    let SLDcube = p[3];
    let SLDmatrix = p[4];
    let gl_deg = p[5] as usize;
    let gh_deg = p[6] as usize;

    let I: Array1<f64>;
    let gl_quad = GaussLegendre::init(gl_deg);
    if sigA > 0.0 && gh_deg > 1 {
        let gh_quad = GaussHermite::init(gh_deg);
        I = q.map(|qval| size_distributed_formfactor(*qval, a, sigA, &gh_quad, &gl_quad));
    } else {
        I = q.map(|qval| orientation_averaged_formfactor(*qval, a, &gl_quad));
    }
    I0 * (SLDcube - SLDmatrix).powi(2) * I
}
