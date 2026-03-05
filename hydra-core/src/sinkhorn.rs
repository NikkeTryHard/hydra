//! Sinkhorn-Knopp projection for Mixture-SIB belief inference.

pub struct SinkhornConfig {
    pub max_iters: u16,
    pub tol: f64,
    pub num_components: u8,
}

impl Default for SinkhornConfig {
    fn default() -> Self {
        Self {
            max_iters: 50,
            tol: 1e-8,
            num_components: 4,
        }
    }
}

pub fn sinkhorn_project(
    kernel: &[f64; 136],
    row_sums: &[f64; 34],
    col_sums: &[f64; 4],
    max_iters: u16,
    tol: f64,
) -> [f64; 136] {
    let mut b = [1.0f64; 136];
    let mut u = [1.0f64; 34];
    let mut v = [1.0f64; 4];

    for _ in 0..max_iters {
        for i in 0..34 {
            let mut s = 0.0;
            for j in 0..4 {
                s += kernel[i * 4 + j] * v[j] * b[i * 4 + j];
            }
            u[i] = if s > 1e-15 { row_sums[i] / s } else { 0.0 };
        }
        for j in 0..4 {
            let mut s = 0.0;
            for i in 0..34 {
                s += kernel[i * 4 + j] * u[i] * b[i * 4 + j];
            }
            v[j] = if s > 1e-15 { col_sums[j] / s } else { 0.0 };
        }

        for i in 0..34 {
            for j in 0..4 {
                b[i * 4 + j] = kernel[i * 4 + j] * u[i] * v[j];
            }
        }

        let mut row_err = 0.0f64;
        for i in 0..34 {
            let mut s = 0.0;
            for j in 0..4 {
                s += b[i * 4 + j];
            }
            row_err += (s - row_sums[i]).abs();
        }
        if row_err < tol {
            break;
        }
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinkhorn_converges_to_margins() {
        let kernel = [1.0f64; 136];
        let row_sums = [4.0f64; 34];
        let col_sums = [34.0; 4];
        let b = sinkhorn_project(&kernel, &row_sums, &col_sums, 100, 1e-6);
        for i in 0..34 {
            let s: f64 = (0..4).map(|j| b[i * 4 + j]).sum();
            assert!((s - 4.0).abs() < 0.01, "row {i} sum = {s}");
        }
        for j in 0..4 {
            let s: f64 = (0..34).map(|i| b[i * 4 + j]).sum();
            assert!((s - 34.0).abs() < 0.01, "col {j} sum = {s}");
        }
    }
}
