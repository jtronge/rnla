//! Rusty Numerical Linear Algebra package.
//!
//! Some experimental NLA code in Rust.

mod operations;
mod rng;
use rng::RNG;

pub struct Matrix {
    pub m: usize,
    pub n: usize,
    data: Vec<f64>,
}

impl Matrix {
    /// Return a randomly generated matrix.
    pub fn rand(m: usize, n: usize) -> Matrix {
        let mut rng = RNG::new(1);
        Matrix {
            m,
            n,
            data: (0..m*n).map(|_| rng.rand_f64()).collect(),
        }
    }

    /// Get a matrix entry.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.m + j]
    }

    /// Set a value in the matrix.
    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        self.data[i * self.m + j] = value;
    }
}

/// Matrix-vector multiply (u = A * v).
pub fn matvecmul(a: &Matrix, v: &[f64], u: &mut [f64]) {
    assert_eq!(a.n, v.len());
    assert_eq!(a.m, u.len());
    for i in 0..a.m {
        u[i] = 0.0;
        for j in 0..a.n {
            u[i] += a.get(i, j) * v[j];
        }
    }
}

/// Matrix-matrix multiply (C = A * B).
pub fn matmul(a: &Matrix, b: &Matrix, c: &mut Matrix) {
    assert_eq!(a.n, b.m);
    assert_eq!(a.m, c.m);
    assert_eq!(b.n, c.n);
    for i in 0..c.m {
        for j in 0..c.n {
            c.set(i, j, 0.0);
        }
    }

    for i in 0..a.m {
        for j in 0..a.n {
            for k in 0..b.n {
                c.set(i, k, c.get(i, k) + a.get(i, j) * b.get(j, k));
            }
        }
    }
}
