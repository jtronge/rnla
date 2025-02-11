//! Rusty Numerical Linear Algebra package.
//!
//! Some experimental NLA code in Rust.

pub mod bench;
mod operations;
use rand::prelude::*;

pub struct Matrix {
    pub m: usize,
    pub n: usize,
    data: Vec<f64>,
}

impl Matrix {
    /// Return a zeroed matrix.
    pub fn zero(m: usize, n: usize) -> Matrix {
        Matrix {
            m,
            n,
            data: (0..m * n).map(|_| 0.0).collect(),
        }
    }

    /// Return a randomly generated matrix.
    pub fn rand(m: usize, n: usize) -> Matrix {
        let mut rng = rand::rng();
        Matrix {
            m,
            n,
            data: (0..m * n).map(|_| rng.random()).collect(),
        }
    }

    pub fn view(&self) -> MatrixView {
        let view_data = (0..self.m)
            .map(|i| &self.data[i * self.n..i * self.n + self.n])
            .collect();
        MatrixView {
            m: self.m,
            n: self.n,
            data: view_data,
        }
    }

    pub fn view_mut(&mut self) -> MatrixViewMut {
        let mut view_data = vec![];

        let mut remain = &mut self.data[..];
        for _ in 0..self.m {
            let (row, left) = remain.split_at_mut(self.n);
            view_data.push(row);
            remain = left;
        }

        MatrixViewMut {
            m: self.m,
            n: self.n,
            data: view_data,
        }
    }
}

/// Matrix indexing trait abstraction.
pub trait MatrixIndex {
    /// Get a matrix entry.
    fn get(&self, i: usize, j: usize) -> f64;

    /// Set a matrix entry.
    fn set(&mut self, i: usize, j: usize, value: f64);

    /// Get a matrix entry without bounds checking.
    unsafe fn get_unchecked(&self, i: usize, j: usize) -> f64;

    /// Set a matrix entry without bounds checking.
    unsafe fn set_unchecked(&mut self, i: usize, j: usize, value: f64);
}

impl MatrixIndex for Matrix {
    /// Get a matrix entry.
    #[inline]
    fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.n + j]
    }

    /// Set a value in the matrix.
    #[inline]
    fn set(&mut self, i: usize, j: usize, value: f64) {
        self.data[i * self.n + j] = value;
    }

    /// Get a matrix entry.
    #[inline]
    unsafe fn get_unchecked(&self, i: usize, j: usize) -> f64 {
        *self.data.get_unchecked(i * self.n + j)
    }

    /// Set a value in the matrix.
    #[inline]
    unsafe fn set_unchecked(&mut self, i: usize, j: usize, value: f64) {
        *self.data.get_unchecked_mut(i * self.n + j) = value;
    }
}

pub struct MatrixView<'a> {
    pub m: usize,
    pub n: usize,
    data: Vec<&'a [f64]>,
}

impl<'a> MatrixIndex for MatrixView<'a> {
    #[inline]
    fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i][j]
    }

    #[inline]
    fn set(&mut self, i: usize, j: usize, value: f64) {
        panic!("cannot set entries with MatrixView");
    }

    #[inline]
    unsafe fn get_unchecked(&self, i: usize, j: usize) -> f64 {
        *self.data
            .get_unchecked(i)
            .get_unchecked(j)
    }

    #[inline]
    unsafe fn set_unchecked(&mut self, i: usize, j: usize, value: f64) {
        panic!("cannot set entries with MatrixView");
    }
}

pub struct MatrixViewMut<'a> {
    pub m: usize,
    pub n: usize,
    data: Vec<&'a mut [f64]>,
}

impl<'a> MatrixIndex for MatrixViewMut<'a> {
    #[inline]
    fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i][j]
    }

    #[inline]
    fn set(&mut self, i: usize, j: usize, value: f64) {
        self.data[i][j] = value;
    }

    #[inline]
    unsafe fn get_unchecked(&self, i: usize, j: usize) -> f64 {
        *self.data
            .get_unchecked(i)
            .get_unchecked(j)
    }

    #[inline]
    unsafe fn set_unchecked(&mut self, i: usize, j: usize, value: f64) {
        *self.data
            .get_unchecked_mut(i)
            .get_unchecked_mut(j) = value;
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

pub fn matmul(a: &MatrixView, b: &MatrixView, c: &mut MatrixViewMut) {
    assert_eq!(a.m, c.m);
    assert_eq!(a.n, b.m);
    assert_eq!(b.n, c.n);

    for i in 0..c.m {
        for j in 0..c.n {
            unsafe { c.set_unchecked(i, j, 0.0) };
        }
    }

    for i in 0..a.m {
        for k in 0..a.n {
            for j in 0..b.n {
                unsafe {
                    c.set_unchecked(i, j, c.get_unchecked(i, j)
                                          + a.get_unchecked(i, k)
                                            * b.get_unchecked(k, j));
                }
            }
        }
    }
}
