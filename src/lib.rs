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
    /// Create from a vector.
    pub fn from_vec(m: usize, n: usize, data: Vec<f64>) -> Matrix {
        assert_eq!(m * n, data.len());
        Matrix {
            m,
            n,
            data,
        }
    }

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

    /// Return the internal vector data.
    pub fn to_vec(self) -> Vec<f64> {
        self.data
    }

    /// Return a matrix view.
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

    /// Return a mutable matrix view.
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

const EPSILON: f64 = 10e-8;

/// Return whether a is approximately equal to b.
pub fn approx(a: f64, b: f64) -> bool {
    (a-b).abs() < EPSILON
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn matmul_times_zero() {
        let m = 10;
        let n = 4;
        let p = 8;
        let x = Matrix::zero(m, n);
        let y = Matrix::rand(n, p);
        let mut z = Matrix::zero(m, p);

        matmul(&x.view(), &y.view(), &mut z.view_mut());

        for i in 0..m {
            for j in 0..p {
                assert!(approx(z.get(i, j), 0.0));
            }
        }
    }

    #[test]
    fn matmul_simple() {
        let x = Matrix::from_vec(3, 2, vec![1.0, 2.0,
                                            3.0, 4.0,
                                            5.0, 6.0]);
        let y = Matrix::from_vec(2, 2, vec![1.0, 2.0,
                                            0.0, 8.0]);
        let mut z = Matrix::zero(3, 2);

        matmul(&x.view(), &y.view(), &mut z.view_mut());

        let z = z.to_vec();
        let expected = vec![1.0, 18.0, 3.0, 38.0, 5.0, 58.0];
        assert!(z.iter().zip(&expected).all(|(a, b)| approx(*a, *b)));
    }
}
