use crate::Matrix;

fn saxpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), y.len());
    for (xval, yval) in x.iter().zip(y.iter_mut()) {
        *yval = alpha * *xval + *yval;
    }
}

fn gaxpy() {}
