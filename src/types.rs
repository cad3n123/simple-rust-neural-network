use ndarray::{Array1, Array2};

pub type Matrix<T> = Array2<T>;
pub type Vector<T> = Array1<T>;
#[cfg(feature = "f32")]
pub(crate) type Float = f32;
#[cfg(feature = "f64")]
pub(crate) type Float = f64;
