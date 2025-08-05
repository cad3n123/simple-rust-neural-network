use crate::types::Vector;

#[derive(Clone)]
pub struct TrainingDatum<T> {
    pub input: Vector<T>,
    pub output: Vector<T>,
}
