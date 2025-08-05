use ndarray::{Array1, Array2, Axis, ScalarOperand};
use ndarray_rand::{
    RandomExt,
    rand::{Rng, seq::SliceRandom, thread_rng},
    rand_distr::{Distribution, Normal, StandardNormal},
};
use num_traits::{Float, FromPrimitive};

use crate::{
    mutate::{MutateConfig, mutate_matrix, mutate_vector},
    training_data::TrainingDatum,
    types::{Matrix, Vector},
};

#[derive(Debug, Clone)]
pub struct NeuralNetwork<T> {
    layers: Box<[Layer<T>]>,
}
#[derive(Debug, Clone)]
struct Layer<T> {
    weights: Weights<T>,
    biases: Biases<T>,
}
#[derive(Debug, Clone)]
struct Weights<T>(Matrix<T>);
#[derive(Debug, Clone)]
struct Biases<T>(Vector<T>);

impl<T> NeuralNetwork<T>
where
    T: Float + FromPrimitive + ScalarOperand + 'static,
    StandardNormal: Distribution<T>,
{
    #[must_use]
    pub fn new(num_inputs: usize, num_outputs: usize, mut layer_shapes: Vec<usize>) -> Self {
        layer_shapes.push(num_outputs);
        layer_shapes.insert(0, num_inputs);
        let shapes_iter = layer_shapes.iter();
        let input_shapes = shapes_iter.clone().take(shapes_iter.len() - 1);
        let output_shapes = shapes_iter.skip(1);

        let mut layers = vec![];
        for (&input_shape, &output_shape) in input_shapes.zip(output_shapes) {
            layers.push(Layer::new_random(input_shape, output_shape));
        }

        Self {
            layers: layers.into_boxed_slice(),
        }
    }
    #[must_use]
    pub fn run(&self, input: &Array1<T>) -> Array1<T> {
        let mut result = input.clone();
        for layer in &self.layers {
            let wx = layer.weights.0.dot(&result);
            result = (wx + &layer.biases.0).map(Self::activation_function);
        }
        result
    }
    pub fn train_data(&mut self, epochs: usize, mut training_data: Vec<TrainingDatum<T>>) {
        let mut rng = thread_rng();
        for _ in 0..epochs {
            training_data.shuffle(&mut rng);
            for training_datum in training_data.clone() {
                self.evolve_target(&training_datum);
            }
        }
    }
    fn activation_function(value: &T) -> T {
        value.max(T::zero())
    }
    fn activation_derivative(value: &T) -> T {
        if *value > T::zero() {
            T::one()
        } else {
            T::zero()
        }
    }
    fn evolve_target(&mut self, training_datum: &TrainingDatum<T>) {
        // Learning rate (tweak as needed or pass in)
        let lr = T::from_f64(1e-2).unwrap();

        // ---------- Forward pass with caches ----------
        // a[0] = input, a[L] = output; zs[l] = pre-activation of layer l (1..=L)
        let mut activations: Vec<Array1<T>> = Vec::with_capacity(self.layers.len() + 1);
        let mut zs: Vec<Array1<T>> = Vec::with_capacity(self.layers.len());

        activations.push(training_datum.input.clone());
        let mut a = training_datum.input.clone();

        for layer in &self.layers {
            let z = layer.weights.0.dot(&a) + &layer.biases.0;
            let anext = z.map(Self::activation_function);
            zs.push(z);
            activations.push(anext.clone());
            a = anext;
        }

        // ---------- Backward pass ----------
        // Loss: 0.5 * ||a_L - y||^2 -> dL/da_L = (a_L - y)
        let mut delta = activations.last().unwrap() - training_datum.output.clone();
        // Chain with ReLU' at output layer
        {
            let z_l = zs.last().unwrap();
            delta = delta * &z_l.map(Self::activation_derivative);
        }

        // Iterate layers in reverse
        for l_rev in (0..self.layers.len()).rev() {
            let a_prev = &activations[l_rev];

            // dW = delta ⊗ a_prev
            let d_w = delta
                .view()
                .insert_axis(Axis(1)) // (out, 1)
                .dot(&a_prev.view().insert_axis(Axis(0))); // (1, in) -> (out, in)

            // db = delta
            // Update params: W -= lr*dW, b -= lr*db
            self.layers[l_rev].weights.0.scaled_add(-lr, &d_w); // W -= lr * dW
            self.layers[l_rev].biases.0.scaled_add(-lr, &delta); // b -= lr * db

            // Prepare delta for previous layer (if any):
            if l_rev > 0 {
                // delta_prev = (W^T * delta) ⊙ ReLU'(z_{l-1})
                let wt = self.layers[l_rev].weights.0.t().to_owned();
                let mut delta_prev = wt.dot(&delta);
                let z_prev = &zs[l_rev - 1];
                delta_prev = delta_prev * &z_prev.map(Self::activation_derivative);
                delta = delta_prev;
            }
        }
    }
    /// Pure (non-destructive) mutation that returns a new network.
    #[must_use]
    pub fn mutate_using<R: Rng + ?Sized>(&self, rng: &mut R, cfg: &MutateConfig<T>) -> Self {
        let layers = self
            .layers
            .iter()
            .map(|ly| ly.mutated_using(rng, cfg))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self { layers }
    }

    /// Convenience: uses `thread_rng()`.
    #[must_use]
    pub fn mutate(&self, cfg: &MutateConfig<T>) -> Self {
        let mut rng = thread_rng();
        self.mutate_using(&mut rng, cfg)
    }
}
impl<T> Layer<T>
where
    T: Float + FromPrimitive + ScalarOperand,
    StandardNormal: Distribution<T>,
{
    fn new_random(num_inputs: usize, num_outputs: usize) -> Self {
        Self {
            weights: Weights::new_random(num_inputs, num_outputs),
            biases: Biases::zeros(num_outputs),
        }
    }
    fn mutated_using<R: Rng + ?Sized>(&self, rng: &mut R, cfg: &MutateConfig<T>) -> Self {
        let w = mutate_matrix(&self.weights.0, rng, cfg);
        let b = mutate_vector(&self.biases.0, rng, cfg);
        Self {
            weights: Weights(w),
            biases: Biases(b),
        }
    }
}
impl<T> Weights<T>
where
    T: Float + FromPrimitive,
    StandardNormal: Distribution<T>,
{
    fn new_random(num_inputs: usize, num_outputs: usize) -> Self {
        let two = T::from_f64(2.0).unwrap();
        let n_inputs = T::from_usize(num_inputs).unwrap();
        let stddev = (two / n_inputs).sqrt();

        Self(Array2::random(
            (num_outputs, num_inputs),
            Normal::new(T::zero(), stddev).unwrap(),
        ))
    }
}
impl<T> Biases<T>
where
    T: Float + FromPrimitive,
    StandardNormal: Distribution<T>,
{
    fn zeros(shape: usize) -> Self {
        Self(Array1::zeros(shape))
    }
}
