use ndarray::{Array1, Array2, Axis, s};
use ndarray_rand::{
    RandomExt,
    rand::{Rng, seq::SliceRandom, thread_rng},
    rand_distr::Normal,
};
use serde::{Deserialize, Serialize};

use crate::{
    mutate::{MutateConfig, mutate_matrix, mutate_vector},
    training_data::TrainingDatum,
    types::{Float, Matrix, Vector},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetwork {
    layers: Box<[Layer]>,
}
#[derive(Debug, Clone, Serialize)]
struct Layer {
    weights: Weights,
    biases: Biases,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Weights(Matrix<Float>);
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Biases(Vector<Float>);

impl NeuralNetwork {
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
    pub fn run(&self, input: &Array1<Float>) -> Array1<Float> {
        let mut result = input.clone();
        for (j, layer) in self.layers.iter().enumerate() {
            let wx = layer.weights.0.dot(&result) + &layer.biases.0;
            let is_last = j + 1 == self.layers.len();
            if is_last {
                result = wx; // linear output
            } else {
                let mut a = wx;
                #[cfg(not(feature = "rayon"))]
                {
                    a.mapv_inplace(Self::activation_function);
                }
                #[cfg(feature = "rayon")]
                {
                    a.par_mapv_inplace(Self::activation_function);
                }
                result = a;
            }
        }
        result
    }
    pub fn train_data(&mut self, epochs: usize, mut training_data: Vec<TrainingDatum<Float>>) {
        let mut rng = thread_rng();
        for _ in 0..epochs {
            training_data.shuffle(&mut rng);
            for training_datum in &training_data {
                self.evolve_target(training_datum);
            }
        }
    }
    const fn activation_function(value: Float) -> Float {
        value.max(0.)
    }
    fn activation_derivative(value: Float) -> Float {
        if value > 0. { 1. } else { 0. }
    }
    #[allow(clippy::missing_panics_doc)]
    pub fn evolve_target(&mut self, training_datum: &TrainingDatum<Float>) {
        // Learning rate (tweak as needed or pass in)
        let lr = 1e-2;

        // ---------- Forward pass with caches ----------
        // a[0] = input, a[L] = output; zs[l] = pre-activation of layer l (1..=L)
        let mut activations: Vec<Array1<Float>> = Vec::with_capacity(self.layers.len() + 1);
        let mut zs: Vec<Array1<Float>> = Vec::with_capacity(self.layers.len());

        activations.push(training_datum.input.clone());
        let mut a = training_datum.input.clone();

        for (j, layer) in self.layers.iter().enumerate() {
            let z = layer.weights.0.dot(&a) + &layer.biases.0;
            let is_last = j + 1 == self.layers.len();
            let anext = if is_last {
                z.clone() // linear output
            } else {
                let mut t = z.clone();
                #[cfg(not(feature = "rayon"))]
                {
                    t.mapv_inplace(Self::activation_function);
                }
                #[cfg(feature = "rayon")]
                {
                    t.par_mapv_inplace(Self::activation_function);
                }
                t
            };
            zs.push(z);
            activations.push(anext.clone());
            a = anext;
        }

        // ---------- Backward pass ----------

        // Iterate layers in reverse
        let mut delta = activations.last().unwrap() - training_datum.output.clone();

        for l_rev in (0..self.layers.len()).rev() {
            let a_prev = &activations[l_rev];
            let d_w = delta
                .view()
                .insert_axis(Axis(1))
                .dot(&a_prev.view().insert_axis(Axis(0)));

            self.layers[l_rev].weights.0.scaled_add(-lr, &d_w);
            self.layers[l_rev].biases.0.scaled_add(-lr, &delta);

            if l_rev > 0 {
                let wt = self.layers[l_rev].weights.0.t();
                let mut delta_prev = wt.dot(&delta);

                // ReLU’ for the *hidden* pre-activation z_{l-1}
                let z_prev = &mut zs[l_rev - 1];
                #[cfg(not(feature = "rayon"))]
                {
                    z_prev.mapv_inplace(Self::activation_derivative);
                }
                #[cfg(feature = "rayon")]
                {
                    z_prev.par_mapv_inplace(Self::activation_derivative);
                }

                ndarray::Zip::from(&mut delta_prev)
                    .and(&*z_prev)
                    .for_each(|d, &g| *d *= g);
                delta = delta_prev;
            }
        }
    }
    /// Pure (non-destructive) mutation that returns a new network.
    #[must_use]
    pub fn mutate_using<R: Rng + ?Sized>(&self, rng: &mut R, cfg: &MutateConfig) -> Self {
        let mut layers = self.layers.to_vec();
        let mut new_layers = vec![];

        let mut input_size = self.input_size();
        let mut i = 0;

        while i < layers.len() {
            // Possibly insert a new layer before the current one
            if rng.gen_bool(cfg.insert_layer_chance) {
                let inserted = Layer::new_random(input_size, layers[i].weights.0.shape()[1]);
                layers.insert(i, inserted);
                i += 1;
            }

            let layer = &layers[i];
            let last_idx = layers.len().saturating_sub(1);
            let is_output_layer = i == last_idx;
            let is_penultimate = i + 1 == last_idx;

            // Now create a mutated copy of the current layer
            let mut new_layer = layer.mutated_using(rng, cfg);
            let mut updated_next = None;
            if rng.gen_bool(cfg.add_neuron_chance) && !is_penultimate && !is_output_layer {
                new_layer.add_neuron(rng);

                if let Some(next_layer) = layers.get(i + 1) {
                    let mut next_layer = next_layer.clone();
                    next_layer.add_input(rng);
                    updated_next = Some(next_layer);
                }
            }

            let can_delete_here = !is_output_layer
                && !is_penultimate
                && new_layer.biases.0.len() > cfg.min_neurons
                && rng.gen_bool(cfg.delete_neuron_chance);

            if can_delete_here {
                new_layer.remove_first_neuron();

                if let Some(updated_next) = &mut updated_next {
                    updated_next.remove_first_input();
                } else if let Some(next_layer) = layers.get(i + 1) {
                    let mut next_layer = next_layer.clone();
                    next_layer.remove_first_input();
                    updated_next = Some(next_layer);
                }
            }

            input_size = new_layer.weights.0.shape()[0];
            new_layers.push(new_layer);

            if let Some(updated_next) = updated_next {
                layers[i + 1] = updated_next;
            }
            i += 1;
        }

        // Possibly insert a layer after the last one
        if rng.gen_bool(cfg.insert_layer_chance) {
            let inserted = Layer::new_random(input_size, self.output_size());
            new_layers.push(inserted);
        }

        Self {
            layers: new_layers.into_boxed_slice(),
        }
    }

    /// Convenience: uses `thread_rng()`.
    #[must_use]
    pub fn mutated(&self, cfg: &MutateConfig) -> Self {
        let mut rng = thread_rng();
        self.mutate_using(&mut rng, cfg)
    }
    fn input_size(&self) -> usize {
        self.layers.first().map_or(1, |l| l.weights.0.shape()[1])
    }

    fn output_size(&self) -> usize {
        self.layers.last().map_or(1, |l| l.weights.0.shape()[0])
    }

    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    #[must_use]
    pub fn get_layer_output_sizes(&self) -> Vec<usize> {
        self.layers.iter().map(|l| l.weights.0.shape()[0]).collect()
    }
}
impl Layer {
    fn new_random(num_inputs: usize, num_outputs: usize) -> Self {
        Self {
            weights: Weights::new_random(num_inputs, num_outputs),
            biases: Biases::zeros(num_outputs),
        }
    }
    fn mutated_using<R: Rng + ?Sized>(&self, rng: &mut R, cfg: &MutateConfig) -> Self {
        let w = mutate_matrix(&self.weights.0, rng, cfg);
        let b = mutate_vector(&self.biases.0, rng, cfg);
        Self {
            weights: Weights(w),
            biases: Biases(b),
        }
    }
    fn add_neuron<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        let (out, inp) = self.weights.0.dim();

        // Add row to weights
        let mut new_weights = Array2::zeros((out + 1, inp));
        new_weights.slice_mut(s![..out, ..]).assign(&self.weights.0);
        new_weights.row_mut(out).assign(&Array1::random_using(
            inp,
            Normal::new(0., 0.1).unwrap(),
            rng,
        ));

        // Add bias
        let mut new_biases = Array1::zeros(out + 1);
        new_biases.slice_mut(s![..out]).assign(&self.biases.0);
        new_biases[out] = 0.; // or random

        self.weights.0 = new_weights;
        self.biases.0 = new_biases;
    }
    fn add_input<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        let (out, inp) = self.weights.0.dim();

        let mut new_weights = Array2::zeros((out, inp + 1));
        new_weights.slice_mut(s![.., ..inp]).assign(&self.weights.0);
        let new_col = Array1::random_using(out, Normal::new(0., 0.1).unwrap(), rng);
        new_weights.column_mut(inp).assign(&new_col);

        self.weights.0 = new_weights;
        // Biases unchanged
    }
    fn remove_first_neuron(&mut self) {
        // Remove first row from weights
        self.weights.0 = self.weights.0.slice(s![1.., ..]).to_owned();

        // Remove bias
        self.biases.0 = self.biases.0.slice(s![1..]).to_owned();
    }
    fn remove_first_input(&mut self) {
        // Create a new matrix without the first column
        self.weights.0 = self.weights.0.slice(s![.., 1..]).to_owned();
        // Biases unchanged
    }
}
impl<'de> Deserialize<'de> for Layer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct CStructLayer {
            weights: Weights,
            biases: Biases,
        }

        let CStructLayer { weights, biases } = CStructLayer::deserialize(deserializer)?;
        Ok(Self { weights, biases })
    }
}
impl Weights {
    fn new_random(num_inputs: usize, num_outputs: usize) -> Self {
        let two = 2.0;
        let n_inputs = num_inputs as Float;
        let stddev = (two / n_inputs).sqrt();

        Self(Array2::random(
            (num_outputs, num_inputs),
            Normal::new(0., stddev).unwrap(),
        ))
    }
}
impl Biases {
    fn zeros(shape: usize) -> Self {
        Self(Array1::zeros(shape))
    }
}
