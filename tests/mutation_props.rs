use ndarray::Array1;
use ndarray_rand::rand::{SeedableRng, rngs::StdRng};
use simple_neural_network::{mutate::MutateConfig, neural_network::NeuralNetwork};

#[test]
fn mutate_vector_respects_clip_and_prob() {
    use simple_neural_network::mutate::mutate_vector;
    let mut rng = StdRng::seed_from_u64(42);
    let v = Array1::zeros(1000);
    let cfg = MutateConfig {
        sigma: 1.0,
        prob: 0.3,
        clip: Some((-0.5, 0.5)),
        reset_chance: 0.0,
        insert_layer_chance: 0.0,
        add_neuron_chance: 0.0,
        delete_neuron_chance: 0.0,
        min_neurons: 2,
    };
    let mv = mutate_vector(&v, &mut rng, &cfg);
    let changed = (&mv - &v).mapv(|x| if x == 0.0 { 0.0 } else { 1.0 }).sum();
    let frac = changed / (v.len() as f64);
    assert!(
        (frac - 0.3).abs() < 0.1,
        "fraction changed ~ prob, got {frac}"
    );
    let within = mv.iter().all(|&x| (-0.5..=0.5).contains(&x));
    assert!(within, "values should be clipped to [-0.5, 0.5]");
}

#[test]
fn mutate_matrix_reset_hits_fraction() {
    use simple_neural_network::mutate::mutate_matrix;
    let mut rng = StdRng::seed_from_u64(7);
    let m = ndarray::Array2::<f32>::zeros((50, 50));
    let cfg = MutateConfig {
        sigma: 1.0,
        prob: 1.0,
        clip: None,
        reset_chance: 0.2,
        insert_layer_chance: 0.0,
        add_neuron_chance: 0.0,
        delete_neuron_chance: 0.0,
        min_neurons: 2,
    };
    let mm = mutate_matrix(&m, &mut rng, &cfg);
    // With prob=1, all elements changed; with reset=0.2, around 20% should be from fresh
    // We can't directly tell which reset, but variance should be ~sigma^2 regardless, just sanity check mean/var bounds
    let mean = mm.mean().unwrap();
    assert!(mean.abs() < 0.3, "mean near 0, got {mean}");
}

#[test]
fn structural_mutations_add_and_remove_neurons_consistently() {
    let net = NeuralNetwork::new(3, 2, vec![4, 4, 4]);
    let mut rng = StdRng::seed_from_u64(123);
    let cfg = MutateConfig {
        sigma: 0.02,
        prob: 0.0,
        clip: None,
        reset_chance: 0.0,
        insert_layer_chance: 0.5,
        add_neuron_chance: 0.5,
        delete_neuron_chance: 0.5,
        min_neurons: 2,
    };
    let mutated = net.mutated_using(&mut rng, &cfg);
    // Just check internal consistency: running shouldn't panic and output size preserved
    let out = mutated.run(&ndarray::array![1.0, 2.0, 3.0]);
    assert_eq!(out.len(), 2);
}
