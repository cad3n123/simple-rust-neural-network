use ndarray::Array1;
use ndarray_rand::rand::{SeedableRng, rngs::StdRng};
use simple_neural_network::{mutate::MutateConfig, neural_network::NeuralNetwork};

#[test]
fn test_structural_mutation_behavior() {
    let input_size = 3;
    let output_size = 2;
    let mut rng = StdRng::seed_from_u64(42);

    // Config for probabilistic mutation
    let cfg_chance = MutateConfig {
        sigma: 0.1,
        prob: 1.0,
        reset_chance: 0.1,
        insert_layer_chance: 0.25, // realistic
        add_neuron_chance: 0.25,
        clip: Some((-1.0, 1.0)),
    };

    // Config for forced mutation
    let cfg_forced = MutateConfig {
        sigma: 0.1,
        prob: 1.0,
        reset_chance: 0.1,
        insert_layer_chance: 1.0,
        add_neuron_chance: 1.0,
        clip: Some((-1.0, 1.0)),
    };

    let input = Array1::ones(input_size);

    // -------- Case 1: Empty network mutation with guaranteed insertion
    let net_empty = NeuralNetwork::new(input_size, output_size, vec![]);
    let mutated_forced = net_empty.mutate_using(&mut rng, &cfg_forced);
    assert!(
        mutated_forced.layer_count() >= 1,
        "Forced mutation should add a layer"
    );

    let out = mutated_forced.run(&input);
    assert_eq!(out.len(), output_size, "Output size must match");

    // -------- Case 2: Check neuron expansion in forced config
    let net_single = NeuralNetwork::new(input_size, output_size, vec![2]);
    let mutated_neurons = net_single.mutate_using(&mut rng, &cfg_forced);
    assert!(
        mutated_neurons
            .get_layer_output_sizes()
            .into_iter()
            .any(|size| size > 2),
        "At least one layer should have more neurons"
    );

    // -------- Case 3: Probabilistic mutation eventually adds structure
    let mut success = false;
    for _ in 0..50 {
        let net = net_empty.mutate_using(&mut rng, &cfg_chance);
        if net.layer_count() > 0 {
            success = true;
            break;
        }
    }
    assert!(
        success,
        "Probabilistic mutation should eventually add layers"
    );
}
