use ndarray::Array1;
use proptest::prelude::*;
use simple_neural_network::neural_network::NeuralNetwork;

proptest! {
    #[test]
    fn run_works_for_random_layer_layout(num_inputs in 1usize..6, num_outputs in 1usize..6, l1 in 1usize..6, l2 in 1usize..6) {
        let mut layers = vec![l1, l2];
        // sometimes include only one hidden layer
        if l2 % 2 == 0 { layers.pop(); }
        let net = NeuralNetwork::new(num_inputs, num_outputs, layers);
        let x = Array1::ones(num_inputs);
        let y = net.run(&x);
        prop_assert_eq!(y.len(), num_outputs);
    }
}
