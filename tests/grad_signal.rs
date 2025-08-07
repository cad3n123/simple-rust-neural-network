use ndarray::array;
use simple_neural_network::neural_network::NeuralNetwork;
use simple_neural_network::training_data::TrainingDatum;

fn loss_of(nn: &NeuralNetwork, d: &TrainingDatum<f32>) -> f32 {
    let y = nn.run(&d.input);
    0.5 * (&y - &d.output).mapv(|v| v * v).sum()
}

#[test]
fn evolve_target_reduces_loss_for_single_example() {
    let mut nn = NeuralNetwork::new(3, 1, vec![4, 4]);
    let datum = TrainingDatum {
        input: array![0.5, -0.2, 0.3],
        output: array![0.7],
    };
    let before = loss_of(&nn, &datum);
    // apply several small steps to smooth stochasticity
    for _ in 0..20 {
        nn.evolve_target(&datum);
    }
    let after = loss_of(&nn, &datum);
    assert!(
        after < before,
        "loss did not reduce: before={before}, after={after}"
    );
}
