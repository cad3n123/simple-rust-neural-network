use ndarray::{Array1, array};
use simple_neural_network::{neural_network::NeuralNetwork, training_data::TrainingDatum};

fn loss(a: &Array1<f32>, y: &Array1<f32>) -> f32 {
    // 0.5 * ||a - y||^2
    0.5 * (a - y).mapv(|v| v * v).sum()
}

#[test]
fn training_decreases_loss_identity_small() {
    // Simple 2->2 identity mapping under ReLU should converge somewhat
    let mut nn = NeuralNetwork::new(2, 2, vec![4, 4]);
    let data = vec![
        TrainingDatum {
            input: array![0.1, 0.2],
            output: array![0.1, 0.2],
        },
        TrainingDatum {
            input: array![0.3, 0.4],
            output: array![0.3, 0.4],
        },
        TrainingDatum {
            input: array![0.5, 0.6],
            output: array![0.5, 0.6],
        },
    ];

    // Baseline loss
    let before: f32 = data
        .iter()
        .map(|d| loss(&nn.run(&d.input), &d.output))
        .sum();

    // Train
    nn.train_data(50, data.clone());

    let after: f32 = data
        .iter()
        .map(|d| loss(&nn.run(&d.input), &d.output))
        .sum();

    assert!(
        after < before,
        "loss should decrease: before={before}, after={after}"
    );
}

#[test]
fn forward_shapes_match() {
    let nn = NeuralNetwork::new(3, 2, vec![5, 4]);
    let out = nn.run(&array![1.0, 2.0, 3.0]);
    assert_eq!(out.len(), 2);
}
