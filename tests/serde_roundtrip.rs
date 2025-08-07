use simple_neural_network::neural_network::NeuralNetwork;

#[test]
fn serde_roundtrip_network() {
    let net = NeuralNetwork::new(3, 2, vec![4, 5]);
    let s = serde_json::to_string(&net).expect("serialize");
    let net2: NeuralNetwork = serde_json::from_str(&s).expect("deserialize");
    assert_eq!(net.layer_count(), net2.layer_count());
    assert_eq!(net.get_layer_output_sizes(), net2.get_layer_output_sizes());
}
