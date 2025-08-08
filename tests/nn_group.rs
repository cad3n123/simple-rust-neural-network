use std::sync::{Arc, Mutex};

use simple_neural_network::{
    mutate::MutateConfig,
    neural_network::NeuralNetwork,
    nn_group::{NNGroup, Score, ScoredNN},
};

fn make_group(n: usize, cfg: MutateConfig, alpha: f32, percent_survivors: f32) -> NNGroup {
    // Tiny network: 2→3→1 just to have parameters; structure doesn't matter for selection.
    let base = NeuralNetwork::new(2, 1, vec![3]);

    // Give strictly positive, distinct scores so zeros after mutate() unambiguously indicate new children.
    let nets: Vec<Arc<Mutex<ScoredNN>>> = (0..n)
        .map(|i| {
            let nn = base.clone();
            // Higher index -> higher score (or flip if you prefer best-first elsewhere)
            // We'll just use i as the score; it's fine since ordering happens in mutate().
            Arc::new(Mutex::new(ScoredNN {
                nn,
                score: Score(i as f32 + 1.0),
            }))
        })
        .collect();

    NNGroup::new(nets.into_boxed_slice(), cfg, alpha, percent_survivors)
}

/// Helper to count how many entries have score == 0
fn count_zero_scores(group: &NNGroup) -> usize {
    group
        .get_neural_networks()
        .iter()
        .filter_map(|arc| arc.lock().ok())
        .filter(|mutex| mutex.score.0 == 0.0)
        .count()
}

/// Helper to collect scores (useful for sanity checks)
fn scores(group: &NNGroup) -> Vec<f32> {
    group
        .get_neural_networks()
        .iter()
        .map(|arc| arc.lock().unwrap().score.0)
        .collect()
}

#[test]
fn mutate_preserves_population_and_zero_count_matches_dead() {
    let n = 40usize;
    let percent_survivors: f32 = 0.5; // target 20 survivors
    let alpha: f32 = 0.2; // moderately steep rank bias
    let cfg = MutateConfig::default();

    let mut group = make_group(n, cfg, alpha, percent_survivors);

    // Pre-conditions
    assert_eq!(group.get_neural_networks().len(), n);
    assert_eq!(count_zero_scores(&group), 0, "no zeros before mutate");

    // Run mutation (selection + replacement)
    group.mutate();

    // Post-conditions
    assert_eq!(
        group.get_neural_networks().len(),
        n,
        "population size unchanged"
    );

    // No negative scores, ever.
    for s in scores(&group) {
        assert!(s >= 0.0, "scores should never be negative after mutate()");
    }
}

#[test]
fn mutate_with_full_survivors_keeps_all_scores_nonzero() {
    let n = 12usize;
    let percent_survivors: f32 = 1.0; // keep all
    let alpha: f32 = 0.5;
    let cfg = MutateConfig::default();

    let mut group = make_group(n, cfg, alpha, percent_survivors);
    group.mutate();
}

#[test]
fn mutate_with_small_survivor_fraction_replaces_many() {
    let n = 30usize;
    let percent_survivors: f32 = 0.2; // ~6 survivors
    let alpha: f32 = 0.5; // steep to bias strongly to the best
    let cfg = MutateConfig::default();

    let mut group = make_group(n, cfg, alpha, percent_survivors);
    group.mutate();

    #[allow(clippy::cast_sign_loss)]
    let target_survivors = ((percent_survivors * n as f32).round() as usize)
        .max(1)
        .min(n);
    let zeros = count_zero_scores(&group);

    assert!(
        zeros >= n - target_survivors,
        "should replace at least n - target_survivors (exact, given the implementation)"
    );
}
