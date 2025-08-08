use std::sync::{Arc, Mutex};

use ndarray_rand::rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{mutate::MutateConfig, neural_network::NeuralNetwork, types::Float};

pub struct Score(pub Float);
type NNScored = (NeuralNetwork, Score);

pub struct NNGroup {
    neural_networks: Box<[Arc<Mutex<NNScored>>]>,
    config: MutateConfig,
    alpha: Float,
    percent_survivors: Float,
}

type ScoredNN = (NeuralNetwork, Score);

impl Default for NNGroup {
    fn default() -> Self {
        Self {
            neural_networks: Box::default(),
            config: MutateConfig::default(),
            alpha: 0.2,
            percent_survivors: 0.5,
        }
    }
}
impl NNGroup {
    #[must_use]
    pub const fn new(
        neural_networks: Box<[Arc<Mutex<ScoredNN>>]>,
        config: MutateConfig,
        alpha: Float,
        percent_survivors: Float,
    ) -> Self {
        Self {
            neural_networks,
            config,
            alpha,
            percent_survivors,
        }
    }
    #[must_use]
    pub fn get_neural_networks(&self) -> &[Arc<Mutex<ScoredNN>>] {
        &self.neural_networks
    }
    #[allow(clippy::missing_panics_doc)]
    pub fn mutate(&mut self) {
        let n = self.neural_networks.len();
        if n == 0 {
            return;
        }

        let mut scores: Vec<(usize, Float)> = (0..n)
            .map(|i| {
                let g = self.neural_networks[i].lock().unwrap();
                // higher score is better
                (i, g.1.0)
            })
            .collect();

        // ----- 1) Sort indices by score best→worst -----
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().reverse());
        let idx_best: Vec<usize> = scores.iter().map(|(i, _)| *i).collect();

        // ----- 2) Decide how many survive -----
        let target_survivors = {
            #[allow(clippy::cast_sign_loss)]
            let m = (self.percent_survivors as Float * n as Float).round() as usize;
            m.max(1).min(n) // <- was .min(1); use .max(1)
        };

        // ----- 3) Exponential-by-rank weights (by position k=0..n-1) -----
        let alpha = self.alpha as Float;
        let weights_by_rank: Vec<Float> = (0..n).map(|k| (-alpha * (k as Float)).exp()).collect();
        let sum_w: Float = weights_by_rank.iter().sum();
        // Scale to hit expected survivors
        let scale = target_survivors as Float / sum_w;
        let probs_by_rank: Vec<Float> = weights_by_rank
            .iter()
            .map(|w| (w * scale).clamp(0.0, 1.0))
            .collect();

        // ----- 4) Sample survivors in RANK space -----
        let mut rng = StdRng::seed_from_u64(42); // or pass rng in for real randomness
        let mut survivors_rank: Vec<usize> = vec![];

        for (i, &prob) in probs_by_rank.iter().enumerate() {
            if rng.gen_bool(f64::from(prob)) {
                survivors_rank.push(i);
            }
        }

        // Adjust to exactly target_survivors (fill or trim) using roulette on weights
        match survivors_rank.len().cmp(&target_survivors) {
            std::cmp::Ordering::Less => {
                let mut pool: Vec<usize> = (0..n).filter(|k| !survivors_rank.contains(k)).collect();
                while survivors_rank.len() < target_survivors && !pool.is_empty() {
                    // roulette on weights_by_rank
                    let total: Float = pool.iter().map(|&k| weights_by_rank[k]).sum();
                    let mut t = rng.r#gen::<Float>() * total;
                    let mut picked_pos = 0usize;
                    for (pi, &k) in pool.iter().enumerate() {
                        t -= weights_by_rank[k];
                        if t <= 0.0 {
                            picked_pos = pi;
                            break;
                        }
                    }
                    survivors_rank.push(pool.remove(picked_pos));
                }
            }
            std::cmp::Ordering::Equal => {}
            std::cmp::Ordering::Greater => {
                // drop lowest-weight survivors first
                survivors_rank
                    .sort_by(|&a, &b| weights_by_rank[b].partial_cmp(&weights_by_rank[a]).unwrap());
                survivors_rank.truncate(target_survivors);
            }
        }

        // Convert survivors from RANK space → ORIGINAL indices
        survivors_rank.sort_unstable(); // sort k so we can iterate cleanly
        let survivors: Vec<usize> = survivors_rank.iter().map(|&k| idx_best[k]).collect();

        // ----- 5) Build replacement list (dead slots) -----
        let survivor_set: std::collections::HashSet<usize> = survivors.iter().copied().collect();
        let dead: Vec<usize> = (0..n).filter(|i| !survivor_set.contains(i)).collect();

        // Parent selection weights *for survivors only* (still rank-based)
        let parent_weights: Vec<Float> =
            survivors_rank.iter().map(|&k| weights_by_rank[k]).collect();
        let parent_weight_sum: Float = parent_weights.iter().sum();

        // Helper: roulette pick a survivor index in ORIGINAL index space
        let pick_parent = |rng: &mut StdRng| -> usize {
            let mut t = rng.r#gen::<Float>() * parent_weight_sum;
            for (j, &k) in survivors_rank.iter().enumerate() {
                t -= parent_weights[j];
                if t <= 0.0 {
                    return idx_best[k]; // original index of chosen parent
                }
            }
            idx_best[*survivors_rank.last().unwrap()]
        };

        // ----- 6) Replace dead with mutated clones of survivors -----
        for slot in dead {
            let parent_idx = pick_parent(&mut rng);
            // Mutate clone
            let child = {
                let mut parent_nn = self.neural_networks[parent_idx].lock().unwrap();
                parent_nn.1 = Score(0.);
                parent_nn.0.clone()
            }
            .mutate(&self.config);

            // Reset score for fresh evaluation
            let new_score = Score(0.); // or Score::zero()

            // Write back into the dead slot
            if let Ok(mut guard) = self.neural_networks[slot].lock() {
                *guard = (child, new_score);
            }
        }

        // (Optional) If you want survivors physically compacted at the front:
        // for (dst, &src) in (0..survivors.len()).zip(survivors.iter()) {
        //     if dst != src {
        //         let clone_pair = self.neural_networks[src].lock().unwrap().clone();
        //         *self.neural_networks[dst].lock().unwrap() = clone_pair;
        //     }
        // }
    }
}
