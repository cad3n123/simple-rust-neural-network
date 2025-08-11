use std::sync::{Arc, Mutex};

use ndarray_rand::rand::{self, Rng, seq::SliceRandom, thread_rng};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[cfg(feature = "rayon")]
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
    },
    slice::ParallelSliceMut,
};

use crate::{mutate::MutateConfig, neural_network::NeuralNetwork, types::Float};

#[derive(Serialize, Deserialize)]
pub struct NNGroup {
    #[serde(
        deserialize_with = "deserialize_arc_mutex_box_slice",
        serialize_with = "serialize_arc_mutex_box_slice"
    )]
    pub neural_networks: Box<[Arc<Mutex<ScoredNN>>]>,
    pub config: MutateConfig,
    pub alpha: Float,
    pub percent_survivors: Float,
}
#[derive(Serialize, Deserialize, Clone)]
pub struct ScoredNN {
    pub nn: NeuralNetwork,
    pub score: Score,
}
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct Score(pub Float);
pub struct Replacement {
    pub child_index: usize,
    pub parent_index: usize,
}

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
    #[allow(clippy::missing_panics_doc, clippy::too_many_lines)]
    #[must_use]
    pub fn plan_replacements(&self) -> (Vec<usize>, Vec<Replacement>) {
        let n = self.neural_networks.len();
        if n == 0 {
            return (Vec::new(), Vec::new());
        }

        let mut rng = thread_rng();

        // ----- Scores (shuffled) -----
        #[cfg(not(feature = "rayon"))]
        let mut scores: Vec<(usize, Float)> = self
            .neural_networks
            .iter()
            .enumerate()
            .map(|(i, nn)| {
                let s = nn.lock().unwrap().score.0;
                (i, s)
            })
            .collect();

        #[cfg(feature = "rayon")]
        let mut scores: Vec<(usize, Float)> = self
            .neural_networks
            .par_iter()
            .enumerate()
            .map(|(i, nn)| {
                let s = nn.lock().unwrap().score.0;
                (i, s)
            })
            .collect();

        scores.shuffle(&mut rng);

        // ----- Sort best→worst -----
        let cmp = |a: &(usize, Float), b: &(usize, Float)| a.1.partial_cmp(&b.1).unwrap().reverse();

        #[cfg(not(feature = "rayon"))]
        scores.sort_unstable_by(cmp);
        #[cfg(feature = "rayon")]
        scores.par_sort_unstable_by(cmp);

        let idx_best: Vec<usize> = scores.iter().map(|(i, _)| *i).collect();

        // ----- Target survivors -----
        let target_survivors = {
            #[allow(clippy::cast_sign_loss)]
            let m = (self.percent_survivors as Float * n as Float).round() as usize;
            m.max(1).min(n)
        };

        // ----- Rank weights w_k = exp(-alpha * k) -----
        let alpha = self.alpha as Float;

        #[cfg(not(feature = "rayon"))]
        let weights_by_rank: Vec<Float> = (0..n).map(|k| (-alpha * (k as Float)).exp()).collect();
        #[cfg(feature = "rayon")]
        let weights_by_rank: Vec<Float> = (0..n)
            .into_par_iter()
            .map(|k| (-alpha * (k as Float)).exp())
            .collect();

        #[cfg(not(feature = "rayon"))]
        let sum_w: Float = weights_by_rank.iter().sum();
        #[cfg(feature = "rayon")]
        let sum_w: Float = weights_by_rank.par_iter().sum();

        // Scale to match expected survivors
        let scale = target_survivors as Float / sum_w;

        #[cfg(not(feature = "rayon"))]
        let probs_by_rank: Vec<Float> = weights_by_rank
            .iter()
            .map(|w| (w * scale).clamp(0.0, 1.0))
            .collect();
        #[cfg(feature = "rayon")]
        let probs_by_rank: Vec<Float> = weights_by_rank
            .par_iter()
            .map(|w| (w * scale).clamp(0.0, 1.0))
            .collect();

        // ----- Sample survivors in rank space -----
        let mut survivors_rank: Vec<usize> = Vec::new();
        for (k, &p) in probs_by_rank.iter().enumerate() {
            if rng.gen_bool(f64::from(p)) {
                survivors_rank.push(k);
            }
        }

        // Adjust to exactly target_survivors
        match survivors_rank.len().cmp(&target_survivors) {
            std::cmp::Ordering::Less => {
                let mut pool: Vec<usize> = (0..n).filter(|k| !survivors_rank.contains(k)).collect();
                while survivors_rank.len() < target_survivors && !pool.is_empty() {
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
            std::cmp::Ordering::Greater => {
                survivors_rank
                    .sort_by(|&a, &b| weights_by_rank[b].partial_cmp(&weights_by_rank[a]).unwrap());
                survivors_rank.truncate(target_survivors);
            }
            std::cmp::Ordering::Equal => {}
        }

        // Map rank→original
        survivors_rank.sort_unstable();
        let survivors: Vec<usize> = survivors_rank.iter().map(|&k| idx_best[k]).collect();

        // Dead slots are everything not in survivors
        let survivor_set: std::collections::HashSet<usize> = survivors.iter().copied().collect();
        let dead: Vec<usize> = (0..n).filter(|i| !survivor_set.contains(i)).collect();

        // Parent selection weights (for survivors only)
        let parent_weights: Vec<Float> =
            survivors_rank.iter().map(|&k| weights_by_rank[k]).collect();
        let parent_weight_sum: Float = parent_weights.iter().sum();

        // Roulette parent picker over survivors_rank
        let pick_parent = |rng: &mut rand::rngs::ThreadRng| -> usize {
            let mut t = rng.r#gen::<Float>() * parent_weight_sum;
            for (j, &k) in survivors_rank.iter().enumerate() {
                t -= parent_weights[j];
                if t <= 0.0 {
                    return idx_best[k]; // original index of chosen parent
                }
            }
            idx_best[*survivors_rank.last().unwrap()]
        };

        // Build the plan: (dead_slot, parent_idx)
        let mut plan: Vec<Replacement> = Vec::with_capacity(dead.len());
        for child_index in dead {
            let parent_index = pick_parent(&mut rng);
            plan.push(Replacement {
                child_index,
                parent_index,
            });
        }

        (survivors, plan)
    }

    /// Step 2: apply replacements.
    /// Uses the plan to write mutated children and resets survivor scores.
    #[allow(clippy::missing_panics_doc)]
    pub fn mutate_with_plan(&mut self, plan: &[Replacement], survivors: &[usize]) {
        // Replace each dead slot with a mutated clone of its parent
        for &Replacement {
            child_index,
            parent_index,
        } in plan
        {
            let child = {
                let parent_nn = self.neural_networks[parent_index].lock().unwrap();
                parent_nn.nn.clone()
            }
            .mutated(&self.config);

            if let Ok(mut guard) = self.neural_networks[child_index].lock() {
                *guard = ScoredNN {
                    nn: child,
                    score: Score(0.0),
                };
            }
        }

        // Reset scores of survivors for fresh evaluation
        for &i in survivors {
            if let Ok(mut guard) = self.neural_networks[i].lock() {
                guard.score = Score(0.0);
            }
        }
    }
    pub fn mutate(&mut self) {
        let (survivors, plan) = self.plan_replacements();
        self.mutate_with_plan(&plan, &survivors);
    }
}
#[allow(clippy::type_complexity)]
fn deserialize_arc_mutex_box_slice<'de, D>(
    deserializer: D,
) -> Result<Box<[Arc<Mutex<ScoredNN>>]>, D::Error>
where
    D: Deserializer<'de>,
{
    let raw: Vec<ScoredNN> = Vec::deserialize(deserializer)?;
    let wrapped: Vec<Arc<Mutex<ScoredNN>>> =
        raw.into_iter().map(|v| Arc::new(Mutex::new(v))).collect();
    Ok(wrapped.into_boxed_slice())
}
#[allow(clippy::borrowed_box)]
fn serialize_arc_mutex_box_slice<S>(
    value: &Box<[Arc<Mutex<ScoredNN>>]>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let tmp: Vec<ScoredNN> = value.iter().map(|a| a.lock().unwrap().clone()).collect();
    tmp.serialize(serializer)
}
