use ndarray::{Array1, Array2, Zip};
use ndarray_rand::{
    RandomExt,
    rand::Rng,
    rand_distr::{Bernoulli, Distribution, Normal, StandardNormal},
};
use serde::{Deserialize, Serialize};

use crate::types::Float;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct MutateConfig {
    /// Stddev of Gaussian noise added to parameters.
    pub sigma: Float,
    /// Probability an element is perturbed (1.0 = dense noise).
    pub prob: f64,
    /// Optional clamp range after mutation.
    pub clip: Option<(Float, Float)>,
    /// Chance to reinitialize an element (rare “reset” mutation).
    pub reset_chance: f64,
    /// Chance to insert a new layer
    pub insert_layer_chance: f64,
    /// Chance to add a neuron to an existing layer
    pub add_neuron_chance: f64,
    /// Chance to delete a neuron to an existing layer
    pub delete_neuron_chance: f64,
    pub min_neurons: usize,
}

impl Default for MutateConfig {
    fn default() -> Self {
        Self {
            sigma: 0.02,
            prob: 0.30,
            clip: Some((-1.0, 1.0)), // None for ReLU/linear nets
            reset_chance: 0.001,
            insert_layer_chance: 0.001,
            add_neuron_chance: 0.01,
            delete_neuron_chance: 0.005,
            min_neurons: 2,
        }
    }
}
#[allow(clippy::too_many_lines, clippy::missing_panics_doc)]
pub fn mutate_matrix<R>(m: &Array2<Float>, rng: &mut R, cfg: &MutateConfig) -> Array2<Float>
where
    R: Rng + ?Sized,
    StandardNormal: Distribution<Float>,
{
    let shape = m.raw_dim();
    let mut out = m.clone();

    // Precompute random fields as needed
    let noise = Array2::random_using(shape, Normal::new(0., cfg.sigma).unwrap(), rng);
    let fresh = if cfg.reset_chance > 0.0 {
        Some(Array2::random_using(
            shape,
            Normal::new(0., cfg.sigma).unwrap(),
            rng,
        ))
    } else {
        None
    };

    let mask = if cfg.prob < 1.0 {
        Some(Array2::random_using(
            shape,
            Bernoulli::new(cfg.prob).unwrap(),
            rng,
        ))
    } else {
        None
    };

    let reset_mask = if cfg.reset_chance > 0.0 {
        Some(Array2::random_using(
            shape,
            Bernoulli::new(cfg.reset_chance).unwrap(),
            rng,
        ))
    } else {
        None
    };

    // Single fused pass over all elements
    let lohi = cfg.clip;
    match (mask.as_ref(), reset_mask.as_ref(), fresh.as_ref()) {
        (Some(mask), Some(rm), Some(fresh)) => {
            let z = Zip::from(&mut out).and(&noise).and(mask).and(rm).and(fresh);
            let f = |o: &mut Float, &n, &use_noise, &do_reset, &fresh_v| {
                if do_reset {
                    *o = fresh_v;
                } else if use_noise {
                    *o += n;
                }
                if let Some((lo, hi)) = lohi {
                    *o = o.max(lo).min(hi);
                }
            };
            #[cfg(not(feature = "rayon"))]
            {
                z.for_each(f);
            }
            #[cfg(feature = "rayon")]
            {
                z.par_for_each(f);
            }
        }
        (Some(mask), None, None) => {
            let z = Zip::from(&mut out).and(&noise).and(mask);
            let f = |o: &mut Float, &n, &use_noise| {
                if use_noise {
                    *o += n;
                }
                if let Some((lo, hi)) = lohi {
                    *o = o.max(lo).min(hi);
                }
            };
            #[cfg(not(feature = "rayon"))]
            {
                z.for_each(f);
            }
            #[cfg(feature = "rayon")]
            {
                z.par_for_each(f);
            }
        }
        (None, Some(rm), Some(fresh)) => {
            let z = Zip::from(&mut out).and(&noise).and(rm).and(fresh);
            let f = |o: &mut Float, &n, &do_reset, &fresh_v| {
                if do_reset {
                    *o = fresh_v;
                } else {
                    *o += n;
                }
                if let Some((lo, hi)) = lohi {
                    *o = o.max(lo).min(hi);
                }
            };
            #[cfg(not(feature = "rayon"))]
            {
                z.for_each(f);
            }
            #[cfg(feature = "rayon")]
            {
                z.par_for_each(f);
            }
        }
        _ => {
            let z = Zip::from(&mut out).and(&noise);
            let f = |o: &mut Float, &n| {
                *o += n;
                if let Some((lo, hi)) = lohi {
                    *o = o.max(lo).min(hi);
                }
            };
            #[cfg(not(feature = "rayon"))]
            {
                z.for_each(f);
            }
            #[cfg(feature = "rayon")]
            {
                z.par_for_each(f);
            }
        }
    }

    out
}

#[allow(clippy::too_many_lines, clippy::missing_panics_doc)]
pub fn mutate_vector<R>(v: &Array1<Float>, rng: &mut R, cfg: &MutateConfig) -> Array1<Float>
where
    R: Rng + ?Sized,
    StandardNormal: Distribution<Float>,
{
    let shape = v.raw_dim();
    let mut out = v.clone();

    // Precompute fields
    let noise = Array1::random_using(shape, Normal::new(0., cfg.sigma).unwrap(), rng);
    let fresh = if cfg.reset_chance > 0.0 {
        Some(Array1::random_using(
            shape,
            Normal::new(0., cfg.sigma).unwrap(),
            rng,
        ))
    } else {
        None
    };

    let mask = if cfg.prob < 1.0 {
        Some(Array1::random_using(
            shape,
            Bernoulli::new(cfg.prob).unwrap(),
            rng,
        ))
    } else {
        None
    };

    let reset_mask = if cfg.reset_chance > 0.0 {
        Some(Array1::random_using(
            shape,
            Bernoulli::new(cfg.reset_chance).unwrap(),
            rng,
        ))
    } else {
        None
    };

    // Fused element pass
    let lohi = cfg.clip;
    match (mask.as_ref(), reset_mask.as_ref(), fresh.as_ref()) {
        (Some(mask), Some(rm), Some(fresh)) => {
            let z = Zip::from(&mut out).and(&noise).and(mask).and(rm).and(fresh);
            let f = |o: &mut Float, &n, &use_noise, &do_reset, &fresh_v| {
                if do_reset {
                    *o = fresh_v;
                } else if use_noise {
                    *o += n;
                }
                if let Some((lo, hi)) = lohi {
                    *o = o.max(lo).min(hi);
                }
            };
            #[cfg(not(feature = "rayon"))]
            {
                z.for_each(f);
            }
            #[cfg(feature = "rayon")]
            {
                z.par_for_each(f);
            }
        }
        (Some(mask), None, None) => {
            let z = Zip::from(&mut out).and(&noise).and(mask);
            let f = |o: &mut Float, &n, &use_noise| {
                if use_noise {
                    *o += n;
                }
                if let Some((lo, hi)) = lohi {
                    *o = o.max(lo).min(hi);
                }
            };
            #[cfg(not(feature = "rayon"))]
            {
                z.for_each(f);
            }
            #[cfg(feature = "rayon")]
            {
                z.par_for_each(f);
            }
        }
        (None, Some(rm), Some(fresh)) => {
            let z = Zip::from(&mut out).and(&noise).and(rm).and(fresh);
            let f = |o: &mut Float, &n, &do_reset, &fresh_v| {
                if do_reset {
                    *o = fresh_v;
                } else {
                    *o += n;
                }
                if let Some((lo, hi)) = lohi {
                    *o = o.max(lo).min(hi);
                }
            };
            #[cfg(not(feature = "rayon"))]
            {
                z.for_each(f);
            }
            #[cfg(feature = "rayon")]
            {
                z.par_for_each(f);
            }
        }
        _ => {
            let z = Zip::from(&mut out).and(&noise);
            let f = |o: &mut Float, &n| {
                *o += n;
                if let Some((lo, hi)) = lohi {
                    *o = o.max(lo).min(hi);
                }
            };
            #[cfg(not(feature = "rayon"))]
            {
                z.for_each(f);
            }
            #[cfg(feature = "rayon")]
            {
                z.par_for_each(f);
            }
        }
    }

    out
}
