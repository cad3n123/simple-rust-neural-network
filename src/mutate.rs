use ndarray::{Array1, Array2, Zip};
use ndarray_rand::{
    RandomExt,
    rand::Rng,
    rand_distr::{Bernoulli, Distribution, Normal, StandardNormal},
};

use crate::types::Float;

#[derive(Debug, Clone, Copy)]
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
            sigma: 0.02,       // gentle noise
            prob: 0.25,        // mutate ~25% of params
            clip: None,        // no clipping by default
            reset_chance: 0.0, // no resets by default
            insert_layer_chance: 0.01,
            add_neuron_chance: 0.02,
            delete_neuron_chance: 0.01,
            min_neurons: 2,
        }
    }
}
pub(crate) fn mutate_matrix<R>(m: &Array2<Float>, rng: &mut R, cfg: &MutateConfig) -> Array2<Float>
where
    R: Rng + ?Sized,
    StandardNormal: Distribution<Float>,
{
    let shape = m.raw_dim();
    let noise = Array2::random_using(shape, Normal::new(0., cfg.sigma).unwrap(), rng);
    let mut out = m.clone();

    if cfg.prob < 1.0 {
        let mask = Array2::random_using(shape, Bernoulli::new(cfg.prob).unwrap(), rng);
        let zip = Zip::from(&mut out).and(&noise).and(&mask);
        let for_each_closure = |o: &mut Float, &n, &use_it| {
            if use_it {
                *o += n;
            }
        };

        #[cfg(not(feature = "rayon"))]
        {
            zip.for_each(for_each_closure);
        }
        #[cfg(feature = "rayon")]
        {
            zip.par_for_each(for_each_closure);
        }
    } else {
        out = &out + &noise;
    }

    if cfg.reset_chance > 0.0 {
        let reset_mask =
            Array2::random_using(shape, Bernoulli::new(cfg.reset_chance).unwrap(), rng);
        let fresh = Array2::random_using(shape, Normal::new(0., cfg.sigma).unwrap(), rng);
        let zip = Zip::from(&mut out).and(&reset_mask).and(&fresh);
        let for_each_closure = |o: &mut Float, &rm, &f| {
            if rm {
                *o = f;
            }
        };

        #[cfg(not(feature = "rayon"))]
        {
            zip.for_each(for_each_closure);
        }
        #[cfg(feature = "rayon")]
        {
            zip.par_for_each(for_each_closure);
        }
    }

    if let Some((lo, hi)) = cfg.clip {
        #[cfg(not(feature = "rayon"))]
        {
            out.mapv_inplace(|x| x.max(lo).min(hi));
        }
        #[cfg(feature = "rayon")]
        {
            out.par_mapv_inplace(|x| x.max(lo).min(hi));
        }
    }

    out
}

pub(crate) fn mutate_vector<R>(v: &Array1<Float>, rng: &mut R, cfg: &MutateConfig) -> Array1<Float>
where
    R: Rng + ?Sized,
    StandardNormal: Distribution<Float>,
{
    let shape = v.raw_dim();
    let noise = Array1::random_using(shape, Normal::new(0., cfg.sigma).unwrap(), rng);

    let out = if cfg.prob < 1.0 {
        let mask_b = Array1::random_using(shape, Bernoulli::new(cfg.prob).unwrap(), rng);
        let mask = mask_b.mapv_into_any(|b| if b { 1. } else { 0. });
        v + &(noise * &mask)
    } else {
        v + &noise
    };

    let mut out = if cfg.reset_chance > 0.0 {
        let reset_mask_b =
            Array1::random_using(shape, Bernoulli::new(cfg.reset_chance).unwrap(), rng);
        let fresh = Array1::random_using(shape, Normal::new(0., cfg.sigma).unwrap(), rng);
        out.iter()
            .zip(reset_mask_b.iter())
            .zip(fresh.iter())
            .map(|((val, &do_reset), &fresh_val)| if do_reset { fresh_val } else { *val })
            .collect::<Array1<Float>>()
    } else {
        out
    };

    if let Some((lo, hi)) = cfg.clip {
        #[cfg(not(feature = "rayon"))]
        {
            out.mapv_inplace(|x| x.max(lo).min(hi));
        }
        #[cfg(feature = "rayon")]
        {
            out.par_mapv_inplace(|x| x.max(lo).min(hi));
        }
    }
    out
}
