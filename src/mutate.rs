use ndarray::{Array1, Array2, ScalarOperand, Zip};
use ndarray_rand::{
    RandomExt,
    rand::Rng,
    rand_distr::{Bernoulli, Distribution, Normal, StandardNormal},
};
use num_traits::{Float, FromPrimitive};

#[derive(Debug, Clone, Copy)]
pub struct MutateConfig<T> {
    /// Stddev of Gaussian noise added to parameters.
    pub sigma: T,
    /// Probability an element is perturbed (1.0 = dense noise).
    pub prob: f64,
    /// Optional clamp range after mutation.
    pub clip: Option<(T, T)>,
    /// Chance to reinitialize an element (rare “reset” mutation).
    pub reset_chance: f64,
}

impl<T> Default for MutateConfig<T>
where
    T: num_traits::Float + num_traits::FromPrimitive,
{
    fn default() -> Self {
        Self {
            sigma: T::from_f64(0.02).unwrap(), // gentle noise
            prob: 0.25,                        // mutate ~25% of params
            clip: None,                        // no clipping by default
            reset_chance: 0.0,                 // no resets by default
        }
    }
}
pub(crate) fn mutate_matrix<T, R>(m: &Array2<T>, rng: &mut R, cfg: &MutateConfig<T>) -> Array2<T>
where
    T: Float + FromPrimitive + Send + Sync + 'static,
    R: Rng + ?Sized,
    StandardNormal: Distribution<T>,
{
    let shape = m.raw_dim();
    let noise = Array2::random_using(shape, Normal::new(T::zero(), cfg.sigma).unwrap(), rng);
    let mut out = m.clone();

    if cfg.prob < 1.0 {
        let mask = Array2::random_using(shape, Bernoulli::new(cfg.prob).unwrap(), rng);
        let zip = Zip::from(&mut out).and(&noise).and(&mask);
        let for_each_closure = |o: &mut T, &n, &use_it| {
            if use_it {
                *o = *o + n;
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
        let fresh = Array2::random_using(shape, Normal::new(T::zero(), cfg.sigma).unwrap(), rng);
        let zip = Zip::from(&mut out).and(&reset_mask).and(&fresh);
        let for_each_closure = |o: &mut T, &rm, &f| {
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

pub(crate) fn mutate_vector<T, R>(v: &Array1<T>, rng: &mut R, cfg: &MutateConfig<T>) -> Array1<T>
where
    T: Float + FromPrimitive + ScalarOperand + Sync + Send + 'static,
    R: Rng + ?Sized,
    StandardNormal: Distribution<T>,
{
    let shape = v.raw_dim();
    let noise = Array1::random_using(shape, Normal::new(T::zero(), cfg.sigma).unwrap(), rng);

    let out = if cfg.prob < 1.0 {
        let mask_b = Array1::random_using(shape, Bernoulli::new(cfg.prob).unwrap(), rng);
        let mask = mask_b.mapv_into_any(|b| if b { T::one() } else { T::zero() });
        v + &(noise * &mask)
    } else {
        v + &noise
    };

    let mut out = if cfg.reset_chance > 0.0 {
        let reset_mask_b =
            Array1::random_using(shape, Bernoulli::new(cfg.reset_chance).unwrap(), rng);
        let fresh = Array1::random_using(shape, Normal::new(T::zero(), cfg.sigma).unwrap(), rng);
        out.iter()
            .zip(reset_mask_b.iter())
            .zip(fresh.iter())
            .map(|((val, &do_reset), &fresh_val)| if do_reset { fresh_val } else { *val })
            .collect::<Array1<T>>()
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
