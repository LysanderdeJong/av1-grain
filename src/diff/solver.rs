mod util;

use std::ops::{Add, AddAssign};

use anyhow::anyhow;
use arrayvec::ArrayVec;
use rayon::{prelude::*, scope};
use v_frame::{chroma::ChromaSubsampling, frame::Frame, plane::Plane};

use self::util::{
    extract_ar_row, extract_ar_row_with_alt, get_block_mean, get_noise_var, linsolve, multiply_mat,
};
use super::{BLOCK_SIZE, BLOCK_SIZE_SQUARED, NoiseStatus};
use crate::{
    DEFAULT_GRAIN_SEED, GrainTableSegment, NUM_UV_COEFFS, NUM_UV_POINTS, NUM_Y_COEFFS,
    NUM_Y_POINTS,
    diff::solver::util::normalized_cross_correlation,
    profile::{self, MetricId},
    util::{get_dbg, get_dbg_mut},
};

const LOW_POLY_NUM_PARAMS: usize = 3;
const NOISE_MODEL_LAG: usize = 3;
const BLOCK_NORMALIZATION: f64 = 255.0f64;
const INV_BLOCK_NORMALIZATION_SQUARED: f64 = 1.0f64 / (BLOCK_NORMALIZATION * BLOCK_NORMALIZATION);

#[derive(Debug, Clone)]
pub(super) struct FlatBlockFinder {
    a: Box<[f64]>,
    a_t_a_inv: [f64; LOW_POLY_NUM_PARAMS * LOW_POLY_NUM_PARAMS],
}

impl FlatBlockFinder {
    #[must_use]
    pub fn new() -> Self {
        let mut eqns = EquationSystem::new(LOW_POLY_NUM_PARAMS);
        let mut a_t_a_inv = [0.0f64; LOW_POLY_NUM_PARAMS * LOW_POLY_NUM_PARAMS];
        let mut a = vec![0.0f64; LOW_POLY_NUM_PARAMS * BLOCK_SIZE_SQUARED];

        let bs_half = (BLOCK_SIZE / 2) as f64;
        (0..BLOCK_SIZE).for_each(|y| {
            let yd = (y as f64 - bs_half) / bs_half;
            (0..BLOCK_SIZE).for_each(|x| {
                let xd = (x as f64 - bs_half) / bs_half;
                let coords = [yd, xd, 1.0f64];
                let row = y * BLOCK_SIZE + x;
                *get_dbg_mut(&mut a, LOW_POLY_NUM_PARAMS * row) = yd;
                *get_dbg_mut(&mut a, LOW_POLY_NUM_PARAMS * row + 1) = xd;
                *get_dbg_mut(&mut a, LOW_POLY_NUM_PARAMS * row + 2) = 1.0f64;

                (0..LOW_POLY_NUM_PARAMS).for_each(|i| {
                    (0..LOW_POLY_NUM_PARAMS).for_each(|j| {
                        *get_dbg_mut(&mut eqns.a, LOW_POLY_NUM_PARAMS * i + j) +=
                            *get_dbg(&coords, i) * *get_dbg(&coords, j);
                    });
                });
            });
        });

        // Lazy inverse using existing equation solver.
        (0..LOW_POLY_NUM_PARAMS).for_each(|i| {
            eqns.b.fill(0.0f64);
            *get_dbg_mut(&mut eqns.b, i) = 1.0f64;
            eqns.solve();

            (0..LOW_POLY_NUM_PARAMS).for_each(|j| {
                *get_dbg_mut(&mut a_t_a_inv, j * LOW_POLY_NUM_PARAMS + i) = *get_dbg(&eqns.x, j);
            });
        });

        FlatBlockFinder {
            a: a.into_boxed_slice(),
            a_t_a_inv,
        }
    }

    // The gradient-based features used in this code are based on:
    //  A. Kokaram, D. Kelly, H. Denman and A. Crawford, "Measuring noise
    //  correlation for improved video denoising," 2012 19th, ICIP.
    // The thresholds are more lenient to allow for correct grain modeling
    // in extreme cases.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn run(&self, plane: &Plane<u8>) -> (Vec<u8>, usize) {
        const TRACE_THRESHOLD: f64 = 0.15f64 / BLOCK_SIZE_SQUARED as f64;
        const RATIO_THRESHOLD: f64 = 1.25f64;
        const NORM_THRESHOLD: f64 = 0.08f64 / BLOCK_SIZE_SQUARED as f64;
        const VAR_THRESHOLD: f64 = 0.005f64 / BLOCK_SIZE_SQUARED as f64;

        // The following weights are used to combine the above features to give
        // a sigmoid score for flatness. If the input was normalized to [0,100]
        // the magnitude of these values would be close to 1 (e.g., weights
        // corresponding to variance would be a factor of 10000x smaller).
        const VAR_WEIGHT: f64 = -6682f64;
        const RATIO_WEIGHT: f64 = -0.2056f64;
        const TRACE_WEIGHT: f64 = 13087f64;
        const NORM_WEIGHT: f64 = -12434f64;
        const OFFSET: f64 = 2.5694f64;

        let num_blocks_w = plane.width().get().div_ceil(BLOCK_SIZE);
        let num_blocks_h = plane.height().get().div_ceil(BLOCK_SIZE);
        let num_blocks = num_blocks_w * num_blocks_h;
        let mut flat_blocks = vec![0u8; num_blocks];
        let mut num_flat = 0;
        let mut scores = (0..num_blocks)
            .into_par_iter()
            .map(|index| {
                let by = index / num_blocks_w;
                let bx = index % num_blocks_w;
                let mut plane_result = [0.0f64; BLOCK_SIZE_SQUARED];
                let mut block_result = [0.0f64; BLOCK_SIZE_SQUARED];

                // Compute gradient covariance matrix.
                let mut gxx = 0f64;
                let mut gxy = 0f64;
                let mut gyy = 0f64;
                let mut var = 0f64;
                let mut mean = 0f64;

                self.extract_block(
                    plane,
                    bx * BLOCK_SIZE,
                    by * BLOCK_SIZE,
                    &mut plane_result,
                    &mut block_result,
                );
                for yi in 1..(BLOCK_SIZE - 1) {
                    for xi in 1..(BLOCK_SIZE - 1) {
                        // SAFETY: We know the size of `block_result` and that we cannot exceed the bounds of it
                        unsafe {
                            let result_ptr = block_result.as_ptr().add(yi * BLOCK_SIZE + xi);

                            let gx = (*result_ptr.add(1) - *result_ptr.sub(1)) / 2f64;
                            let gy =
                                (*result_ptr.add(BLOCK_SIZE) - *result_ptr.sub(BLOCK_SIZE)) / 2f64;
                            gxx += gx * gx;
                            gxy += gx * gy;
                            gyy += gy * gy;

                            let block_val = *result_ptr;
                            mean += block_val;
                            var += block_val * block_val;
                        }
                    }
                }
                let block_size_norm_factor = (BLOCK_SIZE - 2).pow(2) as f64;
                mean /= block_size_norm_factor;

                // Normalize gradients by block_size.
                gxx /= block_size_norm_factor;
                gxy /= block_size_norm_factor;
                gyy /= block_size_norm_factor;
                var = mean.mul_add(-mean, var / block_size_norm_factor);

                let trace = gxx + gyy;
                let det = gxx.mul_add(gyy, -gxy.powi(2));
                let e_sub = (trace.mul_add(trace, -4f64 * det)).max(0.).sqrt();
                let e1 = f64::midpoint(trace, e_sub);
                let e2 = (trace - e_sub) / 2.0f64;
                // Spectral norm
                let norm = e1;
                let ratio = e1 / e2.max(1.0e-6_f64);
                let is_flat = trace < TRACE_THRESHOLD
                    && ratio < RATIO_THRESHOLD
                    && norm < NORM_THRESHOLD
                    && var > VAR_THRESHOLD;

                let sum_weights = NORM_WEIGHT.mul_add(
                    norm,
                    TRACE_WEIGHT.mul_add(
                        trace,
                        VAR_WEIGHT.mul_add(var, RATIO_WEIGHT.mul_add(ratio, OFFSET)),
                    ),
                );
                // clamp the value to [-25.0, 100.0] to prevent overflow
                let sum_weights = sum_weights.clamp(-25.0f64, 100.0f64);
                let score = (1.0f64 / (1.0f64 + (-sum_weights).exp())) as f32;

                (
                    index,
                    if is_flat { 255 } else { 0 },
                    IndexAndScore {
                        score: if var > VAR_THRESHOLD { score } else { 0f32 },
                        index,
                    },
                )
            })
            .collect::<Vec<_>>();

        for (index, flat_block, _) in &scores {
            *get_dbg_mut(&mut flat_blocks, *index) = *flat_block;
            if *flat_block != 0 {
                num_flat += 1;
            }
        }
        let mut scores = scores
            .drain(..)
            .map(|(_, _, score)| score)
            .collect::<Vec<_>>();

        scores.sort_unstable_by(|a, b| a.score.partial_cmp(&b.score).expect("Shouldn't be NaN"));

        let top_nth_percentile = num_blocks * 90 / 100;
        let score_threshold = get_dbg(&scores, top_nth_percentile).score;
        for score in &scores {
            if score.score >= score_threshold {
                let block_ref = get_dbg_mut(&mut flat_blocks, score.index);
                if *block_ref == 0 {
                    num_flat += 1;
                }
                *block_ref |= 1;
            }
        }

        (flat_blocks, num_flat)
    }

    fn extract_block(
        &self,
        plane: &Plane<u8>,
        offset_x: usize,
        offset_y: usize,
        plane_result: &mut [f64; BLOCK_SIZE_SQUARED],
        block_result: &mut [f64; BLOCK_SIZE_SQUARED],
    ) {
        let mut plane_coords = [0f64; LOW_POLY_NUM_PARAMS];
        let mut a_t_a_inv_b = [0f64; LOW_POLY_NUM_PARAMS];
        let plane_origin = get_dbg(plane.data(), plane.data_origin()..);
        let width = plane.width().get();
        let height = plane.height().get();
        let stride = plane.geometry().stride.get();

        if offset_x + BLOCK_SIZE <= width && offset_y + BLOCK_SIZE <= height {
            for yi in 0..BLOCK_SIZE {
                let row_start = (offset_y + yi) * stride + offset_x;
                let row = get_dbg(plane_origin, row_start..row_start + BLOCK_SIZE);
                let out_row = get_dbg_mut(block_result, yi * BLOCK_SIZE..(yi + 1) * BLOCK_SIZE);
                for (out, pixel) in out_row.iter_mut().zip(row.iter()) {
                    *out = f64::from(*pixel) / BLOCK_NORMALIZATION;
                }
            }
        } else {
            for yi in 0..BLOCK_SIZE {
                let y = (offset_y + yi).clamp(0, height - 1);
                for xi in 0..BLOCK_SIZE {
                    let x = (offset_x + xi).clamp(0, width - 1);
                    *get_dbg_mut(block_result, yi * BLOCK_SIZE + xi) =
                        f64::from(*get_dbg(plane_origin, y * stride + x)) / BLOCK_NORMALIZATION;
                }
            }
        }

        multiply_mat(
            block_result,
            &self.a,
            &mut a_t_a_inv_b,
            1,
            BLOCK_SIZE_SQUARED,
            LOW_POLY_NUM_PARAMS,
        );
        multiply_mat(
            &self.a_t_a_inv,
            &a_t_a_inv_b,
            &mut plane_coords,
            LOW_POLY_NUM_PARAMS,
            LOW_POLY_NUM_PARAMS,
            1,
        );
        multiply_mat(
            &self.a,
            &plane_coords,
            plane_result,
            BLOCK_SIZE_SQUARED,
            LOW_POLY_NUM_PARAMS,
            1,
        );

        for (block_res, plane_res) in block_result.iter_mut().zip(plane_result.iter()) {
            *block_res -= *plane_res;
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct IndexAndScore {
    pub index: usize,
    pub score: f32,
}

/// Wrapper of data required to represent linear system of eqns and soln.
#[derive(Debug, Clone)]
struct EquationSystem {
    a: Vec<f64>,
    b: Vec<f64>,
    x: Vec<f64>,
    n: usize,
}

impl EquationSystem {
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            a: vec![0.0f64; n * n],
            b: vec![0.0f64; n],
            x: vec![0.0f64; n],
            n,
        }
    }

    pub fn solve(&mut self) -> bool {
        let n = self.n;
        let mut a = self.a.clone();
        let mut b = self.b.clone();

        linsolve(n, &mut a, self.n, &mut b, &mut self.x)
    }

    pub fn set_chroma_coefficient_fallback_solution(&mut self) {
        const TOLERANCE: f64 = 1.0e-6f64;
        let last = self.n - 1;
        // Set all of the AR coefficients to zero, but try to solve for correlation
        // with the luma channel
        self.x.fill(0f64);
        if get_dbg(&self.a, last * self.n + last).abs() > TOLERANCE {
            *get_dbg_mut(&mut self.x, last) =
                *get_dbg(&self.b, last) / *get_dbg(&self.a, last * self.n + last);
        }
    }

    pub fn copy_from(&mut self, other: &Self) {
        assert_eq!(self.n, other.n);
        self.a.copy_from_slice(&other.a);
        self.x.copy_from_slice(&other.x);
        self.b.copy_from_slice(&other.b);
    }

    pub fn clear(&mut self) {
        self.a.fill(0f64);
        self.b.fill(0f64);
        self.x.fill(0f64);
    }
}

impl Add<&EquationSystem> for EquationSystem {
    type Output = EquationSystem;

    fn add(mut self, addend: &EquationSystem) -> Self::Output {
        self += addend;
        self
    }
}

impl AddAssign<&EquationSystem> for EquationSystem {
    fn add_assign(&mut self, rhs: &EquationSystem) {
        debug_assert_eq!(self.n, rhs.n);
        for (a, rhs_a) in self.a.iter_mut().zip(rhs.a.iter()) {
            *a += *rhs_a;
        }
        for (b, rhs_b) in self.b.iter_mut().zip(rhs.b.iter()) {
            *b += *rhs_b;
        }
    }
}

fn mirror_upper_triangle(eqns: &mut EquationSystem) {
    let n = eqns.n;
    for i in 0..n {
        for j in (i + 1)..n {
            *get_dbg_mut(&mut eqns.a, j * n + i) = *get_dbg(&eqns.a, i * n + j);
        }
    }
}

fn scale_equation_system(eqns: &mut EquationSystem, scale: f64) {
    for val in &mut eqns.a {
        *val *= scale;
    }
    for val in &mut eqns.b {
        *val *= scale;
    }
}

fn add_ar_observation(eqns: &mut EquationSystem, buffer: &[f64], val: f64) {
    match eqns.n {
        NUM_Y_COEFFS => add_ar_observation_n::<NUM_Y_COEFFS>(eqns, buffer, val),
        NUM_UV_COEFFS => add_ar_observation_n::<NUM_UV_COEFFS>(eqns, buffer, val),
        _ => unreachable!("unsupported AR coefficient count"),
    }
}

#[inline]
fn add_ar_observation_n<const N: usize>(eqns: &mut EquationSystem, buffer: &[f64], val: f64) {
    debug_assert_eq!(eqns.n, N);
    debug_assert!(buffer.len() >= N);
    debug_assert!(eqns.a.len() >= N * N);
    debug_assert!(eqns.b.len() >= N);
    for (i, row) in eqns.a.chunks_exact_mut(N).take(N).enumerate() {
        let buffer_i = buffer[i];
        for (a, buffer_j) in row[i..].iter_mut().zip(buffer[i..N].iter()) {
            *a += buffer_i * *buffer_j;
        }
        eqns.b[i] += buffer_i * val;
    }
}

/// Representation of a piecewise linear curve
///
/// Holds n points as (x, y) pairs, that store the curve.
struct NoiseStrengthLut {
    points: Vec<[f64; 2]>,
}

impl NoiseStrengthLut {
    #[must_use]
    pub fn new(num_bins: usize) -> Self {
        assert!(num_bins > 0);
        Self {
            points: vec![[0f64; 2]; num_bins],
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct NoiseModel {
    combined_state: [NoiseModelState; 3],
    latest_state: [NoiseModelState; 3],
    n: usize,
    coords: Vec<[isize; 2]>,
}

impl NoiseModel {
    #[must_use]
    pub fn new() -> Self {
        let n = Self::num_coeffs();
        let combined_state = [
            NoiseModelState::new(n),
            NoiseModelState::new(n + 1),
            NoiseModelState::new(n + 1),
        ];
        let latest_state = [
            NoiseModelState::new(n),
            NoiseModelState::new(n + 1),
            NoiseModelState::new(n + 1),
        ];
        let mut coords = Vec::new();

        let neg_lag = -(NOISE_MODEL_LAG as isize);
        for y in neg_lag..=0 {
            let max_x = if y == 0 {
                -1isize
            } else {
                NOISE_MODEL_LAG as isize
            };
            for x in neg_lag..=max_x {
                coords.push([x, y]);
            }
        }
        assert!(n == coords.len());

        Self {
            combined_state,
            latest_state,
            n,
            coords,
        }
    }

    pub fn update(
        &mut self,
        source: &Frame<u8>,
        denoised: &Frame<u8>,
        flat_blocks: &[u8],
    ) -> NoiseStatus {
        let num_blocks_w = source.y_plane.width().get().div_ceil(BLOCK_SIZE);
        let num_blocks_h = source.y_plane.height().get().div_ceil(BLOCK_SIZE);
        let mut y_model_different = false;

        // Clear the latest equation system
        for i in 0..3 {
            let state = get_dbg_mut(&mut self.latest_state, i);
            state.eqns.clear();
            state.num_observations = 0;
            state.strength_solver.clear();
        }

        // Check that we have enough flat blocks
        let flat_block_indices = flat_blocks
            .iter()
            .enumerate()
            .filter_map(|(index, block)| (*block > 0).then_some(index))
            .collect::<Vec<_>>();
        if flat_block_indices.len() <= 1 {
            return NoiseStatus::Error(anyhow!("Not enough flat blocks to update noise estimate"));
        }

        let frame_dims = (source.y_plane.width().get(), source.y_plane.height().get());
        if let Err(err) = profile::time(MetricId::UpdateLatestY, || {
            Self::update_latest_channel(
                0,
                self.n,
                &self.coords,
                &mut self.latest_state[0],
                &source.y_plane,
                &denoised.y_plane,
                None,
                None,
                frame_dims,
                flat_blocks,
                &flat_block_indices,
                num_blocks_w,
                num_blocks_h,
                0f64,
                None,
            )
        }) {
            return NoiseStatus::Error(err);
        }

        let channel_count = if source.subsampling == ChromaSubsampling::Monochrome {
            1
        } else {
            let source_u = source
                .u_plane
                .as_ref()
                .expect("unreachable due to subsampling check");
            let source_v = source
                .v_plane
                .as_ref()
                .expect("unreachable due to subsampling check");
            let denoised_u = denoised
                .u_plane
                .as_ref()
                .expect("unreachable due to subsampling check");
            let denoised_v = denoised
                .v_plane
                .as_ref()
                .expect("unreachable due to subsampling check");
            let (y_states, chroma_states) = self.latest_state.split_at_mut(1);
            let y_state = &y_states[0];
            let (cb_states, cr_states) = chroma_states.split_at_mut(1);
            let luma_gain = y_state.ar_gain;
            let luma_strength_solver = &y_state.strength_solver;
            let cb_result = profile::time(MetricId::UpdateLatestCb, || {
                Self::update_latest_channel(
                    1,
                    self.n,
                    &self.coords,
                    &mut cb_states[0],
                    source_u,
                    denoised_u,
                    Some(&source.y_plane),
                    Some(&denoised.y_plane),
                    frame_dims,
                    flat_blocks,
                    &flat_block_indices,
                    num_blocks_w,
                    num_blocks_h,
                    luma_gain,
                    Some(luma_strength_solver),
                )
            });
            let cr_result = profile::time(MetricId::UpdateLatestCr, || {
                Self::update_latest_channel(
                    2,
                    self.n,
                    &self.coords,
                    &mut cr_states[0],
                    source_v,
                    denoised_v,
                    Some(&source.y_plane),
                    Some(&denoised.y_plane),
                    frame_dims,
                    flat_blocks,
                    &flat_block_indices,
                    num_blocks_w,
                    num_blocks_h,
                    luma_gain,
                    Some(luma_strength_solver),
                )
            });
            if let Err(err) = cb_result.and(cr_result) {
                return NoiseStatus::Error(err);
            }
            3
        };

        if let Err(status) = profile::time(MetricId::CombineState, || {
            for channel in 0..channel_count {
                let is_chroma = channel > 0;
                // Check noise characteristics and return if error
                let is_different = self.is_different();
                let combined_state = get_dbg_mut(&mut self.combined_state, channel);
                if channel == 0 && combined_state.strength_solver.num_equations > 0 && is_different
                {
                    y_model_different = true;
                }

                if y_model_different {
                    continue;
                }

                combined_state.num_observations +=
                    get_dbg(&self.latest_state, channel).num_observations;
                combined_state.eqns += &get_dbg(&self.latest_state, channel).eqns;
                if !combined_state.ar_equation_system_solve(is_chroma) {
                    if is_chroma {
                        combined_state
                            .eqns
                            .set_chroma_coefficient_fallback_solution();
                    } else {
                        return Err(NoiseStatus::Error(anyhow!(
                            "Solving combined noise equation system failed on plane {}",
                            channel
                        )));
                    }
                }

                combined_state.strength_solver +=
                    &get_dbg(&self.latest_state, channel).strength_solver;

                if !combined_state.strength_solver.solve() {
                    return Err(NoiseStatus::Error(anyhow!(
                        "Failed to solve strength solver for combined state"
                    )));
                };
            }
            Ok(())
        }) {
            return status;
        }

        if y_model_different {
            return NoiseStatus::DifferentType;
        }

        NoiseStatus::Ok
    }

    #[allow(clippy::too_many_lines)]
    #[must_use]
    pub fn get_grain_parameters(&self, start_ts: u64, end_ts: u64) -> GrainTableSegment {
        // Both the domain and the range of the scaling functions in the film_grain
        // are normalized to 8-bit (e.g., they are implicitly scaled during grain
        // synthesis).
        let mut scaling_points_y = None;
        let mut scaling_points_cb = None;
        let mut scaling_points_cr = None;
        scope(|s| {
            s.spawn(|_| {
                scaling_points_y = Some(
                    self.combined_state[0]
                        .strength_solver
                        .fit_piecewise(NUM_Y_POINTS)
                        .points,
                );
            });
            s.spawn(|_| {
                scaling_points_cb = Some(
                    self.combined_state[1]
                        .strength_solver
                        .fit_piecewise(NUM_UV_POINTS)
                        .points,
                );
            });
            s.spawn(|_| {
                scaling_points_cr = Some(
                    self.combined_state[2]
                        .strength_solver
                        .fit_piecewise(NUM_UV_POINTS)
                        .points,
                );
            });
        });
        let scaling_points_y = scaling_points_y.expect("scope task should set luma scaling points");
        let scaling_points_cb = scaling_points_cb.expect("scope task should set Cb scaling points");
        let scaling_points_cr = scaling_points_cr.expect("scope task should set Cr scaling points");

        let mut max_scaling_value: f64 = 1.0e-4f64;
        for p in scaling_points_y
            .iter()
            .chain(scaling_points_cb.iter())
            .chain(scaling_points_cr.iter())
            .map(|p| p[1])
        {
            if p > max_scaling_value {
                max_scaling_value = p;
            }
        }

        // Scaling_shift values are in the range [8,11]
        let max_scaling_value_log2 =
            ((max_scaling_value.log2() + 1f64).floor() as u8).clamp(2u8, 5u8);
        let scale_factor = f64::from(1u32 << (8u8 - max_scaling_value_log2));
        let map_scaling_point = |p: [f64; 2]| {
            [
                (p[0] + 0.5f64) as u8,
                (scale_factor.mul_add(p[1], 0.5f64) as i32).clamp(0i32, 255i32) as u8,
            ]
        };

        let scaling_points_y: ArrayVec<_, NUM_Y_POINTS> = scaling_points_y
            .into_iter()
            .map(map_scaling_point)
            .collect();
        let scaling_points_cb: ArrayVec<_, NUM_UV_POINTS> = scaling_points_cb
            .into_iter()
            .map(map_scaling_point)
            .collect();
        let scaling_points_cr: ArrayVec<_, NUM_UV_POINTS> = scaling_points_cr
            .into_iter()
            .map(map_scaling_point)
            .collect();

        // Convert the ar_coeffs into 8-bit values
        let n_coeff = self.combined_state[0].eqns.n;
        let mut max_coeff = 1.0e-4f64;
        let mut min_coeff = 1.0e-4f64;
        let mut y_corr = [0f64; 2];
        let mut avg_luma_strength = 0f64;
        for c in 0..3 {
            let eqns = &get_dbg(&self.combined_state, c).eqns;
            for i in 0..n_coeff {
                let xi = *get_dbg(&eqns.x, i);
                if xi > max_coeff {
                    max_coeff = xi;
                }
                if xi < min_coeff {
                    min_coeff = xi;
                }
            }

            // Since the correlation between luma/chroma was computed in an already
            // scaled space, we adjust it in the un-scaled space.
            let solver = &get_dbg(&self.combined_state, c).strength_solver;
            // Compute a weighted average of the strength for the channel.
            let mut average_strength = 0f64;
            let mut total_weight = 0f64;
            for i in 0..solver.eqns.n {
                let mut w = 0f64;
                for j in 0..solver.eqns.n {
                    w += *get_dbg(&solver.eqns.a, i * solver.eqns.n + j);
                }
                w = w.sqrt();
                average_strength += *get_dbg(&solver.eqns.x, i) * w;
                total_weight += w;
            }
            if total_weight.abs() < f64::EPSILON {
                average_strength = 1f64;
            } else {
                average_strength /= total_weight;
            }
            if c == 0 {
                avg_luma_strength = average_strength;
            } else {
                let y_corr_cur = get_dbg_mut(&mut y_corr, c - 1);
                *y_corr_cur = avg_luma_strength * *get_dbg(&eqns.x, n_coeff) / average_strength;
                max_coeff = max_coeff.max(*y_corr_cur);
                min_coeff = min_coeff.min(*y_corr_cur);
            }
        }

        // Shift value: AR coeffs range (values 6-9)
        // 6: [-2, 2),  7: [-1, 1), 8: [-0.5, 0.5), 9: [-0.25, 0.25)
        let ar_coeff_shift = (7i32
            - (1.0f64 + max_coeff.log2().floor()).max((-min_coeff).log2().ceil()) as i32)
            .clamp(6i32, 9i32) as u8;
        let scale_ar_coeff = f64::from(1u16 << ar_coeff_shift);
        let ar_coeffs_y = self.get_ar_coeffs_y(n_coeff, scale_ar_coeff);
        let ar_coeffs_cb = self.get_ar_coeffs_uv(1, n_coeff, scale_ar_coeff, y_corr);
        let ar_coeffs_cr = self.get_ar_coeffs_uv(2, n_coeff, scale_ar_coeff, y_corr);

        GrainTableSegment {
            random_seed: if start_ts == 0 { DEFAULT_GRAIN_SEED } else { 0 },
            start_time: start_ts,
            end_time: end_ts,
            ar_coeff_lag: NOISE_MODEL_LAG as u8,
            scaling_points_y,
            scaling_points_cb,
            scaling_points_cr,
            scaling_shift: 5 + (8 - max_scaling_value_log2),
            ar_coeff_shift,
            ar_coeffs_y,
            ar_coeffs_cb,
            ar_coeffs_cr,
            // At the moment, the noise modeling code assumes that the chroma scaling
            // functions are a function of luma.
            cb_mult: 128,
            cb_luma_mult: 192,
            cb_offset: 256,
            cr_mult: 128,
            cr_luma_mult: 192,
            cr_offset: 256,
            chroma_scaling_from_luma: false,
            grain_scale_shift: 0,
            overlap_flag: true,
        }
    }

    pub fn save_latest(&mut self) {
        for c in 0..3 {
            let latest_state = get_dbg(&self.latest_state, c);
            let combined_state = get_dbg_mut(&mut self.combined_state, c);
            combined_state.eqns.copy_from(&latest_state.eqns);
            combined_state
                .strength_solver
                .eqns
                .copy_from(&latest_state.strength_solver.eqns);
            combined_state.strength_solver.num_equations =
                latest_state.strength_solver.num_equations;
            combined_state.num_observations = latest_state.num_observations;
            combined_state.ar_gain = latest_state.ar_gain;
        }
    }

    #[must_use]
    const fn num_coeffs() -> usize {
        let n = 2 * NOISE_MODEL_LAG + 1;
        (n * n) / 2
    }

    #[must_use]
    fn get_ar_coeffs_y(&self, n_coeff: usize, scale_ar_coeff: f64) -> ArrayVec<i8, NUM_Y_COEFFS> {
        assert!(n_coeff <= NUM_Y_COEFFS);
        let mut coeffs = ArrayVec::new();
        let eqns = &self.combined_state[0].eqns;
        for i in 0..n_coeff {
            let xi = *get_dbg(&eqns.x, i);
            coeffs.push(((scale_ar_coeff * xi).round() as i32).clamp(-128i32, 127i32) as i8);
        }
        coeffs
    }

    #[must_use]
    fn get_ar_coeffs_uv(
        &self,
        channel: usize,
        n_coeff: usize,
        scale_ar_coeff: f64,
        y_corr: [f64; 2],
    ) -> ArrayVec<i8, NUM_UV_COEFFS> {
        assert!(n_coeff <= NUM_Y_COEFFS);
        let mut coeffs = ArrayVec::new();
        let eqns = &get_dbg(&self.combined_state, channel).eqns;
        for i in 0..n_coeff {
            let xi = *get_dbg(&eqns.x, i);
            coeffs.push(((scale_ar_coeff * xi).round() as i32).clamp(-128i32, 127i32) as i8);
        }
        coeffs.push(
            ((scale_ar_coeff * *get_dbg(&y_corr, channel - 1)).round() as i32)
                .clamp(-128i32, 127i32) as i8,
        );
        coeffs
    }

    // Return true if the noise estimate appears to be different from the combined
    // (multi-frame) estimate. The difference is measured by checking whether the
    // AR coefficients have diverged (using a threshold on normalized cross
    // correlation), or whether the noise strength has changed.
    #[must_use]
    fn is_different(&self) -> bool {
        const COEFF_THRESHOLD: f64 = 0.9f64;
        const STRENGTH_THRESHOLD: f64 = 0.005f64;

        let latest = &self.latest_state[0];
        let combined = &self.combined_state[0];
        let corr = normalized_cross_correlation(&latest.eqns.x, &combined.eqns.x, combined.eqns.n);
        if corr < COEFF_THRESHOLD {
            return true;
        }

        let dx = 1.0f64 / latest.strength_solver.num_bins as f64;
        let latest_eqns = &latest.strength_solver.eqns;
        let combined_eqns = &combined.strength_solver.eqns;
        let mut diff = 0.0f64;
        let mut total_weight = 0.0f64;
        for j in 0..latest_eqns.n {
            let mut weight = 0.0f64;
            for i in 0..latest_eqns.n {
                weight += *get_dbg(&latest_eqns.a, i * latest_eqns.n + j);
            }
            weight = weight.sqrt();
            diff += weight * (*get_dbg(&latest_eqns.x, j) - *get_dbg(&combined_eqns.x, j)).abs();
            total_weight += weight;
        }

        diff * dx / total_weight > STRENGTH_THRESHOLD
    }

    #[allow(clippy::too_many_arguments)]
    fn update_latest_channel(
        channel: usize,
        model_n: usize,
        coords: &[[isize; 2]],
        state: &mut NoiseModelState,
        source: &Plane<u8>,
        denoised: &Plane<u8>,
        alt_source: Option<&Plane<u8>>,
        alt_denoised: Option<&Plane<u8>>,
        frame_dims: (usize, usize),
        flat_blocks: &[u8],
        flat_block_indices: &[usize],
        num_blocks_w: usize,
        num_blocks_h: usize,
        luma_gain: f64,
        luma_strength_solver: Option<&StrengthSolver>,
    ) -> anyhow::Result<()> {
        let is_chroma = channel > 0;
        let add_block_metric = match channel {
            0 => MetricId::AddBlockObservationsY,
            1 => MetricId::AddBlockObservationsCb,
            _ => MetricId::AddBlockObservationsCr,
        };
        profile::time(add_block_metric, || {
            Self::add_block_observations(
                model_n,
                coords,
                state,
                source,
                denoised,
                alt_source,
                alt_denoised,
                frame_dims,
                flat_blocks,
                flat_block_indices,
                num_blocks_w,
                num_blocks_h,
            );
        });

        if !state.ar_equation_system_solve(is_chroma) {
            if is_chroma {
                state.eqns.set_chroma_coefficient_fallback_solution();
            } else {
                return Err(anyhow!(
                    "Solving latest noise equation system failed on plane {}",
                    channel
                ));
            }
        }

        let add_noise_std_metric = match channel {
            0 => MetricId::AddNoiseStdY,
            1 => MetricId::AddNoiseStdCb,
            _ => MetricId::AddNoiseStdCr,
        };
        profile::time(add_noise_std_metric, || {
            Self::add_noise_std_observations(
                channel,
                model_n,
                state,
                source,
                denoised,
                alt_source,
                frame_dims,
                flat_blocks,
                flat_block_indices,
                num_blocks_w,
                num_blocks_h,
                luma_gain,
                luma_strength_solver,
            );
        });
        if !state.strength_solver.solve() {
            return Err(anyhow!("Failed to solve strength solver for latest state"));
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn add_block_observations(
        model_n: usize,
        coords: &[[isize; 2]],
        state: &mut NoiseModelState,
        source: &Plane<u8>,
        denoised: &Plane<u8>,
        alt_source: Option<&Plane<u8>>,
        alt_denoised: Option<&Plane<u8>>,
        frame_dims: (usize, usize),
        flat_blocks: &[u8],
        flat_block_indices: &[usize],
        num_blocks_w: usize,
        _num_blocks_h: usize,
    ) {
        let num_coords = model_n;
        let n = state.eqns.n;
        let block_w = BLOCK_SIZE / source.geometry().subsampling_x.get() as usize;
        let block_h = BLOCK_SIZE / source.geometry().subsampling_y.get() as usize;

        let dec = (
            source.geometry().subsampling_x.get() as usize >> 1,
            source.geometry().subsampling_y.get() as usize >> 1,
        );
        let stride = source.geometry().stride.get();
        let source_origin = get_dbg(source.data(), source.data_origin()..);
        let denoised_origin = get_dbg(denoised.data(), denoised.data_origin()..);
        let alt_stride = alt_source.map_or(0, |s| s.geometry().stride.get());
        let alt_source_origin = alt_source.map(|s| get_dbg(s.data(), s.data_origin()..));
        let alt_denoised_origin = alt_denoised.map(|s| get_dbg(s.data(), s.data_origin()..));
        let alt_origins = alt_source_origin.zip(alt_denoised_origin);
        let mut coord_offsets = [0isize; NUM_Y_COEFFS];
        for (offset, coord) in coord_offsets.iter_mut().zip(coords.iter()).take(num_coords) {
            *offset = coord[1] * stride as isize + coord[0];
        }
        const BLOCKS_PER_CHUNK: usize = 8;
        let observations = flat_block_indices
            .par_chunks(BLOCKS_PER_CHUNK)
            .map(|chunk| {
                let mut chunk_eqns = EquationSystem::new(n);
                let mut chunk_observations = 0usize;
                let mut eqns = EquationSystem::new(n);
                let mut buffer = [0f64; NUM_UV_COEFFS];

                for &block_index in chunk {
                    eqns.clear();
                    let mut num_observations = 0usize;
                    let by = block_index / num_blocks_w;
                    let bx = block_index % num_blocks_w;
                    let flat_block_index = by * num_blocks_w + bx;

                    let y_o = by * block_h;
                    let x_o = bx * block_w;
                    let y_start = if by > 0 && flat_blocks[flat_block_index - num_blocks_w] > 0 {
                        0
                    } else {
                        NOISE_MODEL_LAG
                    };
                    let x_start = if bx > 0 && flat_blocks[flat_block_index - 1] > 0 {
                        0
                    } else {
                        NOISE_MODEL_LAG
                    };
                    let y_end = ((frame_dims.1 >> dec.1) - by * block_h).min(block_h);
                    let x_end = ((frame_dims.0 >> dec.0) - bx * block_w - NOISE_MODEL_LAG).min(
                        if bx + 1 < num_blocks_w && flat_blocks[flat_block_index + 1] > 0 {
                            block_w
                        } else {
                            block_w - NOISE_MODEL_LAG
                        },
                    );

                    if let Some((alt_source_origin, alt_denoised_origin)) = alt_origins {
                        for y in y_start..y_end {
                            let row_index = (y + y_o) * stride + x_o;
                            for x in x_start..x_end {
                                let base_index = row_index + x;
                                let val = extract_ar_row_with_alt(
                                    &coord_offsets,
                                    num_coords,
                                    source_origin,
                                    denoised_origin,
                                    base_index,
                                    dec,
                                    alt_source_origin,
                                    alt_denoised_origin,
                                    alt_stride,
                                    x + x_o,
                                    y + y_o,
                                    &mut buffer,
                                );
                                add_ar_observation(&mut eqns, &buffer, val);
                                num_observations += 1;
                            }
                        }
                    } else {
                        for y in y_start..y_end {
                            let row_index = (y + y_o) * stride + x_o;
                            for x in x_start..x_end {
                                let base_index = row_index + x;
                                let val = extract_ar_row(
                                    &coord_offsets,
                                    num_coords,
                                    source_origin,
                                    denoised_origin,
                                    base_index,
                                    &mut buffer,
                                );
                                add_ar_observation(&mut eqns, &buffer, val);
                                num_observations += 1;
                            }
                        }
                    }
                    if num_observations > 0 {
                        scale_equation_system(&mut eqns, INV_BLOCK_NORMALIZATION_SQUARED);
                        mirror_upper_triangle(&mut eqns);
                        chunk_eqns += &eqns;
                        chunk_observations += num_observations;
                    }
                }
                (chunk_eqns, chunk_observations)
            })
            .collect::<Vec<_>>();

        let mut eqns = EquationSystem::new(n);
        let mut num_observations = 0usize;
        for (block_eqns, block_observations) in observations {
            eqns += &block_eqns;
            num_observations += block_observations;
        }

        state.eqns = eqns;
        state.num_observations = num_observations;
    }

    #[allow(clippy::too_many_arguments)]
    fn add_noise_std_observations(
        channel: usize,
        model_n: usize,
        state: &mut NoiseModelState,
        source: &Plane<u8>,
        denoised: &Plane<u8>,
        alt_source: Option<&Plane<u8>>,
        frame_dims: (usize, usize),
        _flat_blocks: &[u8],
        flat_block_indices: &[usize],
        num_blocks_w: usize,
        _num_blocks_h: usize,
        luma_gain: f64,
        luma_strength_solver: Option<&StrengthSolver>,
    ) {
        let num_coords = model_n;
        let noise_gain = state.ar_gain;
        let block_w = BLOCK_SIZE / source.geometry().subsampling_x.get() as usize;
        let block_h = BLOCK_SIZE / source.geometry().subsampling_y.get() as usize;
        let source_subsampling_x = source.geometry().subsampling_x.get() as usize;
        let source_subsampling_y = source.geometry().subsampling_y.get() as usize;
        let corr = if channel > 0 {
            *get_dbg(&state.eqns.x, num_coords)
        } else {
            0f64
        };

        let measurements = flat_block_indices
            .par_iter()
            .map(|&block_index| {
                let by = block_index / num_blocks_w;
                let bx = block_index % num_blocks_w;
                let y_o = by * block_h;
                let x_o = bx * block_w;
                let num_samples_h =
                    ((frame_dims.1 / source_subsampling_y) - by * block_h).min(block_h);
                let num_samples_w =
                    ((frame_dims.0 / source_subsampling_x) - bx * block_w).min(block_w);

                // Make sure that we have a reasonable amount of samples to consider the block.
                if num_samples_w * num_samples_h <= BLOCK_SIZE {
                    return None;
                }

                let block_mean = get_block_mean(
                    alt_source.unwrap_or(source),
                    frame_dims,
                    x_o << (source.geometry().subsampling_x.get() >> 1),
                    y_o << (source.geometry().subsampling_y.get() >> 1),
                );
                let noise_var = get_noise_var(
                    source,
                    denoised,
                    (
                        frame_dims.0 >> (source.geometry().subsampling_x.get() >> 1),
                        frame_dims.1 >> (source.geometry().subsampling_y.get() >> 1),
                    ),
                    x_o,
                    y_o,
                    block_w,
                    block_h,
                );
                // We want to remove the part of the noise that came from being
                // correlated with luma. Note that the noise solver for luma must
                // have already been run.
                let luma_strength = if channel > 0 {
                    luma_gain
                        * luma_strength_solver
                            .expect("luma strength solver required for chroma")
                            .get_value(block_mean)
                } else {
                    0f64
                };
                // Chroma noise:
                //    N(0, noise_var) = N(0, uncorr_var) + corr * N(0, luma_strength^2)
                // The uncorrelated component:
                //   uncorr_var = noise_var - (corr * luma_strength)^2
                // But don't allow fully correlated noise (hence the max), since the
                // synthesis cannot model it.
                let uncorr_std = (noise_var / 16f64)
                    .max((corr * luma_strength).mul_add(-(corr * luma_strength), noise_var))
                    .sqrt();
                Some((block_mean, uncorr_std / noise_gain))
            })
            .collect::<Vec<_>>();

        let mut strength_solver = StrengthSolver::new(state.strength_solver.num_bins);
        for (block_mean, adjusted_strength) in measurements.into_iter().flatten() {
            strength_solver.add_measurement(block_mean, adjusted_strength);
        }

        state.strength_solver = strength_solver;
    }
}

#[derive(Debug, Clone)]
struct NoiseModelState {
    eqns: EquationSystem,
    ar_gain: f64,
    num_observations: usize,
    strength_solver: StrengthSolver,
}

impl NoiseModelState {
    #[must_use]
    pub fn new(n: usize) -> Self {
        const NUM_BINS: usize = 20;

        Self {
            eqns: EquationSystem::new(n),
            ar_gain: 1.0f64,
            num_observations: 0usize,
            strength_solver: StrengthSolver::new(NUM_BINS),
        }
    }

    pub fn ar_equation_system_solve(&mut self, is_chroma: bool) -> bool {
        let ret = self.eqns.solve();
        self.ar_gain = 1.0f64;
        if !ret {
            return ret;
        }

        // Update the AR gain from the equation system as it will be used to fit
        // the noise strength as a function of intensity.  In the Yule-Walker
        // equations, the diagonal should be the variance of the correlated noise.
        // In the case of the least squares estimate, there will be some variability
        // in the diagonal. So use the mean of the diagonal as the estimate of
        // overall variance (this works for least squares or Yule-Walker formulation).
        let mut var = 0f64;
        let n_adjusted = self.eqns.n - usize::from(is_chroma);
        for i in 0..n_adjusted {
            var += *get_dbg(&self.eqns.a, i * self.eqns.n + i) / self.num_observations as f64;
        }
        var /= n_adjusted as f64;

        // Keep track of E(Y^2) = <b, x> + E(X^2)
        // In the case that we are using chroma and have an estimate of correlation
        // with luma we adjust that estimate slightly to remove the correlated bits by
        // subtracting out the last column of a scaled by our correlation estimate
        // from b. E(y^2) = <b - A(:, end)*x(end), x>
        let mut sum_covar = 0f64;
        for i in 0..n_adjusted {
            let mut bi = *get_dbg(&self.eqns.b, i);
            if is_chroma {
                bi -= *get_dbg(&self.eqns.a, i * self.eqns.n + n_adjusted)
                    * *get_dbg(&self.eqns.x, n_adjusted);
            }
            sum_covar += (bi * *get_dbg(&self.eqns.x, i)) / self.num_observations as f64;
        }

        // Now, get an estimate of the variance of uncorrelated noise signal and use
        // it to determine the gain of the AR filter.
        let noise_var = (var - sum_covar).max(1e-6f64);
        self.ar_gain = 1f64.max((var / noise_var).max(1e-6f64).sqrt());
        ret
    }
}

#[derive(Debug, Clone)]
struct StrengthSolver {
    eqns: EquationSystem,
    num_bins: usize,
    num_equations: usize,
    total: f64,
}

impl StrengthSolver {
    #[must_use]
    pub fn new(num_bins: usize) -> Self {
        Self {
            eqns: EquationSystem::new(num_bins),
            num_bins,
            num_equations: 0usize,
            total: 0f64,
        }
    }

    pub fn add_measurement(&mut self, block_mean: f64, noise_std: f64) {
        let bin = self.get_bin_index(block_mean);
        let bin_i0 = bin.floor() as usize;
        let bin_i1 = (self.num_bins - 1).min(bin_i0 + 1);
        let a = bin - bin_i0 as f64;
        let n = self.num_bins;
        let eqns = &mut self.eqns;
        *get_dbg_mut(&mut eqns.a, bin_i0 * n + bin_i0) += (1f64 - a).powi(2);
        *get_dbg_mut(&mut eqns.a, bin_i1 * n + bin_i0) += a * (1f64 - a);
        *get_dbg_mut(&mut eqns.a, bin_i1 * n + bin_i1) += a.powi(2);
        *get_dbg_mut(&mut eqns.a, bin_i0 * n + bin_i1) += (1f64 - a) * a;
        *get_dbg_mut(&mut eqns.b, bin_i0) += (1f64 - a) * noise_std;
        *get_dbg_mut(&mut eqns.b, bin_i1) += a * noise_std;
        self.total += noise_std;
        self.num_equations += 1;
    }

    pub fn solve(&mut self) -> bool {
        // Add regularization proportional to the number of constraints
        let n = self.num_bins;
        let alpha = 2f64 * self.num_equations as f64 / n as f64;

        // Do this in a non-destructive manner so it is not confusing to the caller
        let old_a = self.eqns.a.clone();
        for i in 0..n {
            let i_lo = i.saturating_sub(1);
            let i_hi = (n - 1).min(i + 1);
            *get_dbg_mut(&mut self.eqns.a, i * n + i_lo) -= alpha;
            *get_dbg_mut(&mut self.eqns.a, i * n + i) += 2f64 * alpha;
            *get_dbg_mut(&mut self.eqns.a, i * n + i_hi) -= alpha;
        }

        // Small regularization to give average noise strength
        let mean = self.total / self.num_equations as f64;
        for i in 0..n {
            *get_dbg_mut(&mut self.eqns.a, i * n + i) += 1f64 / 8192f64;
            *get_dbg_mut(&mut self.eqns.b, i) += mean / 8192f64;
        }
        let result = self.eqns.solve();
        self.eqns.a = old_a;
        result
    }

    #[must_use]
    pub fn fit_piecewise(&self, max_output_points: usize) -> NoiseStrengthLut {
        const TOLERANCE: f64 = 0.00625f64;

        let mut lut = NoiseStrengthLut::new(self.num_bins);
        for i in 0..self.num_bins {
            get_dbg_mut(&mut lut.points, i)[0] = self.get_center(i);
            get_dbg_mut(&mut lut.points, i)[1] = *get_dbg(&self.eqns.x, i);
        }

        let mut residual = vec![0.0f64; self.num_bins];
        self.update_piecewise_linear_residual(&lut, &mut residual, 0, self.num_bins);

        // Greedily remove points if there are too many or if it doesn't hurt local
        // approximation (never remove the end points)
        while lut.points.len() > 2 {
            let mut min_index = 1usize;
            for j in 1..(lut.points.len() - 1) {
                if *get_dbg(&residual, j) < *get_dbg(&residual, min_index) {
                    min_index = j;
                }
            }
            let dx =
                get_dbg(&lut.points, min_index + 1)[0] - get_dbg(&lut.points, min_index - 1)[0];
            let avg_residual = get_dbg(&residual, min_index) / dx;
            if lut.points.len() <= max_output_points && avg_residual > TOLERANCE {
                break;
            }

            lut.points.remove(min_index);
            self.update_piecewise_linear_residual(
                &lut,
                &mut residual,
                min_index - 1,
                min_index + 1,
            );
        }

        lut
    }

    #[must_use]
    pub fn get_value(&self, x: f64) -> f64 {
        let bin = self.get_bin_index(x);
        let bin_i0 = bin.floor() as usize;
        let bin_i1 = (self.num_bins - 1).min(bin_i0 + 1);
        let a = bin - bin_i0 as f64;
        (1f64 - a).mul_add(
            *get_dbg(&self.eqns.x, bin_i0),
            a * *get_dbg(&self.eqns.x, bin_i1),
        )
    }

    pub fn clear(&mut self) {
        self.eqns.clear();
        self.num_equations = 0;
        self.total = 0f64;
    }

    #[must_use]
    fn get_bin_index(&self, value: f64) -> f64 {
        let max = 255f64;
        let val = value.clamp(0f64, max);
        (self.num_bins - 1) as f64 * val / max
    }

    fn update_piecewise_linear_residual(
        &self,
        lut: &NoiseStrengthLut,
        residual: &mut [f64],
        start: usize,
        end: usize,
    ) {
        let dx = 255f64 / self.num_bins as f64;
        #[allow(clippy::needless_range_loop)]
        for i in start.max(1)..end.min(lut.points.len() - 1) {
            let lower =
                0usize.max(self.get_bin_index(get_dbg(&lut.points, i - 1)[0]).floor() as usize);
            let upper = (self.num_bins - 1)
                .min(self.get_bin_index(get_dbg(&lut.points, i + 1)[0]).ceil() as usize);
            let mut r = 0f64;
            for j in lower..=upper {
                let x = self.get_center(j);
                if x < get_dbg(&lut.points, i - 1)[0] || x >= get_dbg(&lut.points, i + 1)[0] {
                    continue;
                }

                let y = *get_dbg(&self.eqns.x, j);
                let a = (x - get_dbg(&lut.points, i - 1)[0])
                    / (get_dbg(&lut.points, i + 1)[0] - get_dbg(&lut.points, i - 1)[0]);
                let estimate_y = get_dbg(&lut.points, i - 1)[1]
                    .mul_add(1f64 - a, get_dbg(&lut.points, i + 1)[1] * a);
                r += (y - estimate_y).abs();
            }
            *get_dbg_mut(residual, i) = r * dx;
        }
    }

    #[must_use]
    fn get_center(&self, i: usize) -> f64 {
        let range = 255f64;
        let n = self.num_bins;
        i as f64 / (n - 1) as f64 * range
    }
}

impl Add<&StrengthSolver> for StrengthSolver {
    type Output = StrengthSolver;

    fn add(self, addend: &StrengthSolver) -> Self::Output {
        let mut dest = self;
        dest.eqns += &addend.eqns;
        dest.num_equations += addend.num_equations;
        dest.total += addend.total;
        dest
    }
}

impl AddAssign<&StrengthSolver> for StrengthSolver {
    fn add_assign(&mut self, rhs: &StrengthSolver) {
        self.eqns += &rhs.eqns;
        self.num_equations += rhs.num_equations;
        self.total += rhs.total;
    }
}
