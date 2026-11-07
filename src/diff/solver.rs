mod gram;
mod util;

use std::ops::{Add, AddAssign};

use anyhow::anyhow;
use arrayvec::ArrayVec;
use rayon::{prelude::*, scope};
use v_frame::{chroma::ChromaSubsampling, frame::Frame, plane::Plane};

use self::util::{DiffPlane, get_block_mean, get_noise_var, linsolve, multiply_mat};
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
/// Minimum colour grain strength (8-bit std) worth synthesising.
const CHROMA_FLOOR: f64 = 0.5f64;
/// Minimum colour grain strength relative to luma.
const CHROMA_SHARE_OF_LUMA: f64 = 0.15f64;
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
                *unsafe { a.get_unchecked_mut(LOW_POLY_NUM_PARAMS * row) } = yd;
                *unsafe { a.get_unchecked_mut(LOW_POLY_NUM_PARAMS * row + 1) } = xd;
                *unsafe { a.get_unchecked_mut(LOW_POLY_NUM_PARAMS * row + 2) } = 1.0f64;

                (0..LOW_POLY_NUM_PARAMS).for_each(|i| {
                    (0..LOW_POLY_NUM_PARAMS).for_each(|j| {
                        *unsafe { eqns.a.get_unchecked_mut(LOW_POLY_NUM_PARAMS * i + j) } +=
                            *unsafe { coords.get_unchecked(i) }
                                * *unsafe { coords.get_unchecked(j) };
                    });
                });
            });
        });

        // Lazy inverse using existing equation solver.
        (0..LOW_POLY_NUM_PARAMS).for_each(|i| {
            eqns.b.fill(0.0f64);
            *unsafe { eqns.b.get_unchecked_mut(i) } = 1.0f64;
            eqns.solve();

            (0..LOW_POLY_NUM_PARAMS).for_each(|j| {
                *unsafe { a_t_a_inv.get_unchecked_mut(j * LOW_POLY_NUM_PARAMS + i) } =
                    *unsafe { eqns.x.get_unchecked(j) };
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
    /// Marks blocks whose content is flat. The gradient features come from the denoised
    /// plane, so grain doesn't hide flat content and denoiser halos at edges are rejected; the
    /// variance check uses the source, to skip clipped areas that carry no grain.
    pub fn run(&self, plane: &Plane<u8>, denoised: &Plane<u8>) -> (Vec<u8>, usize) {
        const TRACE_THRESHOLD: f64 = 0.15f64 / BLOCK_SIZE_SQUARED as f64;
        const NORM_THRESHOLD: f64 = 0.08f64 / BLOCK_SIZE_SQUARED as f64;
        // A grain std of 0.3 (8-bit): below this the block is clipped or carries no grain.
        const VAR_THRESHOLD: f64 = (0.3 / BLOCK_NORMALIZATION) * (0.3 / BLOCK_NORMALIZATION);

        let num_blocks_w = plane.width().div_ceil(BLOCK_SIZE);
        let num_blocks_h = plane.height().div_ceil(BLOCK_SIZE);
        let num_blocks = num_blocks_w * num_blocks_h;
        let flat_blocks = (0..num_blocks)
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
                let mut denoised_result = [0.0f64; BLOCK_SIZE_SQUARED];

                self.extract_block(
                    plane,
                    bx * BLOCK_SIZE,
                    by * BLOCK_SIZE,
                    &mut plane_result,
                    &mut block_result,
                );
                self.extract_block(
                    denoised,
                    bx * BLOCK_SIZE,
                    by * BLOCK_SIZE,
                    &mut plane_result,
                    &mut denoised_result,
                );
                for yi in 1..(BLOCK_SIZE - 1) {
                    for xi in 1..(BLOCK_SIZE - 1) {
                        // SAFETY: We know the size of `block_result` and that we cannot exceed the bounds of it
                        unsafe {
                            let result_ptr = denoised_result.as_ptr().add(yi * BLOCK_SIZE + xi);

                            let gx = (*result_ptr.add(1) - *result_ptr.sub(1)) / 2f64;
                            let gy =
                                (*result_ptr.add(BLOCK_SIZE) - *result_ptr.sub(BLOCK_SIZE)) / 2f64;
                            gxx += gx * gx;
                            gxy += gx * gy;
                            gyy += gy * gy;

                            let block_val = *block_result.as_ptr().add(yi * BLOCK_SIZE + xi);
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
                // Spectral norm. libaom also tests the eigenvalue ratio, which tells grain from
                // structure on a noisy plane; on the clean denoised plane the norm catches edges.
                let norm = f64::midpoint(trace, e_sub);
                let is_flat =
                    trace < TRACE_THRESHOLD && norm < NORM_THRESHOLD && var > VAR_THRESHOLD;

                if is_flat { 255u8 } else { 0 }
            })
            .collect::<Vec<_>>();
        let num_flat = flat_blocks.iter().filter(|&&block| block != 0).count();

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
        let plane_origin = get_dbg(plane.data(), plane.geometry().data_origin()..);
        let width = plane.width();
        let height = plane.height();
        let stride = plane.geometry().stride();

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
        if unsafe { self.a.get_unchecked(last * self.n + last) }.abs() > TOLERANCE {
            *unsafe { self.x.get_unchecked_mut(last) } = *unsafe { self.b.get_unchecked(last) }
                / *unsafe { self.a.get_unchecked(last * self.n + last) };
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

fn to_coeff(value: f64) -> i8 {
    let rounded = value.round() as i32;
    if !(-128..=127).contains(&rounded) {
        log::warn!("AR coefficient {value:.1} clamped to the i8 range");
    }
    rounded.clamp(-128, 127) as i8
}

/// Subtract the block's diff plane from each neighbour: its value `p` at the current pixel plus
/// the plane's rise to that neighbour.
#[cfg(test)]
fn remove_plane(values: &mut [f64], p: f64, taps: &[f64]) {
    for (value, tap) in values.iter_mut().zip(taps) {
        *value -= p + tap;
    }
}

#[cfg(test)]
fn add_ar_observation(eqns: &mut EquationSystem, buffer: &[f64], val: f64) {
    match eqns.n {
        NUM_Y_COEFFS => add_ar_observation_n::<NUM_Y_COEFFS>(eqns, buffer, val),
        NUM_UV_COEFFS => add_ar_observation_n::<NUM_UV_COEFFS>(eqns, buffer, val),
        _ => unreachable!("unsupported AR coefficient count"),
    }
}

#[cfg(test)]
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
        allow_split: bool,
    ) -> NoiseStatus {
        let num_blocks_w = source.y_plane.width().div_ceil(BLOCK_SIZE);
        let num_blocks_h = source.y_plane.height().div_ceil(BLOCK_SIZE);
        let mut y_model_different = false;

        // Clear the latest equation system
        for i in 0..3 {
            let state = unsafe { self.latest_state.get_unchecked_mut(i) };
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
        // ponytail: fixed 1% share; tune per resolution if small frames starve.
        let min_flat_blocks = (flat_blocks.len() / 100).max(2);
        if flat_block_indices.len() < min_flat_blocks {
            return NoiseStatus::Error(anyhow!(
                "{} flat blocks, need {min_flat_blocks} to update noise estimate",
                flat_block_indices.len()
            ));
        }

        let frame_dims = (source.y_plane.width(), source.y_plane.height());
        // Luma uses these planes for its own fit and chroma for its luma input.
        let luma_planes = flat_block_indices
            .par_iter()
            .map(|&index| {
                let (bx, by) = (index % num_blocks_w, index / num_blocks_w);
                DiffPlane::fit(
                    &source.y_plane,
                    &denoised.y_plane,
                    frame_dims,
                    bx * BLOCK_SIZE,
                    by * BLOCK_SIZE,
                    BLOCK_SIZE,
                    BLOCK_SIZE,
                )
            })
            .collect::<Vec<_>>();
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
                &luma_planes,
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
                    &luma_planes,
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
                    &luma_planes,
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
                if allow_split
                    && channel == 0
                    && combined_state.strength_solver.num_equations > 0
                    && is_different
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
        let mut scaling_points_cb =
            scaling_points_cb.expect("scope task should set Cb scaling points");
        let mut scaling_points_cr =
            scaling_points_cr.expect("scope task should set Cr scaling points");
        // Colour grain only when clearly present: a denoiser's colour changes otherwise
        // become coloured specks.
        let luma_strength = self.combined_state[0].strength_solver.average_strength();
        let chroma_floor = CHROMA_FLOOR.max(CHROMA_SHARE_OF_LUMA * luma_strength);
        // Cb and Cr go together: with 4:2:0, AV1 forbids points on one chroma plane only.
        if self.combined_state[1..]
            .iter()
            .any(|state| state.strength_solver.average_strength() < chroma_floor)
        {
            scaling_points_cb.clear();
            scaling_points_cr.clear();
        }

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
            let eqns = &unsafe { self.combined_state.get_unchecked(c) }.eqns;
            for i in 0..n_coeff {
                let xi = *unsafe { eqns.x.get_unchecked(i) };
                if xi > max_coeff {
                    max_coeff = xi;
                }
                if xi < min_coeff {
                    min_coeff = xi;
                }
            }

            // Since the correlation between luma/chroma was computed in an already
            // scaled space, we adjust it in the un-scaled space.
            let solver = &unsafe { self.combined_state.get_unchecked(c) }.strength_solver;
            let average_strength = solver.average_strength_or_one();
            if c == 0 {
                avg_luma_strength = average_strength;
            } else {
                let y_corr_cur = unsafe { y_corr.get_unchecked_mut(c - 1) };
                *y_corr_cur = avg_luma_strength * *unsafe { eqns.x.get_unchecked(n_coeff) }
                    / average_strength;
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

    /// Forget everything, so the next frame starts a fresh estimate.
    pub fn reset(&mut self) {
        for state in &mut self.combined_state {
            state.eqns.clear();
            state.strength_solver.clear();
            state.num_observations = 0;
            state.ar_gain = 1.0f64;
        }
    }

    /// Whether any frame since the last reset contributed to the luma estimate.
    #[must_use]
    pub const fn has_estimate(&self) -> bool {
        self.combined_state[0].strength_solver.num_equations > 0
    }

    pub fn save_latest(&mut self) {
        for c in 0..3 {
            let latest_state = unsafe { self.latest_state.get_unchecked(c) };
            let combined_state = unsafe { self.combined_state.get_unchecked_mut(c) };
            combined_state.eqns.copy_from(&latest_state.eqns);
            combined_state
                .strength_solver
                .eqns
                .copy_from(&latest_state.strength_solver.eqns);
            combined_state.strength_solver.num_equations =
                latest_state.strength_solver.num_equations;
            combined_state.strength_solver.total = latest_state.strength_solver.total;
            combined_state
                .strength_solver
                .bin_weight
                .clone_from(&latest_state.strength_solver.bin_weight);
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
            let xi = *unsafe { eqns.x.get_unchecked(i) };
            coeffs.push(to_coeff(scale_ar_coeff * xi));
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
        let eqns = &unsafe { self.combined_state.get_unchecked(channel) }.eqns;
        for i in 0..n_coeff {
            let xi = *unsafe { eqns.x.get_unchecked(i) };
            coeffs.push(to_coeff(scale_ar_coeff * xi));
        }
        coeffs.push(to_coeff(
            scale_ar_coeff * *unsafe { y_corr.get_unchecked(channel - 1) },
        ));
        coeffs
    }

    // Return true if the noise estimate appears to be different from the combined
    // (multi-frame) estimate. The difference is measured by checking whether the
    // AR coefficients have diverged (using a threshold on normalized cross
    // correlation), or whether the noise strength has changed.
    #[must_use]
    fn is_different(&self) -> bool {
        const COEFF_THRESHOLD: f64 = 0.9f64;
        // Mean strength change relative to the current strength.
        const STRENGTH_THRESHOLD: f64 = 0.15f64;

        let latest = &self.latest_state[0];
        let combined = &self.combined_state[0];
        // Near-white grain has near-zero coefficients whose correlation is just estimation
        // noise, so the shapes must also differ by a real amount.
        const COEFF_DISTANCE: f64 = 0.15f64;
        let corr = normalized_cross_correlation(&latest.eqns.x, &combined.eqns.x, combined.eqns.n);
        let distance = latest
            .eqns
            .x
            .iter()
            .zip(&combined.eqns.x)
            .take(combined.eqns.n)
            .map(|(l, c)| (l - c) * (l - c))
            .sum::<f64>()
            .sqrt();
        if corr < COEFF_THRESHOLD && distance > COEFF_DISTANCE {
            return true;
        }

        let latest_eqns = &latest.strength_solver.eqns;
        let combined_eqns = &combined.strength_solver.eqns;
        let mut diff = 0.0f64;
        let mut strength = 0.0f64;
        let mut total_weight = 0.0f64;
        for j in 0..latest_eqns.n {
            let mut weight = 0.0f64;
            for i in 0..latest_eqns.n {
                weight += *unsafe { latest_eqns.a.get_unchecked(i * latest_eqns.n + j) };
            }
            weight = weight.sqrt();
            diff += weight
                * (*unsafe { latest_eqns.x.get_unchecked(j) }
                    - *unsafe { combined_eqns.x.get_unchecked(j) })
                .abs();
            strength += weight * *unsafe { combined_eqns.x.get_unchecked(j) };
            total_weight += weight;
        }

        diff > STRENGTH_THRESHOLD * strength.max(total_weight * 0.1)
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
        luma_planes: &[DiffPlane],
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
                luma_planes,
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
        luma_planes: &[DiffPlane],
        num_blocks_w: usize,
        _num_blocks_h: usize,
    ) {
        let num_coords = model_n;
        let n = state.eqns.n;
        let block_w = BLOCK_SIZE / usize::from(source.geometry().subsampling_x());
        let block_h = BLOCK_SIZE / usize::from(source.geometry().subsampling_y());

        let dec = (
            usize::from(source.geometry().subsampling_x()) >> 1,
            usize::from(source.geometry().subsampling_y()) >> 1,
        );
        let stride = source.geometry().stride();
        let source_origin = get_dbg(source.data(), source.geometry().data_origin()..);
        let denoised_origin = get_dbg(denoised.data(), denoised.geometry().data_origin()..);
        let alt_stride = alt_source.map_or(0, |s| s.geometry().stride());
        let alt_source_origin = alt_source.map(|s| get_dbg(s.data(), s.geometry().data_origin()..));
        let alt_denoised_origin =
            alt_denoised.map(|s| get_dbg(s.data(), s.geometry().data_origin()..));
        let alt_origins = alt_source_origin.zip(alt_denoised_origin);
        const BLOCKS_PER_CHUNK: usize = 8;
        let observations = flat_block_indices
            .par_chunks(BLOCKS_PER_CHUNK)
            .zip(luma_planes.par_chunks(BLOCKS_PER_CHUNK))
            .map(|(chunk, luma_planes)| {
                let mut chunk_eqns = EquationSystem::new(n);
                let mut chunk_observations = 0usize;
                let mut eqns = EquationSystem::new(n);
                let mut values = Vec::new();

                for (&block_index, &luma_plane) in chunk.iter().zip(luma_planes) {
                    eqns.clear();
                    let by = block_index / num_blocks_w;
                    let bx = block_index % num_blocks_w;
                    let flat_block_index = by * num_blocks_w + bx;

                    let y_o = by * block_h;
                    let x_o = bx * block_w;
                    // Fit the grain texture, not the denoiser's offset and gradient.
                    let plane_dims = (frame_dims.0 >> dec.0, frame_dims.1 >> dec.1);
                    let (plane, alt_plane) = if alt_source.is_some() {
                        let plane = DiffPlane::fit(
                            source, denoised, plane_dims, x_o, y_o, block_w, block_h,
                        );
                        (plane, Some(luma_plane))
                    } else {
                        (luma_plane, None)
                    };
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
                    let alt =
                        alt_origins
                            .zip(alt_plane)
                            .map(|((alt_source, alt_denoised), plane)| gram::Alt {
                                source: alt_source,
                                denoised: alt_denoised,
                                stride: alt_stride,
                                dec,
                                plane,
                            });
                    let (num_observations, residual_energy) = gram::block_equations(
                        source_origin,
                        denoised_origin,
                        stride,
                        (x_o, y_o),
                        &gram::Window {
                            x0: x_start,
                            x1: x_end,
                            y0: y_start,
                            y1: y_end,
                        },
                        get_dbg(coords, ..num_coords),
                        &plane,
                        alt.as_ref(),
                        n,
                        &mut eqns.a,
                        &mut eqns.b,
                        &mut values,
                    );
                    if num_observations > 0 {
                        // Fitting the plane to the block's own pixels anti-correlates what is
                        // left by about 3 var / samples for every pixel pair; add that back.
                        let bias = 3f64 * residual_energy / plane.samples as f64;
                        for i in 0..num_coords {
                            for j in i..num_coords {
                                *get_dbg_mut(&mut eqns.a, i * n + j) += bias;
                            }
                            *get_dbg_mut(&mut eqns.b, i) += bias;
                        }
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
        let block_w = BLOCK_SIZE / usize::from(source.geometry().subsampling_x());
        let block_h = BLOCK_SIZE / usize::from(source.geometry().subsampling_y());
        let source_subsampling_x = usize::from(source.geometry().subsampling_x());
        let source_subsampling_y = usize::from(source.geometry().subsampling_y());
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
                    x_o << (usize::from(source.geometry().subsampling_x()) >> 1),
                    y_o << (usize::from(source.geometry().subsampling_y()) >> 1),
                );
                let noise_var = get_noise_var(
                    source,
                    denoised,
                    (
                        frame_dims.0 >> (usize::from(source.geometry().subsampling_x()) >> 1),
                        frame_dims.1 >> (usize::from(source.geometry().subsampling_y()) >> 1),
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
        // A small ridge keeps near-singular systems solvable without visibly biasing the fit.
        const RIDGE: f64 = 1.0e-3f64;
        // The AR filter must stay clearly stable: a coefficient sum near 1 synthesises blotches.
        const MAX_ABS_SUM: f64 = 0.95f64;

        let n = self.eqns.n;
        let n_ar = n - usize::from(is_chroma);
        let mean_diag = (0..n_ar)
            .map(|i| *get_dbg(&self.eqns.a, i * n + i))
            .sum::<f64>()
            / n_ar as f64;
        let mut a = self.eqns.a.clone();
        for i in 0..n_ar {
            *get_dbg_mut(&mut a, i * n + i) += RIDGE * mean_diag;
        }
        let mut b = self.eqns.b.clone();
        let ret = linsolve(n, &mut a, n, &mut b, &mut self.eqns.x);
        self.ar_gain = 1.0f64;
        if !ret {
            return ret;
        }
        let coeffs = get_dbg_mut(&mut self.eqns.x, ..n_ar);
        let abs_sum: f64 = coeffs.iter().map(|x| x.abs()).sum();
        if abs_sum >= MAX_ABS_SUM {
            let scale = 0.99f64 * MAX_ABS_SUM / abs_sum;
            for x in coeffs {
                *x *= scale;
            }
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
            var += *unsafe { self.eqns.a.get_unchecked(i * self.eqns.n + i) }
                / self.num_observations as f64;
        }
        var /= n_adjusted as f64;

        // Keep track of E(Y^2) = <b, x> + E(X^2)
        // In the case that we are using chroma and have an estimate of correlation
        // with luma we adjust that estimate slightly to remove the correlated bits by
        // subtracting out the last column of a scaled by our correlation estimate
        // from b. E(y^2) = <b - A(:, end)*x(end), x>
        let mut sum_covar = 0f64;
        for i in 0..n_adjusted {
            let mut bi = *unsafe { self.eqns.b.get_unchecked(i) };
            if is_chroma {
                bi -= *unsafe { self.eqns.a.get_unchecked(i * self.eqns.n + n_adjusted) }
                    * *unsafe { self.eqns.x.get_unchecked(n_adjusted) };
            }
            sum_covar +=
                (bi * *unsafe { self.eqns.x.get_unchecked(i) }) / self.num_observations as f64;
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
    /// Measurement weight per bin; sparse bins don't get to set their own value.
    bin_weight: Vec<f64>,
}

impl StrengthSolver {
    #[must_use]
    pub fn new(num_bins: usize) -> Self {
        Self {
            eqns: EquationSystem::new(num_bins),
            num_bins,
            num_equations: 0usize,
            total: 0f64,
            bin_weight: vec![0f64; num_bins],
        }
    }

    pub fn add_measurement(&mut self, block_mean: f64, noise_std: f64) {
        let bin = self.get_bin_index(block_mean);
        let bin_i0 = bin.floor() as usize;
        let bin_i1 = (self.num_bins - 1).min(bin_i0 + 1);
        let a = bin - bin_i0 as f64;
        let n = self.num_bins;
        let eqns = &mut self.eqns;
        *unsafe { eqns.a.get_unchecked_mut(bin_i0 * n + bin_i0) } += (1f64 - a).powi(2);
        *unsafe { eqns.a.get_unchecked_mut(bin_i1 * n + bin_i0) } += a * (1f64 - a);
        *unsafe { eqns.a.get_unchecked_mut(bin_i1 * n + bin_i1) } += a.powi(2);
        *unsafe { eqns.a.get_unchecked_mut(bin_i0 * n + bin_i1) } += (1f64 - a) * a;
        *unsafe { eqns.b.get_unchecked_mut(bin_i0) } += (1f64 - a) * noise_std;
        *unsafe { eqns.b.get_unchecked_mut(bin_i1) } += a * noise_std;
        *get_dbg_mut(&mut self.bin_weight, bin_i0) += 1f64 - a;
        *get_dbg_mut(&mut self.bin_weight, bin_i1) += a;
        self.total += noise_std;
        self.num_equations += 1;
    }

    pub fn solve(&mut self) -> bool {
        // Add regularization proportional to the number of constraints
        let n = self.num_bins;
        let alpha = 2f64 * self.num_equations as f64 / n as f64;

        // Do this in a non-destructive manner so it is not confusing to the caller
        let old_a = self.eqns.a.clone();
        let old_b = self.eqns.b.clone();
        // Sparse bins leave the data term: a handful of blocks (often edge halos in a bright
        // window) would otherwise set the grain for a whole brightness range, and the
        // smoothing below would spread it into the reliable bins.
        let reliable = self.reliable_bins();
        for (i, &is_reliable) in reliable.iter().enumerate() {
            if !is_reliable {
                for j in 0..n {
                    *get_dbg_mut(&mut self.eqns.a, i * n + j) = 0f64;
                    *get_dbg_mut(&mut self.eqns.a, j * n + i) = 0f64;
                }
                *get_dbg_mut(&mut self.eqns.b, i) = 0f64;
            }
        }
        // Penalise curvature, not slope: a slope penalty pulls both ends of a sloped strength
        // curve towards flat (libaom's choice; it overstated highlight grain by 30%).
        const SECOND_DIFF: [f64; 3] = [1f64, -2f64, 1f64];
        for k in 0..n.saturating_sub(2) {
            for (di, &wi) in SECOND_DIFF.iter().enumerate() {
                for (dj, &wj) in SECOND_DIFF.iter().enumerate() {
                    *get_dbg_mut(&mut self.eqns.a, (k + di) * n + k + dj) += alpha * wi * wj;
                }
            }
        }

        // Small regularization to give average noise strength
        let mean = self.total / self.num_equations as f64;
        for i in 0..n {
            *unsafe { self.eqns.a.get_unchecked_mut(i * n + i) } += 1f64 / 8192f64;
            *unsafe { self.eqns.b.get_unchecked_mut(i) } += mean / 8192f64;
        }
        let result = self.eqns.solve();
        self.eqns.a = old_a;
        self.eqns.b = old_b;
        if result {
            self.hold_sparse_bins(&reliable);
        }
        result
    }

    /// Bins with enough measurements to set their own value.
    fn reliable_bins(&self) -> Vec<bool> {
        const MIN_BIN_WEIGHT: f64 = 2f64;
        const MIN_BIN_SHARE: f64 = 0.01f64;
        let total: f64 = self.bin_weight.iter().sum();
        let min_weight = MIN_BIN_WEIGHT.max(MIN_BIN_SHARE * total);
        self.bin_weight.iter().map(|&w| w >= min_weight).collect()
    }

    /// Sparse bins take their value from the reliable ones: held flat beyond the ends,
    /// interpolated in between, never extrapolated along a slope.
    fn hold_sparse_bins(&mut self, reliable: &[bool]) {
        let reliable: Vec<usize> = (0..self.num_bins)
            .filter(|&i| *get_dbg(reliable, i))
            .collect();
        let (Some(&first), Some(&last)) = (reliable.first(), reliable.last()) else {
            return;
        };
        let x = &mut self.eqns.x;
        let (first_value, last_value) = (*get_dbg(x, first), *get_dbg(x, last));
        get_dbg_mut(x, ..first).fill(first_value);
        get_dbg_mut(x, last + 1..).fill(last_value);
        for (&lo, &hi) in reliable.iter().zip(reliable.iter().skip(1)) {
            let (lo_value, hi_value) = (*get_dbg(x, lo), *get_dbg(x, hi));
            for i in (lo + 1)..hi {
                let t = (i - lo) as f64 / (hi - lo) as f64;
                *get_dbg_mut(x, i) = lo_value.mul_add(1f64 - t, hi_value * t);
            }
        }
    }

    #[must_use]
    pub fn fit_piecewise(&self, max_output_points: usize) -> NoiseStrengthLut {
        const TOLERANCE: f64 = 0.00625f64;

        let mut lut = NoiseStrengthLut::new(self.num_bins);
        for i in 0..self.num_bins {
            (unsafe { lut.points.get_unchecked_mut(i) })[0] = self.get_center(i);
            (unsafe { lut.points.get_unchecked_mut(i) })[1] =
                *unsafe { self.eqns.x.get_unchecked(i) };
        }

        let mut residual = vec![0.0f64; self.num_bins];
        self.update_piecewise_linear_residual(&lut, &mut residual, 0, self.num_bins);

        // Greedily remove points if there are too many or if it doesn't hurt local
        // approximation (never remove the end points)
        while lut.points.len() > 2 {
            let mut min_index = 1usize;
            for j in 1..(lut.points.len() - 1) {
                if *unsafe { residual.get_unchecked(j) }
                    < *unsafe { residual.get_unchecked(min_index) }
                {
                    min_index = j;
                }
            }
            let dx = unsafe { lut.points.get_unchecked(min_index + 1) }[0]
                - unsafe { lut.points.get_unchecked(min_index - 1) }[0];
            let avg_residual = unsafe { residual.get_unchecked(min_index) } / dx;
            if lut.points.len() <= max_output_points && avg_residual > TOLERANCE {
                break;
            }

            lut.points.remove(min_index);
            // Keep residuals aligned with their points (libaom leaves them shifted by one, so
            // a flat run's zero residual can delete the knee after it).
            residual.remove(min_index);
            self.update_piecewise_linear_residual(
                &lut,
                &mut residual,
                min_index - 1,
                min_index + 1,
            );
        }

        lut
    }

    /// Strength averaged over the bins, weighted by their measurements; 0 without any.
    #[must_use]
    fn average_strength(&self) -> f64 {
        let n = self.eqns.n;
        let mut average_strength = 0f64;
        let mut total_weight = 0f64;
        for i in 0..n {
            let w = get_dbg(&self.eqns.a, i * n..(i + 1) * n)
                .iter()
                .sum::<f64>()
                .sqrt();
            average_strength += *get_dbg(&self.eqns.x, i) * w;
            total_weight += w;
        }
        if total_weight.abs() < f64::EPSILON {
            0f64
        } else {
            average_strength / total_weight
        }
    }

    #[must_use]
    fn average_strength_or_one(&self) -> f64 {
        if self.num_equations == 0 {
            1f64
        } else {
            self.average_strength()
        }
    }

    #[must_use]
    pub fn get_value(&self, x: f64) -> f64 {
        let bin = self.get_bin_index(x);
        let bin_i0 = bin.floor() as usize;
        let bin_i1 = (self.num_bins - 1).min(bin_i0 + 1);
        let a = bin - bin_i0 as f64;
        (1f64 - a).mul_add(
            *unsafe { self.eqns.x.get_unchecked(bin_i0) },
            a * *unsafe { self.eqns.x.get_unchecked(bin_i1) },
        )
    }

    pub fn clear(&mut self) {
        self.eqns.clear();
        self.num_equations = 0;
        self.total = 0f64;
        self.bin_weight.fill(0f64);
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
            let lower = self
                .get_bin_index(unsafe { lut.points.get_unchecked(i - 1) }[0])
                .floor() as usize;
            let upper = (self.num_bins - 1).min(
                self.get_bin_index(unsafe { lut.points.get_unchecked(i + 1) }[0])
                    .ceil() as usize,
            );
            let mut r = 0f64;
            for j in lower..=upper {
                let x = self.get_center(j);
                if x < unsafe { lut.points.get_unchecked(i - 1) }[0]
                    || x >= unsafe { lut.points.get_unchecked(i + 1) }[0]
                {
                    continue;
                }

                let y = *unsafe { self.eqns.x.get_unchecked(j) };
                let a = (x - unsafe { lut.points.get_unchecked(i - 1) }[0])
                    / (unsafe { lut.points.get_unchecked(i + 1) }[0]
                        - unsafe { lut.points.get_unchecked(i - 1) }[0]);
                let estimate_y = unsafe { lut.points.get_unchecked(i - 1) }[1]
                    .mul_add(1f64 - a, unsafe { lut.points.get_unchecked(i + 1) }[1] * a);
                r += (y - estimate_y).abs();
            }
            *unsafe { residual.get_unchecked_mut(i) } = r * dx;
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
        dest += addend;
        dest
    }
}

impl AddAssign<&StrengthSolver> for StrengthSolver {
    fn add_assign(&mut self, rhs: &StrengthSolver) {
        self.eqns += &rhs.eqns;
        self.num_equations += rhs.num_equations;
        self.total += rhs.total;
        for (weight, rhs_weight) in self.bin_weight.iter_mut().zip(&rhs.bin_weight) {
            *weight += *rhs_weight;
        }
    }
}

#[cfg(test)]
mod tests {
    use v_frame::frame::FrameBuilder;

    use super::util::{extract_ar_row, extract_ar_row_with_alt};

    use super::*;

    /// The block equations as the old per-pixel loop built them.
    #[allow(clippy::indexing_slicing, reason = "test reference")]
    #[allow(clippy::too_many_arguments)]
    fn per_pixel_equations(
        coords: &[[isize; 2]],
        source: &Plane<u8>,
        denoised: &Plane<u8>,
        alt: Option<(&Plane<u8>, &Plane<u8>)>,
        (x_o, y_o): (usize, usize),
        win: &gram::Window,
        plane: &DiffPlane,
        alt_plane: Option<&DiffPlane>,
        eqns: &mut EquationSystem,
    ) -> f64 {
        let dec = (
            usize::from(source.geometry().subsampling_x()) >> 1,
            usize::from(source.geometry().subsampling_y()) >> 1,
        );
        let stride = source.geometry().stride();
        let origin = |p: &Plane<u8>| p.data()[p.geometry().data_origin()..].to_vec();
        let (so, dno) = (origin(source), origin(denoised));
        let offsets = coords
            .iter()
            .map(|c| c[1] * stride as isize + c[0])
            .collect::<Vec<_>>();
        let taps = coords
            .iter()
            .map(|c| plane.step(c[0], c[1]))
            .collect::<Vec<_>>();
        let mut buffer = [0f64; NUM_UV_COEFFS];
        let mut residual = 0f64;
        for y in win.y0..win.y1 {
            for x in win.x0..win.x1 {
                let base = (y + y_o) * stride + x_o + x;
                let val = match alt {
                    Some((s, d)) => extract_ar_row_with_alt(
                        &offsets,
                        coords.len(),
                        &so,
                        &dno,
                        base,
                        dec,
                        &origin(s),
                        &origin(d),
                        s.geometry().stride(),
                        x + x_o,
                        y + y_o,
                        &mut buffer,
                    ),
                    None => extract_ar_row(&offsets, coords.len(), &so, &dno, base, &mut buffer),
                };
                let p = plane.at((x + x_o) as f64, (y + y_o) as f64);
                remove_plane(&mut buffer[..coords.len()], p, &taps);
                if let Some(alt_plane) = alt_plane {
                    buffer[coords.len()] -= alt_plane.at(
                        ((x + x_o) << dec.0) as f64 + 0.5 * dec.0 as f64,
                        ((y + y_o) << dec.1) as f64 + 0.5 * dec.1 as f64,
                    );
                }
                add_ar_observation(eqns, &buffer, val - p);
                residual += (val - p) * (val - p);
            }
        }
        residual
    }

    #[test]
    #[allow(clippy::indexing_slicing, reason = "test")]
    fn integer_block_equations_match_the_per_pixel_fit() {
        let (w, h) = (192usize, 128usize);
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut noise = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 20.0
        };
        let mut fill = |pw: usize, ph: usize, offset: f64| {
            let clean = (0..pw * ph)
                .map(|i| 90.0 + (i % pw) as f64 * 0.3 + (i / pw) as f64 * 0.2)
                .collect::<Vec<_>>();
            // A denoiser-style offset and slope on top of the grain.
            let noisy = clean
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    let drift = offset + 0.02 * (i % pw) as f64 - 0.03 * (i / pw) as f64;
                    (c + drift + noise()).round().clamp(0.0, 255.0) as u8
                })
                .collect::<Vec<_>>();
            let clean = clean.iter().map(|c| c.round() as u8).collect::<Vec<_>>();
            (noisy, clean)
        };
        let mut src = FrameBuilder::new(w, h, ChromaSubsampling::Yuv420, 8)
            .build()
            .expect("4:2:0 frame builds");
        let mut den = src.clone();
        let (ys, yd) = fill(w, h, 2.5);
        let (us, ud) = fill(w / 2, h / 2, -1.5);
        src.y_plane.copy_from_slice(&ys).expect("luma fits");
        den.y_plane.copy_from_slice(&yd).expect("luma fits");
        let cb = |f: &mut Frame<u8>, data: &[u8]| {
            f.u_plane
                .as_mut()
                .expect("4:2:0 has Cb")
                .copy_from_slice(data)
                .expect("Cb fits");
        };
        cb(&mut src, &us);
        cb(&mut den, &ud);

        let coords = NoiseModel::new().coords;
        let lag = NOISE_MODEL_LAG;
        let origin = |p: &Plane<u8>| p.data()[p.geometry().data_origin()..].to_vec();
        let (ys_o, yd_o) = (origin(&src.y_plane), origin(&den.y_plane));
        let mut values = Vec::new();
        for (bx, by) in [(1usize, 1usize), (2, 2), (4, 2)] {
            for chroma in [false, true] {
                let (s, d, bw, dims) = if chroma {
                    let s = src.u_plane.as_ref().expect("4:2:0 has Cb");
                    let d = den.u_plane.as_ref().expect("4:2:0 has Cb");
                    (s, d, BLOCK_SIZE / 2, (w / 2, h / 2))
                } else {
                    (&src.y_plane, &den.y_plane, BLOCK_SIZE, (w, h))
                };
                // Interior block, block with flat neighbours, and with a flat block above only.
                for (x0, y0, x1) in [(lag, lag, bw - lag), (0, 0, bw), (lag, 0, bw - lag)] {
                    let win = gram::Window { x0, x1, y0, y1: bw };
                    let (x_o, y_o) = (bx * bw, by * bw);
                    let plane = DiffPlane::fit(s, d, dims, x_o, y_o, bw, bw);
                    let alt_plane = chroma.then(|| {
                        let (lx, ly) = (x_o * 2, y_o * 2);
                        DiffPlane::fit(&src.y_plane, &den.y_plane, (w, h), lx, ly, 32, 32)
                    });
                    let n = coords.len() + usize::from(chroma);
                    let mut want = EquationSystem::new(n);
                    let want_residual = per_pixel_equations(
                        &coords,
                        s,
                        d,
                        chroma.then_some((&src.y_plane, &den.y_plane)),
                        (x_o, y_o),
                        &win,
                        &plane,
                        alt_plane.as_ref(),
                        &mut want,
                    );
                    let alt = alt_plane.map(|plane| gram::Alt {
                        source: &ys_o,
                        denoised: &yd_o,
                        stride: src.y_plane.geometry().stride(),
                        dec: (1, 1),
                        plane,
                    });
                    let mut got = EquationSystem::new(n);
                    let (count, got_residual) = gram::block_equations(
                        &origin(s),
                        &origin(d),
                        s.geometry().stride(),
                        (x_o, y_o),
                        &win,
                        &coords,
                        &plane,
                        alt.as_ref(),
                        n,
                        &mut got.a,
                        &mut got.b,
                        &mut values,
                    );
                    assert_eq!(count, (x1 - x0) * (bw - y0));
                    let close = |a: f64, b: f64| (a - b).abs() <= 1e-6 * (1.0 + b.abs());
                    for i in 0..n {
                        for j in i..n {
                            let (g, e) = (got.a[i * n + j], want.a[i * n + j]);
                            assert!(close(g, e), "a[{i}][{j}] {g} != {e}, chroma {chroma}");
                        }
                        let (g, e) = (got.b[i], want.b[i]);
                        assert!(close(g, e), "b[{i}] {g} != {e}, chroma {chroma}");
                    }
                    assert!(
                        close(got_residual, want_residual),
                        "residual, chroma {chroma}"
                    );
                }
            }
        }
    }

    #[test]
    fn sparse_bright_bins_hold_the_mid_tone_strength() {
        let mut solver = StrengthSolver::new(20);
        for i in 0..400 {
            solver.add_measurement(60.0 + f64::from(i % 80), 2.0);
        }
        // Three halo blocks in the highlights with a huge std.
        for _ in 0..3 {
            solver.add_measurement(235.0, 12.0);
        }
        assert!(solver.solve());
        let mid = solver.get_value(100.0);
        let high = solver.get_value(240.0);
        assert!((mid - 2.0).abs() < 0.1, "mid {mid}");
        assert!((high - mid).abs() < 0.1, "high {high} vs mid {mid}");
    }

    #[test]
    fn near_unit_ar_fit_is_shrunk_to_a_stable_filter() {
        let mut state = NoiseModelState::new(2);
        // Two strongly correlated neighbours that predict the target almost exactly.
        for i in 0..200 {
            let v = f64::from(i % 7) - 3.0;
            let buffer = [v, v];
            add_ar_observation_n::<2>(&mut state.eqns, &buffer, v);
            state.num_observations += 1;
        }
        mirror_upper_triangle(&mut state.eqns);
        assert!(state.ar_equation_system_solve(false));
        let abs_sum: f64 = state.eqns.x.iter().map(|x| x.abs()).sum();
        assert!(abs_sum < 0.95, "sum {abs_sum}");
    }
}
