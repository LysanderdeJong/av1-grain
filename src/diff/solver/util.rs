use std::ptr;

use arrayvec::ArrayVec;
use v_frame::plane::Plane;

use super::gram::dot;

use crate::{
    diff::BLOCK_SIZE,
    util::{get_dbg, get_dbg_mut},
};

/// Solves Ax = b, where x and b are column vectors of size nx1 and A is nxn
#[allow(clippy::many_single_char_names)]
pub(super) fn linsolve(
    n: usize,
    a: &mut [f64],
    stride: usize,
    b: &mut [f64],
    x: &mut [f64],
) -> bool {
    // Forward elimination
    for k in 0..(n - 1) {
        // Bring the largest magnitude to the diagonal position
        ((k + 1)..n).rev().for_each(|i| {
            if unsafe { a.get_unchecked((i - 1) * stride + k) }.abs()
                < unsafe { a.get_unchecked(i * stride + k) }.abs()
            {
                (0..n).for_each(|j| {
                    // SAFETY: `linsolve` is crate-private; callers provide matrix dimensions
                    // matching the backing slices.
                    unsafe { swap_unchecked(a, i * stride + j, (i - 1) * stride + j) };
                });
                // SAFETY: `i` iterates over `1..n`, so both row indices are valid for `b`.
                unsafe { swap_unchecked(b, i, i - 1) };
            }
        });

        for i in k..(n - 1) {
            if unsafe { a.get_unchecked(k * stride + k) }.abs() < f64::EPSILON {
                return false;
            }
            let c = *unsafe { a.get_unchecked((i + 1) * stride + k) }
                / *unsafe { a.get_unchecked(k * stride + k) };
            (0..n).for_each(|j| {
                let a2_val = *unsafe { a.get_unchecked(k * stride + j) };
                let a_val = unsafe { a.get_unchecked_mut((i + 1) * stride + j) };
                *a_val = c.mul_add(-a2_val, *a_val);
            });
            let b2_val = *unsafe { b.get_unchecked(k) };
            let b_val = unsafe { b.get_unchecked_mut(i + 1) };
            *b_val = c.mul_add(-b2_val, *b_val);
        }
    }

    // Backward substitution
    for i in (0..n).rev() {
        if unsafe { a.get_unchecked(i * stride + i) }.abs() < f64::EPSILON {
            return false;
        }
        let mut c = 0.0f64;
        for j in (i + 1)..n {
            c = unsafe { a.get_unchecked(i * stride + j) }
                .mul_add(*unsafe { x.get_unchecked(j) }, c);
        }
        *unsafe { x.get_unchecked_mut(i) } =
            (*unsafe { b.get_unchecked(i) } - c) / *unsafe { a.get_unchecked(i * stride + i) };
    }

    true
}

// TODO: This is unstable upstream. Once it's stable upstream, use that.
unsafe fn swap_unchecked<T>(slice: &mut [T], a: usize, b: usize) {
    let ptr = slice.as_mut_ptr();
    // SAFETY: caller has to guarantee that `a < self.len()` and `b < self.len()`
    unsafe {
        ptr::swap(ptr.add(a), ptr.add(b));
    }
}

pub(super) fn multiply_mat(
    m1: &[f64],
    m2: &[f64],
    res: &mut [f64],
    m1_rows: usize,
    inner_dim: usize,
    m2_cols: usize,
) {
    assert!(res.len() >= m1_rows * m2_cols);
    assert!(m1.len() >= m1_rows * inner_dim);
    assert!(m2.len() >= m2_cols * inner_dim);
    if m1_rows == 1 && m2_cols == 3 {
        multiply_row_by_3_cols(m1, m2, res, inner_dim);
        return;
    }
    if inner_dim == 3 && m2_cols == 1 {
        multiply_3_cols_by_col(m1, m2, res, m1_rows);
        return;
    }

    let mut idx = 0;
    for row in 0..m1_rows {
        for col in 0..m2_cols {
            let mut sum = 0f64;
            for inner in 0..inner_dim {
                sum += unsafe { m1.get_unchecked(row * inner_dim + inner) }
                    * unsafe { m2.get_unchecked(inner * m2_cols + col) };
            }

            *unsafe { res.get_unchecked_mut(idx) } = sum;
            idx += 1;
        }
    }
}

#[inline]
fn multiply_row_by_3_cols(m1: &[f64], m2: &[f64], res: &mut [f64], inner_dim: usize) {
    let mut sum0 = 0f64;
    let mut sum1 = 0f64;
    let mut sum2 = 0f64;
    for (m1_val, m2_row) in m1.iter().take(inner_dim).zip(m2.chunks_exact(3)) {
        sum0 += *m1_val * m2_row[0];
        sum1 += *m1_val * m2_row[1];
        sum2 += *m1_val * m2_row[2];
    }
    res[0] = sum0;
    res[1] = sum1;
    res[2] = sum2;
}

#[inline]
fn multiply_3_cols_by_col(m1: &[f64], m2: &[f64], res: &mut [f64], m1_rows: usize) {
    debug_assert!(m2.len() >= 3);
    for (out, row) in res.iter_mut().take(m1_rows).zip(m1.chunks_exact(3)) {
        *out = row[0] * m2[0] + row[1] * m2[1] + row[2] * m2[2];
    }
}

#[must_use]
pub(super) fn normalized_cross_correlation(a: &[f64], b: &[f64], n: usize) -> f64 {
    let mut c = 0f64;
    let mut a_len = 0f64;
    let mut b_len = 0f64;
    for (a, b) in a.iter().zip(b.iter()).take(n) {
        a_len = (*a).mul_add(*a, a_len);
        b_len = (*b).mul_add(*b, b_len);
        c = (*a).mul_add(*b, c);
    }
    c / (a_len.sqrt() * b_len.sqrt())
}

#[inline]
#[cfg(test)]
pub(super) fn extract_ar_row(
    coord_offsets: &[isize],
    num_coords: usize,
    source_origin: &[u8],
    denoised_origin: &[u8],
    base_index: usize,
    buffer: &mut [f64],
) -> f64 {
    debug_assert!(buffer.len() > num_coords);
    debug_assert!(coord_offsets.len() >= num_coords);

    for i in 0..num_coords {
        let index = base_index as isize + *get_dbg(coord_offsets, i);
        debug_assert!(index >= 0);
        let index = index as usize;
        *get_dbg_mut(buffer, i) =
            f64::from(*get_dbg(source_origin, index)) - f64::from(*get_dbg(denoised_origin, index));
    }

    f64::from(*get_dbg(source_origin, base_index))
        - f64::from(*get_dbg(denoised_origin, base_index))
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[cfg(test)]
pub(super) fn extract_ar_row_with_alt(
    coord_offsets: &[isize],
    num_coords: usize,
    source_origin: &[u8],
    denoised_origin: &[u8],
    base_index: usize,
    dec: (usize, usize),
    alt_source_origin: &[u8],
    alt_denoised_origin: &[u8],
    alt_stride: usize,
    x: usize,
    y: usize,
    buffer: &mut [f64],
) -> f64 {
    let val = extract_ar_row(
        coord_offsets,
        num_coords,
        source_origin,
        denoised_origin,
        base_index,
        buffer,
    );

    let mut source_sum = 0u64;
    let mut denoised_sum = 0u64;
    let num_samples = (1 << dec.0) * (1 << dec.1);

    for dy_i in 0..(1 << dec.1) {
        let y_up = (y << dec.1) + dy_i;
        let row_index = y_up * alt_stride + (x << dec.0);
        for dx_i in 0..(1 << dec.0) {
            let index = row_index + dx_i;
            source_sum += u64::from(*get_dbg(alt_source_origin, index));
            denoised_sum += u64::from(*get_dbg(alt_denoised_origin, index));
        }
    }
    *get_dbg_mut(buffer, num_coords) =
        (source_sum as f64 - denoised_sum as f64) / num_samples as f64;

    val
}

#[must_use]
pub(super) fn get_block_mean(
    source: &Plane<u8>,
    frame_dims: (usize, usize),
    x_o: usize,
    y_o: usize,
) -> f64 {
    let max_h = (frame_dims.1 - y_o).min(BLOCK_SIZE);
    let max_w = (frame_dims.0 - x_o).min(BLOCK_SIZE);

    let data_origin = get_dbg(source.data(), source.geometry().data_origin()..);
    let stride = source.geometry().stride();
    let mut block_sum = 0u32;
    for y in 0..max_h {
        let row_start = (y_o + y) * stride + x_o;
        let row = get_dbg(data_origin, row_start..row_start + max_w);
        for pixel in row {
            block_sum += u32::from(*pixel);
        }
    }

    block_sum as f64 / (max_w * max_h) as f64
}

/// Least-squares plane through `source - denoised` over a block: the brightness offset and
/// gentle gradient a denoiser leaves behind, which are not grain.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct DiffPlane {
    /// Pixels the plane was fitted to.
    pub samples: usize,
    mean: f64,
    gx: f64,
    gy: f64,
    cx: f64,
    cy: f64,
}

impl DiffPlane {
    #[must_use]
    pub fn fit(
        source: &Plane<u8>,
        denoised: &Plane<u8>,
        frame_dims: (usize, usize),
        x_o: usize,
        y_o: usize,
        block_w: usize,
        block_h: usize,
    ) -> Self {
        let max_h = (frame_dims.1 - y_o).min(block_h);
        let max_w = (frame_dims.0 - x_o).min(block_w);
        // Centred coordinates make the mean and both slopes independent fits.
        let cx = (max_w as f64 - 1f64) / 2f64;
        let cy = (max_h as f64 - 1f64) / 2f64;

        let source_origin = get_dbg(source.data(), source.geometry().data_origin()..);
        let denoised_origin = get_dbg(denoised.data(), denoised.geometry().data_origin()..);
        let stride = source.geometry().stride();
        // Doubled centred coordinates are integers, so every sum is exact: the same f64 values
        // the per-pixel f64 sums gave, from integer dot products.
        let doubled = |i: usize, len: usize| (2 * i) as i16 - (len as i16 - 1);
        let us = (0..max_w)
            .map(|x| doubled(x, max_w))
            .collect::<ArrayVec<i16, BLOCK_SIZE>>();
        let ones = [1i16; BLOCK_SIZE];
        let ones = get_dbg(&ones, ..max_w);
        let mut diffs = [0i16; BLOCK_SIZE];
        let diffs = get_dbg_mut(&mut diffs, ..max_w);
        let (mut sum, mut sum_u2, mut sum_v2) = (0i64, 0i64, 0i64);
        for y in 0..max_h {
            let row_start = (y_o + y) * stride + x_o;
            let source_row = get_dbg(source_origin, row_start..row_start + max_w);
            let denoised_row = get_dbg(denoised_origin, row_start..row_start + max_w);
            for (diff, (source, denoised)) in
                diffs.iter_mut().zip(source_row.iter().zip(denoised_row))
            {
                *diff = i16::from(*source) - i16::from(*denoised);
            }
            let row_sum = i64::from(dot(diffs, ones));
            sum += row_sum;
            sum_u2 += i64::from(dot(diffs, &us));
            sum_v2 += row_sum * i64::from(doubled(y, max_h));
        }
        let square_sum = |len: usize| {
            (0..len)
                .map(|i| f64::from(doubled(i, len)).powi(2))
                .sum::<f64>()
        };
        let (sum, sum_u, sum_v) = (sum as f64, sum_u2 as f64 / 2f64, sum_v2 as f64 / 2f64);
        let sum_uu = square_sum(max_w) * max_h as f64 / 4f64;
        let sum_vv = square_sum(max_h) * max_w as f64 / 4f64;
        let slope = |num: f64, den: f64| if den > 0f64 { num / den } else { 0f64 };
        Self {
            samples: max_w * max_h,
            mean: sum / (max_w * max_h) as f64,
            gx: slope(sum_u, sum_uu),
            gy: slope(sum_v, sum_vv),
            cx: x_o as f64 + cx,
            cy: y_o as f64 + cy,
        }
    }

    /// The plane's value at plane coordinates `(x, y)`.
    #[must_use]
    pub fn at(&self, x: f64, y: f64) -> f64 {
        self.gy
            .mul_add(y - self.cy, self.gx.mul_add(x - self.cx, self.mean))
    }

    /// How much the plane rises from a pixel to its neighbour at offset `(dx, dy)`.
    #[must_use]
    pub fn step(&self, dx: isize, dy: isize) -> f64 {
        self.gx.mul_add(dx as f64, self.gy * dy as f64)
    }
}

#[must_use]
pub(super) fn get_noise_var(
    source: &Plane<u8>,
    denoised: &Plane<u8>,
    frame_dims: (usize, usize),
    x_o: usize,
    y_o: usize,
    block_w: usize,
    block_h: usize,
) -> f64 {
    let max_h = (frame_dims.1 - y_o).min(block_h);
    let max_w = (frame_dims.0 - x_o).min(block_w);

    let source_origin = get_dbg(source.data(), source.geometry().data_origin()..);
    let denoised_origin = get_dbg(denoised.data(), denoised.geometry().data_origin()..);
    let stride = source.geometry().stride();
    let mut noise_var_sum = 0u32;
    let mut noise_sum = 0i32;
    for y in 0..max_h {
        let row_start = (y_o + y) * stride + x_o;
        let source_row = get_dbg(source_origin, row_start..row_start + max_w);
        let denoised_row = get_dbg(denoised_origin, row_start..row_start + max_w);
        for (source, denoised) in source_row.iter().zip(denoised_row.iter()) {
            let noise = i32::from(*source) - i32::from(*denoised);
            noise_sum += noise;
            let abs_noise = noise.unsigned_abs();
            noise_var_sum += abs_noise * abs_noise;
        }
    }

    let noise_mean = noise_sum as f64 / (max_w * max_h) as f64;
    noise_mean.mul_add(-noise_mean, noise_var_sum as f64 / (max_w * max_h) as f64)
}

#[cfg(test)]
mod tests {
    use v_frame::{chroma::ChromaSubsampling, frame::FrameBuilder};

    use super::*;

    /// The plane fit as the per-pixel f64 sums computed it.
    #[allow(clippy::indexing_slicing, reason = "test reference")]
    fn f64_fit(
        source: &Plane<u8>,
        denoised: &Plane<u8>,
        (w, h): (usize, usize),
        (x_o, y_o): (usize, usize),
    ) -> [f64; 3] {
        let (max_w, max_h) = ((w - x_o).min(BLOCK_SIZE), (h - y_o).min(BLOCK_SIZE));
        let (cx, cy) = ((max_w as f64 - 1.0) / 2.0, (max_h as f64 - 1.0) / 2.0);
        let px = |p: &Plane<u8>, x: usize, y: usize| {
            p.data()[p.geometry().data_origin() + y * p.geometry().stride() + x]
        };
        let (mut sum, mut su, mut sv, mut suu, mut svv) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for y in 0..max_h {
            for x in 0..max_w {
                let (u, v) = (x as f64 - cx, y as f64 - cy);
                let d = f64::from(px(source, x_o + x, y_o + y))
                    - f64::from(px(denoised, x_o + x, y_o + y));
                sum += d;
                su += u * d;
                sv += v * d;
                suu += u * u;
                svv += v * v;
            }
        }
        let slope = |num: f64, den: f64| if den > 0f64 { num / den } else { 0f64 };
        [sum / (max_w * max_h) as f64, slope(su, suu), slope(sv, svv)]
    }

    #[test]
    fn integer_plane_fit_is_bit_identical_to_the_f64_sums() {
        let (w, h) = (100usize, 70usize);
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut byte = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 56) as u8
        };
        let mut src = FrameBuilder::new(w, h, ChromaSubsampling::Monochrome, 8)
            .build()
            .expect("monochrome frame builds");
        let mut den = src.clone();
        let pixels = std::iter::repeat_with(&mut byte)
            .take(w * h)
            .collect::<Vec<_>>();
        src.y_plane.copy_from_slice(&pixels).expect("plane fits");
        let pixels = std::iter::repeat_with(&mut byte)
            .take(w * h)
            .collect::<Vec<_>>();
        den.y_plane.copy_from_slice(&pixels).expect("plane fits");
        // Full blocks and blocks cut off at the right and bottom edges.
        for (x_o, y_o) in [(0, 0), (32, 32), (64, 0), (0, 64), (96, 64), (75, 50)] {
            let fit = DiffPlane::fit(&src.y_plane, &den.y_plane, (w, h), x_o, y_o, 32, 32);
            let want = f64_fit(&src.y_plane, &den.y_plane, (w, h), (x_o, y_o));
            let got = [fit.mean, fit.gx, fit.gy];
            assert_eq!(
                got.map(f64::to_bits),
                want.map(f64::to_bits),
                "block at {x_o},{y_o}"
            );
        }
    }
}
