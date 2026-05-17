use std::ptr;

use v_frame::plane::Plane;

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
    // SAFETY: We need to ensure that `n` doesn't exceed the bounds of these arrays.
    // But this is a crate-private function, so we control all input.
    unsafe {
        // Forward elimination
        for k in 0..(n - 1) {
            // Bring the largest magnitude to the diagonal position
            ((k + 1)..n).rev().for_each(|i| {
                if get_dbg(a, (i - 1) * stride + k).abs() < get_dbg(a, i * stride + k).abs() {
                    (0..n).for_each(|j| {
                        swap_unchecked(a, i * stride + j, (i - 1) * stride + j);
                    });
                    swap_unchecked(b, i, i - 1);
                }
            });

            for i in k..(n - 1) {
                if get_dbg(a, k * stride + k).abs() < f64::EPSILON {
                    return false;
                }
                let c = *get_dbg(a, (i + 1) * stride + k) / *get_dbg(a, k * stride + k);
                (0..n).for_each(|j| {
                    let a2_val = *get_dbg(a, k * stride + j);
                    let a_val = get_dbg_mut(a, (i + 1) * stride + j);
                    *a_val = c.mul_add(-a2_val, *a_val);
                });
                let b2_val = *get_dbg(b, k);
                let b_val = get_dbg_mut(b, i + 1);
                *b_val = c.mul_add(-b2_val, *b_val);
            }
        }

        // Backward substitution
        for i in (0..n).rev() {
            if get_dbg(a, i * stride + i).abs() < f64::EPSILON {
                return false;
            }
            let mut c = 0.0f64;
            for j in (i + 1)..n {
                c = get_dbg(a, i * stride + j).mul_add(*get_dbg(x, j), c);
            }
            *get_dbg_mut(x, i) = (*get_dbg(b, i) - c) / *get_dbg(a, i * stride + i);
        }
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
    let mut idx = 0;
    for row in 0..m1_rows {
        for col in 0..m2_cols {
            let mut sum = 0f64;
            for inner in 0..inner_dim {
                sum += get_dbg(m1, row * inner_dim + inner) * get_dbg(m2, inner * m2_cols + col);
            }

            *get_dbg_mut(res, idx) = sum;
            idx += 1;
        }
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

    let data_origin = get_dbg(source.data(), source.data_origin()..);
    let stride = source.geometry().stride.get();
    let mut block_sum = 0u64;
    for y in 0..max_h {
        let row_start = (y_o + y) * stride + x_o;
        let row = get_dbg(data_origin, row_start..row_start + max_w);
        for pixel in row {
            block_sum += u64::from(*pixel);
        }
    }

    block_sum as f64 / (max_w * max_h) as f64
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

    let source_origin = get_dbg(source.data(), source.data_origin()..);
    let denoised_origin = get_dbg(denoised.data(), denoised.data_origin()..);
    let stride = source.geometry().stride.get();
    let mut noise_var_sum = 0u64;
    let mut noise_sum = 0i64;
    for y in 0..max_h {
        let row_start = (y_o + y) * stride + x_o;
        let source_row = get_dbg(source_origin, row_start..row_start + max_w);
        let denoised_row = get_dbg(denoised_origin, row_start..row_start + max_w);
        for (source, denoised) in source_row.iter().zip(denoised_row.iter()) {
            let noise = i64::from(*source) - i64::from(*denoised);
            noise_sum += noise;
            noise_var_sum += noise.pow(2) as u64;
        }
    }

    let noise_mean = noise_sum as f64 / (max_w * max_h) as f64;
    noise_mean.mul_add(-noise_mean, noise_var_sum as f64 / (max_w * max_h) as f64)
}
