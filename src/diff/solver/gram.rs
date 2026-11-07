//! Normal equations of the AR fit for one block, from exact integer sums.
//!
//! Every AR input is `source - denoised` at a fixed offset, minus the block's fitted plane at
//! that pixel. The plane is affine, so the per-pixel products expand into integer dot products
//! of shifted difference rows plus a closed-form correction per block. The integer part
//! vectorises; the old per-pixel f64 outer product did not.

use super::util::DiffPlane;
use crate::util::{get_dbg, get_dbg_mut};

/// The co-located luma plane, for the extra chroma input.
pub(super) struct Alt<'a> {
    pub source: &'a [u8],
    pub denoised: &'a [u8],
    pub stride: usize,
    /// log2 chroma subsampling (x, y).
    pub dec: (usize, usize),
    pub plane: DiffPlane,
}

/// Block-local pixel window `[x0, x1) x [y0, y1)` the observations come from.
pub(super) struct Window {
    pub x0: usize,
    pub x1: usize,
    pub y0: usize,
    pub y1: usize,
}

/// One AR input: `scale * values - (alpha + beta * u + gamma * v)` at block-local `(u, v)`.
struct Input {
    start: usize,
    scale: f64,
    alpha: f64,
    beta: f64,
    gamma: f64,
}

/// Fills the upper triangle of `a` (`n` x `n`) and `b` for the block's observations and returns
/// `(observations, residual_energy)`. `source` and `denoised` start at the plane's data origin.
#[allow(clippy::too_many_arguments)]
pub(super) fn block_equations(
    source: &[u8],
    denoised: &[u8],
    stride: usize,
    (x_o, y_o): (usize, usize),
    win: &Window,
    coords: &[[isize; 2]],
    plane: &DiffPlane,
    alt: Option<&Alt>,
    n: usize,
    a: &mut [f64],
    b: &mut [f64],
    values: &mut Vec<i16>,
) -> (usize, f64) {
    if win.x1 <= win.x0 || win.y1 <= win.y0 {
        return (0, 0f64);
    }
    let (w, h) = (win.x1 - win.x0, win.y1 - win.y0);
    let len = w * h;
    // Keeps every integer dot product, at most 1020^2 per pixel, inside i32.
    debug_assert!(len <= 1024);
    // Every difference the taps reach, computed once: the window grown by the tap offsets.
    let taps = || coords.iter().chain(std::iter::once(&[0, 0]));
    let left = taps().map(|c| c[0]).min().unwrap_or(0);
    let right = taps().map(|c| c[0]).max().unwrap_or(0);
    let top = taps().map(|c| c[1]).min().unwrap_or(0);
    let bottom = taps().map(|c| c[1]).max().unwrap_or(0);
    let x_first = (x_o + win.x0) as isize + left;
    let y_first = (y_o + win.y0) as isize + top;
    let grown_w = (w as isize + right - left) as usize;
    let grown_h = (h as isize + bottom - top) as usize;
    values.clear();
    for y in 0..grown_h {
        let row = (y_first as usize + y) * stride + x_first as usize;
        let (s, d) = (
            get_dbg(source, row..row + grown_w),
            get_dbg(denoised, row..row + grown_w),
        );
        values.extend(s.iter().zip(d).map(|(&s, &d)| i16::from(s) - i16::from(d)));
    }

    // Inputs: the AR taps, the luma term for chroma, then the target pixel last.
    let mut inputs = Vec::with_capacity(n + 1);
    for &[dx, dy] in taps() {
        if let Some(alt) = alt.filter(|_| inputs.len() == coords.len()) {
            inputs.push(alt_input(alt, (x_o, y_o), win, values));
        }
        let start = values.len();
        for y in 0..h {
            let row = (y as isize + dy - top) as usize * grown_w + (dx - left) as usize;
            values.extend_from_within(row..row + w);
        }
        inputs.push(Input {
            start,
            scale: 1f64,
            alpha: plane.at((x_o as isize + dx) as f64, (y_o as isize + dy) as f64),
            beta: plane.step(1, 0),
            gamma: plane.step(0, 1),
        });
    }
    debug_assert_eq!(inputs.len(), n + 1);

    // Per-input sums of value, value * u and value * v, as dot products with 1, u and v.
    let mut weights = Vec::with_capacity(3 * len);
    weights.resize(len, 1i16);
    for _ in win.y0..win.y1 {
        weights.extend((win.x0..win.x1).map(|u| u as i16));
    }
    for v in win.y0..win.y1 {
        weights.extend(std::iter::repeat_n(v as i16, w));
    }
    let (ones, rest) = weights.split_at(len);
    let (us, vs) = rest.split_at(len);
    let moments = inputs
        .iter()
        .map(|input| {
            let vals = get_dbg(values, input.start..input.start + len);
            [ones, us, vs].map(|weight| f64::from(dot(vals, weight)) * input.scale)
        })
        .collect::<Vec<_>>();

    // Window sums of 1, u, v, u^2, u v, v^2.
    let sum_to = |end: usize| (end * end.saturating_sub(1) / 2) as f64;
    let sq_sum_to =
        |end: usize| (end.saturating_sub(1) * end * (2 * end).saturating_sub(1) / 6) as f64;
    let su1 = sum_to(win.x1) - sum_to(win.x0);
    let sv1 = sum_to(win.y1) - sum_to(win.y0);
    let suu1 = sq_sum_to(win.x1) - sq_sum_to(win.x0);
    let svv1 = sq_sum_to(win.y1) - sq_sum_to(win.y0);
    let (count, su, sv) = (len as f64, su1 * h as f64, sv1 * w as f64);
    let (suu, svv, suv) = (suu1 * h as f64, svv1 * w as f64, su1 * sv1);

    let product = |i: usize, j: usize| {
        let (p, q) = (get_dbg(&inputs, i), get_dbg(&inputs, j));
        let dot = dot(
            get_dbg(values, p.start..p.start + len),
            get_dbg(values, q.start..q.start + len),
        );
        let [ps, psu, psv] = *get_dbg(&moments, i);
        let [qs, qsu, qsv] = *get_dbg(&moments, j);
        let plane_plane = p.alpha * q.alpha * count
            + (p.alpha * q.beta + p.beta * q.alpha) * su
            + (p.alpha * q.gamma + p.gamma * q.alpha) * sv
            + p.beta * q.beta * suu
            + (p.beta * q.gamma + p.gamma * q.beta) * suv
            + p.gamma * q.gamma * svv;
        f64::from(dot) * p.scale * q.scale
            - (q.alpha * ps + q.beta * psu + q.gamma * psv)
            - (p.alpha * qs + p.beta * qsu + p.gamma * qsv)
            + plane_plane
    };

    for i in 0..n {
        for j in i..n {
            *get_dbg_mut(a, i * n + j) += product(i, j);
        }
        *get_dbg_mut(b, i) += product(i, n);
    }
    (len, product(n, n))
}

/// The chroma luma term: mean co-located luma difference minus the luma block's plane.
fn alt_input(alt: &Alt, (x_o, y_o): (usize, usize), win: &Window, values: &mut Vec<i16>) -> Input {
    let start = values.len();
    let (sx, sy) = (1usize << alt.dec.0, 1usize << alt.dec.1);
    for y in win.y0..win.y1 {
        for x in win.x0..win.x1 {
            let mut sum = 0i16;
            for dy in 0..sy {
                let row = ((y_o + y) * sy + dy) * alt.stride + (x_o + x) * sx;
                for index in row..row + sx {
                    sum += i16::from(*get_dbg(alt.source, index))
                        - i16::from(*get_dbg(alt.denoised, index));
                }
            }
            values.push(sum);
        }
    }
    let (half_x, half_y) = (0.5f64 * alt.dec.0 as f64, 0.5f64 * alt.dec.1 as f64);
    Input {
        start,
        scale: 1f64 / (sx * sy) as f64,
        alpha: alt
            .plane
            .at((x_o * sx) as f64 + half_x, (y_o * sy) as f64 + half_y),
        beta: alt.plane.step(sx as isize, 0),
        gamma: alt.plane.step(0, sy as isize),
    }
}

/// Integer dot product; the caller keeps the sum inside i32.
#[cfg(target_arch = "x86_64")]
fn dot(a: &[i16], b: &[i16]) -> i32 {
    use std::arch::x86_64::{
        __m128i, _mm_add_epi32, _mm_cvtsi128_si32, _mm_loadu_si128, _mm_madd_epi16,
        _mm_setzero_si128, _mm_shuffle_epi32,
    };

    let len = a.len().min(b.len());
    let split = len - len % 8;
    // SAFETY: SSE2 is part of the x86_64 baseline, and every 8-lane load stays below `split`.
    let mut total = unsafe {
        let mut acc = _mm_setzero_si128();
        for i in (0..split).step_by(8) {
            let x = _mm_loadu_si128(a.as_ptr().add(i).cast::<__m128i>());
            let y = _mm_loadu_si128(b.as_ptr().add(i).cast::<__m128i>());
            acc = _mm_add_epi32(acc, _mm_madd_epi16(x, y));
        }
        acc = _mm_add_epi32(acc, _mm_shuffle_epi32::<0b0100_1110>(acc));
        acc = _mm_add_epi32(acc, _mm_shuffle_epi32::<0b1011_0001>(acc));
        _mm_cvtsi128_si32(acc)
    };
    for (x, y) in get_dbg(a, split..len).iter().zip(get_dbg(b, split..len)) {
        total += i32::from(*x) * i32::from(*y);
    }
    total
}

/// Integer dot product; the caller keeps the sum inside i32.
#[cfg(not(target_arch = "x86_64"))]
fn dot(a: &[i16], b: &[i16]) -> i32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| i32::from(x) * i32::from(y))
        .sum()
}
