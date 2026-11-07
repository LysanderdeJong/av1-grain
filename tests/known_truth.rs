//! Known-truth integration tests for `av1_grain::DiffGenerator`.
//!
//! Each test renders 48 synthetic 512x256 4:2:0 frame pairs whose film grain is
//! known, runs the estimator over them, and checks the fitted parameters
//! against that truth.
//!
//! Conventions used by the checks:
//! - Grain std at level x is the piecewise-linear interpolation of the scaling
//!   points at x, divided by 2^(scaling_shift - 5). Outside the point range the
//!   end value is used.
//! - AR coefficient i is `ar_coeffs_y[i] / 2^ar_coeff_shift`. Index 23 is the
//!   left neighbour (x-1, same row).
//! - Chroma scaling points are indexed by luma level, because the solver takes
//!   its block means from luma. "Level 128" for Cb/Cr therefore means luma 128.
//! - Measured std of an integer-rounded noise diff is sqrt(sigma^2 + 1/12).

use std::fmt::Debug;

use av1_grain::{
    DiffGenerator, GrainTableSegment,
    v_frame::{
        chroma::ChromaSubsampling,
        frame::{Frame, FrameBuilder},
    },
};
use num_rational::Rational64;

const W: usize = 512;
const H: usize = 256;
const CW: usize = W / 2;
const CH: usize = H / 2;
const FRAMES: usize = 48;

/// Flat strips, one per 64 px column, in the rows above the gradient.
const STRIP_LEVELS: [u8; 8] = [32, 56, 80, 104, 128, 152, 176, 200];
const STRIP_W: usize = 64;
/// The bottom rows hold a horizontal gradient instead of strips.
const GRADIENT_ROWS: usize = 64;
/// Rows above the gradient (the strip region).
const FLAT_ROWS: usize = H - GRADIENT_ROWS;
const EDGE_XS: [usize; 3] = [100, 300, 420];
const EDGE_W: usize = 3;
const EDGE_STEP: u8 = 60;
const EDGE_ROWS: usize = 128;
/// Bright patch rectangle as (x0, x1, y0, y1), end-exclusive.
const PATCH: (usize, usize, usize, usize) = (192, 320, 64, 192);
const PATCH_LEVEL: u8 = 220;

/// Extra variance from rounding a continuous value to an integer.
const QUANT_VAR: f64 = 1.0 / 12.0;

/// Levels printed for the patch case.
const PATCH_PRINT_LEVELS: [u8; 9] = [32, 56, 80, 104, 128, 152, 176, 200, 220];

type Pair = (Frame<u8>, Frame<u8>);

/// xorshift64* with a splitmix64 seed scramble, so nearby seeds give unrelated streams.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // xorshift state must be non-zero.
        Self(z.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [0, 1).
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal sample via Box-Muller.
    fn gaussian(&mut self) -> f64 {
        let u1 = 1.0 - self.uniform(); // in (0, 1], so ln is finite
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Independent stream for each (case, frame, plane).
fn rng_for(case: u64, frame: usize, plane: u64) -> Rng {
    Rng::new((case << 40) ^ ((frame as u64) << 8) ^ plane)
}

/// Round to the nearest integer and clamp to 8 bits.
fn to_u8(v: f64) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

/// Independent white Gaussian noise, `n` samples with the given std.
fn white(rng: &mut Rng, n: usize, std: f64) -> Vec<f64> {
    std::iter::repeat_with(|| std * rng.gaussian())
        .take(n)
        .collect()
}

/// White noise whose std depends on each pixel's clean value.
fn white_by_level(rng: &mut Rng, clean: &[u8], std_of: impl Fn(f64) -> f64) -> Vec<f64> {
    clean
        .iter()
        .map(|&c| std_of(f64::from(c)) * rng.gaussian())
        .collect()
}

/// AR(1) along each row: n[x] = coef * n[x-1] + e[x], with e ~ N(0, innov_std^2) and n[-1] = 0.
fn ar1_rows(rng: &mut Rng, w: usize, h: usize, coef: f64, innov_std: f64) -> Vec<f64> {
    let mut out = vec![0.0; w * h];
    for row in out.chunks_exact_mut(w) {
        let mut prev = 0.0;
        for v in row {
            prev = coef * prev + innov_std * rng.gaussian();
            *v = prev;
        }
    }
    out
}

/// Integer frame samples: round(clean + noise), clamped.
fn quantize(clean: &[u8], noise: &[f64]) -> Vec<u8> {
    clean
        .iter()
        .zip(noise)
        .map(|(&c, &n)| to_u8(f64::from(c) + n))
        .collect()
}

/// Plane of `w` x `h` samples computed from (x, y), rounded and clamped.
fn grid(w: usize, h: usize, f: impl Fn(usize, usize) -> f64) -> Vec<u8> {
    (0..w * h).map(|i| to_u8(f(i % w, i / w))).collect()
}

fn flat(n: usize, value: u8) -> Vec<u8> {
    vec![value; n]
}

/// Bottom-rows gradient: 40 at x = 0 to 200 at the right edge.
fn gradient(x: usize) -> u8 {
    (40.0 + 160.0 * x as f64 / (W - 1) as f64).round() as u8
}

/// Clean luma: 8 strips, a gradient in the bottom rows, three +60 edges, and an
/// optional bright patch. The patch overwrites whatever is underneath it.
fn clean_luma(with_patch: bool) -> Vec<u8> {
    let mut y = vec![0u8; W * H];
    let (rows, _) = y.as_chunks_mut::<W>();
    for (row_idx, row) in rows.iter_mut().enumerate() {
        for (x, px) in row.iter_mut().enumerate() {
            *px = if row_idx >= FLAT_ROWS {
                gradient(x)
            } else {
                STRIP_LEVELS[x / STRIP_W]
            };
        }
    }
    for &ex in &EDGE_XS {
        for row in rows.iter_mut().take(EDGE_ROWS) {
            for px in &mut row[ex..ex + EDGE_W] {
                *px = px.saturating_add(EDGE_STEP);
            }
        }
    }
    if with_patch {
        let (x0, x1, y0, y1) = PATCH;
        for row in rows.iter_mut().skip(y0).take(y1 - y0) {
            row[x0..x1].fill(PATCH_LEVEL);
        }
    }
    y
}

/// True for pixels 1 or 2 px (Chebyshev distance) outside the patch rectangle.
fn in_patch_ring(x: usize, y: usize) -> bool {
    let (x0, x1, y0, y1) = PATCH;
    let gap = |v: usize, lo: usize, hi: usize| {
        if v < lo {
            lo - v
        } else if v >= hi {
            v + 1 - hi
        } else {
            0
        }
    };
    (1..=2).contains(&gap(x, x0, x1).max(gap(y, y0, y1)))
}

/// Gaussian blob: amplitude 6, sigma 80 px, centred at (256, 128).
fn blob(x: usize, y: usize) -> f64 {
    let dx = x as f64 - 256.0;
    let dy = y as f64 - 128.0;
    6.0 * (-(dx * dx + dy * dy) / (2.0 * 80.0 * 80.0)).exp()
}

/// Noise std for K2: 4.0 at level 32 falling linearly to 0.5 at level 200, clamped outside.
fn k2_sigma(level: f64) -> f64 {
    4.0 + (0.5 - 4.0) * ((level - 32.0) / (200.0 - 32.0)).clamp(0.0, 1.0)
}

/// Builds a 4:2:0 8-bit frame from full planes, then checks that the visible
/// pixels read back exactly as written (guards against stride/padding mistakes).
fn frame(y: &[u8], cb: &[u8], cr: &[u8]) -> Frame<u8> {
    let mut f: Frame<u8> = FrameBuilder::new(W, H, ChromaSubsampling::Yuv420, 8)
        .build()
        .expect("512x256 4:2:0 8-bit frame builds");
    f.y_plane.copy_from_slice(y).expect("luma length is W*H");
    f.u_plane
        .as_mut()
        .expect("4:2:0 has a Cb plane")
        .copy_from_slice(cb)
        .expect("Cb length is CW*CH");
    f.v_plane
        .as_mut()
        .expect("4:2:0 has a Cr plane")
        .copy_from_slice(cr)
        .expect("Cr length is CW*CH");

    let round_trip = f.y_plane.pixels().eq(y.iter().copied())
        && f.u_plane
            .as_ref()
            .is_some_and(|p| p.pixels().eq(cb.iter().copied()))
        && f.v_plane
            .as_ref()
            .is_some_and(|p| p.pixels().eq(cr.iter().copied()));
    assert!(
        round_trip,
        "frame round-trip failed: visible pixels do not match the planes written"
    );
    f
}

/// Piecewise-linear grain std at `level`, end values outside the points.
fn grain_std(points: &[[u8; 2]], scaling_shift: u8, level: f64) -> f64 {
    let (Some(first), Some(last)) = (points.first(), points.last()) else {
        return 0.0;
    };
    let y = if level <= f64::from(first[0]) {
        f64::from(first[1])
    } else if level >= f64::from(last[0]) {
        f64::from(last[1])
    } else {
        let i = points.partition_point(|p| f64::from(p[0]) < level);
        let (a, b) = (points[i - 1], points[i]);
        let t = (level - f64::from(a[0])) / (f64::from(b[0]) - f64::from(a[0]));
        f64::from(a[1]) + t * (f64::from(b[1]) - f64::from(a[1]))
    };
    y / 2f64.powi(i32::from(scaling_shift) - 5)
}

fn grain_std_y(seg: &GrainTableSegment, level: f64) -> f64 {
    grain_std(&seg.scaling_points_y, seg.scaling_shift, level)
}

/// Largest knot of a chroma curve in grain std units; zero when there are no points.
fn chroma_max_std(points: &[[u8; 2]], scaling_shift: u8) -> f64 {
    let max_y = points.iter().map(|p| f64::from(p[1])).fold(0.0, f64::max);
    max_y / 2f64.powi(i32::from(scaling_shift) - 5)
}

/// Luma AR coefficient `i` in real units (NaN if missing).
fn ar_luma(seg: &GrainTableSegment, i: usize) -> f64 {
    seg.ar_coeffs_y.get(i).map_or(f64::NAN, |&c| {
        f64::from(c) / 2f64.powi(i32::from(seg.ar_coeff_shift))
    })
}

/// Sum of all luma AR coefficients in real units.
fn ar_sum(seg: &GrainTableSegment) -> f64 {
    let raw: f64 = seg.ar_coeffs_y.iter().map(|&c| f64::from(c)).sum();
    raw / 2f64.powi(i32::from(seg.ar_coeff_shift))
}

struct Estimate {
    segments: Vec<GrainTableSegment>,
    frames_without_estimate: usize,
}

fn estimate(pairs: &[Pair], boundaries: &[usize]) -> Estimate {
    let mut generator = DiffGenerator::new(Rational64::new(24, 1), 8, 8);
    generator.set_boundaries(boundaries.to_vec());
    for (src, den) in pairs {
        generator
            .diff_frame(src, den)
            .expect("8-bit frames of equal size");
    }
    let frames_without_estimate = generator.frames_without_estimate();
    Estimate {
        segments: generator.finish(),
        frames_without_estimate,
    }
}

fn first_segment(est: &Estimate) -> &GrainTableSegment {
    est.segments
        .first()
        .expect("estimator returned at least one segment")
}

/// Per clean-luma level over all frames, flat rows only: (count, mean, std) of
/// noisy - denoised. Measured directly from the pixels, for comparison with the fit.
fn empirical_by_level(pairs: &[Pair], clean_y: &[u8]) -> Vec<(u64, f64, f64)> {
    // (count, sum, sum of squares) per level
    let mut acc = vec![(0u64, 0f64, 0f64); 256];
    for (src, den) in pairs {
        for (i, (s, d)) in src.y_plane.pixels().zip(den.y_plane.pixels()).enumerate() {
            if i / W >= FLAT_ROWS {
                continue;
            }
            let diff = f64::from(s) - f64::from(d);
            let slot = &mut acc[usize::from(clean_y[i])];
            slot.0 += 1;
            slot.1 += diff;
            slot.2 += diff * diff;
        }
    }
    acc.into_iter()
        .map(|(n, sum, sq)| {
            if n == 0 {
                return (0, 0.0, 0.0);
            }
            let mean = sum / n as f64;
            (n, mean, (sq / n as f64 - mean * mean).max(0.0).sqrt())
        })
        .collect()
}

/// One line per segment with fitted std at each level (and empirical std when
/// `emp` is given), plus AR and chroma summaries.
fn summarize(case: &str, est: &Estimate, emp: Option<&[(u64, f64, f64)]>, levels: &[u8]) {
    eprintln!(
        "[{case}] segments={} frames_without_estimate={}",
        est.segments.len(),
        est.frames_without_estimate
    );
    for (i, s) in est.segments.iter().enumerate() {
        let per_level = levels
            .iter()
            .map(|&l| {
                let fit = grain_std_y(s, f64::from(l));
                emp.map_or_else(
                    || format!("{l}:{fit:.3}"),
                    |e| format!("{l}:{fit:.3}/{:.3}", e[usize::from(l)].2),
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        let label = if emp.is_some() { "fit/emp" } else { "fit" };
        eprintln!(
            "[{case}] seg{i} start={} end={} lag={} scale_shift={} ar_shift={} y_pts={} cb_pts={} cr_pts={} | {label} std {per_level} | ar_left={:.4} ar_sum={:.4} cb_max={:.4} cr_max={:.4}",
            s.start_time,
            s.end_time,
            s.ar_coeff_lag,
            s.scaling_shift,
            s.ar_coeff_shift,
            s.scaling_points_y.len(),
            s.scaling_points_cb.len(),
            s.scaling_points_cr.len(),
            ar_luma(s, 23),
            ar_sum(s),
            chroma_max_std(&s.scaling_points_cb, s.scaling_shift),
            chroma_max_std(&s.scaling_points_cr, s.scaling_shift),
        );
    }
}

/// Collects every failed check so one run reports all of them.
#[derive(Default)]
struct Checks {
    failures: Vec<String>,
}

impl Checks {
    fn fail(&mut self, msg: String) {
        self.failures.push(msg);
    }

    /// Relative error of `measured` against `expected` is at most `tol`.
    fn rel(&mut self, label: &str, measured: f64, expected: f64, tol: f64) {
        let err = (measured - expected) / expected;
        if err.is_nan() || err.abs() > tol {
            self.fail(format!(
                "{label}: measured {measured:.4}, expected {expected:.4}, rel err {:+.1}% (tol {:.0}%)",
                err * 100.0,
                tol * 100.0
            ));
        }
    }

    /// `measured` is within `tol` (absolute) of `expected`.
    fn near(&mut self, label: &str, measured: f64, expected: f64, tol: f64) {
        let err = (measured - expected).abs();
        if err.is_nan() || err > tol {
            self.fail(format!(
                "{label}: measured {measured:.4}, expected {expected:.4} +/- {tol:.3}"
            ));
        }
    }

    /// `value <= bound`.
    fn le(&mut self, label: &str, value: f64, bound: f64) {
        if value.is_nan() || value > bound {
            self.fail(format!("{label}: value {value:.4}, bound {bound:.4}"));
        }
    }

    /// `|value| <= bound`.
    fn abs_le(&mut self, label: &str, value: f64, bound: f64) {
        if value.is_nan() || value.abs() > bound {
            self.fail(format!(
                "{label}: value {value:.4}, required |value| <= {bound:.4}"
            ));
        }
    }

    fn exact<T: Copy + PartialEq + Debug>(&mut self, label: &str, got: T, want: T) {
        if got != want {
            self.fail(format!("{label}: got {got:?}, want {want:?}"));
        }
    }

    fn finish(&self, case: &str) {
        assert!(
            self.failures.is_empty(),
            "{case}: {} check(s) failed:\n  {}",
            self.failures.len(),
            self.failures.join("\n  ")
        );
    }
}

/// Both chroma curves stay below 0.3 std (an empty curve counts as 0).
fn check_chroma_quiet(checks: &mut Checks, case: &str, seg: &GrainTableSegment) {
    let cb = chroma_max_std(&seg.scaling_points_cb, seg.scaling_shift);
    let cr = chroma_max_std(&seg.scaling_points_cr, seg.scaling_shift);
    checks.le(&format!("{case} Cb max std"), cb, 0.3);
    checks.le(&format!("{case} Cr max std"), cr, 0.3);
}

/// Expected measured std for a white noise of std `sigma` after integer rounding.
fn expected_std(sigma: f64) -> f64 {
    (sigma * sigma + QUANT_VAR).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// K1: white luma noise std 3, denoised is clean.
    #[test]
    fn k1_white_luma_noise_recovered() {
        let clean_y = clean_luma(false);
        let clean_c = flat(CW * CH, 128);
        let pairs: Vec<Pair> = (0..FRAMES)
            .map(|i| {
                let noise = white(&mut rng_for(1, i, 0), W * H, 3.0);
                (
                    frame(&quantize(&clean_y, &noise), &clean_c, &clean_c),
                    frame(&clean_y, &clean_c, &clean_c),
                )
            })
            .collect();
        let est = estimate(&pairs, &[]);
        let emp = empirical_by_level(&pairs, &clean_y);
        summarize("K1", &est, Some(&emp), &STRIP_LEVELS);

        let mut checks = Checks::default();
        checks.exact("K1 segment count", est.segments.len(), 1);
        let seg = first_segment(&est);
        for &l in &STRIP_LEVELS {
            checks.rel(
                &format!("K1 luma std @{l}"),
                grain_std_y(seg, f64::from(l)),
                expected_std(3.0),
                0.10,
            );
        }
        check_chroma_quiet(&mut checks, "K1", seg);
        checks.abs_le("K1 luma AR sum", ar_sum(seg), 0.05);
        checks.finish("K1");
    }

    /// K2: luma noise std falls linearly with clean level (4.0 at 32 to 0.5 at 200).
    #[test]
    fn k2_level_dependent_luma_noise_recovered() {
        let clean_y = clean_luma(false);
        let clean_c = flat(CW * CH, 128);
        let pairs: Vec<Pair> = (0..FRAMES)
            .map(|i| {
                let noise = white_by_level(&mut rng_for(2, i, 0), &clean_y, k2_sigma);
                (
                    frame(&quantize(&clean_y, &noise), &clean_c, &clean_c),
                    frame(&clean_y, &clean_c, &clean_c),
                )
            })
            .collect();
        let est = estimate(&pairs, &[]);
        let emp = empirical_by_level(&pairs, &clean_y);
        summarize("K2", &est, Some(&emp), &STRIP_LEVELS);

        let mut checks = Checks::default();
        let seg = first_segment(&est);
        for &l in &STRIP_LEVELS {
            checks.rel(
                &format!("K2 luma std @{l}"),
                grain_std_y(seg, f64::from(l)),
                expected_std(k2_sigma(f64::from(l))),
                0.15,
            );
        }
        // No clean content exists at 230, so this compares the fitted curve there
        // (end value, clamped) against the fitted value at 200.
        let at_230 = grain_std_y(seg, 230.0);
        let at_200 = grain_std_y(seg, 200.0);
        checks.le("K2 luma std @230 vs 1.15 * std @200", at_230, 1.15 * at_200);
        checks.finish("K2");
    }

    /// K3: horizontal AR(1) noise, coefficient 0.5, innovation std 3.
    #[test]
    fn k3_ar1_horizontal_coefficient_recovered() {
        let clean_y = clean_luma(false);
        let clean_c = flat(CW * CH, 128);
        let pairs: Vec<Pair> = (0..FRAMES)
            .map(|i| {
                let noise = ar1_rows(&mut rng_for(3, i, 0), W, H, 0.5, 3.0);
                (
                    frame(&quantize(&clean_y, &noise), &clean_c, &clean_c),
                    frame(&clean_y, &clean_c, &clean_c),
                )
            })
            .collect();
        let est = estimate(&pairs, &[]);
        let emp = empirical_by_level(&pairs, &clean_y);
        summarize("K3", &est, Some(&emp), &STRIP_LEVELS);

        let mut checks = Checks::default();
        let seg = first_segment(&est);
        checks.near(
            "K3 left AR coefficient (index 23)",
            ar_luma(seg, 23),
            0.5,
            0.05,
        );
        checks.finish("K3");
    }

    /// K4: white luma noise std 3, denoised has a +4 offset and a +6 blob.
    #[test]
    fn k4_offset_and_blob_in_denoised_ignored() {
        let clean_y = clean_luma(false);
        let clean_c = flat(CW * CH, 128);
        let den_y = grid(W, H, |x, y| {
            f64::from(clean_y[y * W + x]) + 4.0 + blob(x, y)
        });
        let pairs: Vec<Pair> = (0..FRAMES)
            .map(|i| {
                let noise = white(&mut rng_for(4, i, 0), W * H, 3.0);
                (
                    frame(&quantize(&clean_y, &noise), &clean_c, &clean_c),
                    frame(&den_y, &clean_c, &clean_c),
                )
            })
            .collect();
        let est = estimate(&pairs, &[]);
        let emp = empirical_by_level(&pairs, &clean_y);
        summarize("K4", &est, Some(&emp), &STRIP_LEVELS);

        let mut checks = Checks::default();
        let seg = first_segment(&est);
        for &l in &STRIP_LEVELS {
            checks.rel(
                &format!("K4 luma std @{l}"),
                grain_std_y(seg, f64::from(l)),
                expected_std(3.0),
                0.10,
            );
        }
        // The per-block plane removes the offset and the blob's slope; its curvature inside a
        // 32-pixel block still leaks a little (baseline: 0.99).
        checks.abs_le("K4 luma AR sum", ar_sum(seg), 0.10);
        checks.finish("K4");
    }

    /// K5: white luma noise std 3, denoised chroma is offset (Cb) and ramped (Cr).
    #[test]
    fn k5_chroma_bias_in_denoised_ignored() {
        let clean_y = clean_luma(false);
        let clean_c = flat(CW * CH, 128);
        let cb_den = flat(CW * CH, 131);
        let cr_den = grid(CW, CH, |x, _| {
            128.0 + (-3.0 + 6.0 * x as f64 / (CW - 1) as f64).round()
        });
        let pairs: Vec<Pair> = (0..FRAMES)
            .map(|i| {
                let noise = white(&mut rng_for(5, i, 0), W * H, 3.0);
                (
                    frame(&quantize(&clean_y, &noise), &clean_c, &clean_c),
                    frame(&clean_y, &cb_den, &cr_den),
                )
            })
            .collect();
        let est = estimate(&pairs, &[]);
        let emp = empirical_by_level(&pairs, &clean_y);
        summarize("K5", &est, Some(&emp), &STRIP_LEVELS);

        let mut checks = Checks::default();
        check_chroma_quiet(&mut checks, "K5", first_segment(&est));
        checks.finish("K5");
    }

    /// K6: independent luma std 3, Cb and Cr std 2.
    #[test]
    fn k6_chroma_noise_recovered() {
        let clean_y = clean_luma(false);
        let clean_c = flat(CW * CH, 128);
        let pairs: Vec<Pair> = (0..FRAMES)
            .map(|i| {
                let ny = white(&mut rng_for(6, i, 0), W * H, 3.0);
                let ncb = white(&mut rng_for(6, i, 1), CW * CH, 2.0);
                let ncr = white(&mut rng_for(6, i, 2), CW * CH, 2.0);
                (
                    frame(
                        &quantize(&clean_y, &ny),
                        &quantize(&clean_c, &ncb),
                        &quantize(&clean_c, &ncr),
                    ),
                    frame(&clean_y, &clean_c, &clean_c),
                )
            })
            .collect();
        let est = estimate(&pairs, &[]);
        let emp = empirical_by_level(&pairs, &clean_y);
        summarize("K6", &est, Some(&emp), &STRIP_LEVELS);

        let mut checks = Checks::default();
        let seg = first_segment(&est);
        // Chroma curves are indexed by luma level, so 128 is the luma 128 strip.
        let expected = expected_std(2.0);
        checks.rel(
            "K6 Cb std @128",
            grain_std(&seg.scaling_points_cb, seg.scaling_shift, 128.0),
            expected,
            0.15,
        );
        checks.rel(
            "K6 Cr std @128",
            grain_std(&seg.scaling_points_cr, seg.scaling_shift, 128.0),
            expected,
            0.15,
        );
        checks.finish("K6");
    }

    /// K7: three 16-frame shots (luma std 1, 3, 6), boundaries at 0, 16, 32.
    #[test]
    fn k7_shot_boundaries_give_one_segment_per_shot() {
        const SHOT_STD: [f64; 3] = [1.0, 3.0, 6.0];
        const SHOT_START: [usize; 3] = [0, 16, 32];
        let clean_y = clean_luma(false);
        let clean_c = flat(CW * CH, 128);
        let pairs: Vec<Pair> = (0..FRAMES)
            .map(|i| {
                let shot = i / 16;
                let noise = white(&mut rng_for(7, i, 0), W * H, SHOT_STD[shot]);
                (
                    frame(&quantize(&clean_y, &noise), &clean_c, &clean_c),
                    frame(&clean_y, &clean_c, &clean_c),
                )
            })
            .collect();
        let est = estimate(&pairs, &SHOT_START);
        // Empirical std per shot (the segment-level fit is printed without it).
        for (k, shot_pairs) in pairs.chunks(16).enumerate() {
            let shot_emp = empirical_by_level(shot_pairs, &clean_y);
            eprintln!(
                "[K7] shot{k} (std {}) empirical std @128 = {:.4}",
                SHOT_STD[k], shot_emp[128].2
            );
        }
        summarize("K7", &est, None, &STRIP_LEVELS);

        let mut checks = Checks::default();
        checks.exact("K7 segment count", est.segments.len(), 3);
        for (k, &start_frame) in SHOT_START.iter().enumerate() {
            let Some(seg) = est.segments.get(k) else {
                checks.fail(format!("K7 seg{k}: missing"));
                continue;
            };
            let want_start = start_frame as u64 * 10_000_000 / 24;
            checks.exact(
                &format!("K7 seg{k} start_time (frame {start_frame})"),
                seg.start_time,
                want_start,
            );
            checks.rel(
                &format!("K7 seg{k} mid-level std @128 (shot std {})", SHOT_STD[k]),
                grain_std_y(seg, 128.0),
                expected_std(SHOT_STD[k]),
                0.10,
            );
        }
        checks.finish("K7");
    }

    /// K8: K1 plus a bright 220 patch; the denoised frame has a +20 ring just outside it.
    #[test]
    fn k8_bright_patch_with_denoiser_ring() {
        let clean_y = clean_luma(true);
        let clean_c = flat(CW * CH, 128);
        let den_y = grid(W, H, |x, y| {
            let bump = if in_patch_ring(x, y) { 20.0 } else { 0.0 };
            f64::from(clean_y[y * W + x]) + bump
        });
        let pairs: Vec<Pair> = (0..FRAMES)
            .map(|i| {
                let noise = white(&mut rng_for(8, i, 0), W * H, 3.0);
                (
                    frame(&quantize(&clean_y, &noise), &clean_c, &clean_c),
                    frame(&den_y, &clean_c, &clean_c),
                )
            })
            .collect();
        let est = estimate(&pairs, &[]);
        let emp = empirical_by_level(&pairs, &clean_y);
        summarize("K8", &est, Some(&emp), &PATCH_PRINT_LEVELS);

        let mut checks = Checks::default();
        let seg = first_segment(&est);
        let expected = expected_std(3.0);
        checks.rel(
            "K8 luma std @220 (patch)",
            grain_std_y(seg, 220.0),
            expected,
            0.15,
        );
        checks.rel("K8 luma std @200", grain_std_y(seg, 200.0), expected, 0.15);
        checks.finish("K8");
    }

    fn white_pairs(case: u64, stds: &[f64]) -> Vec<Pair> {
        let clean_y = clean_luma(false);
        let clean_c = flat(CW * CH, 128);
        stds.iter()
            .enumerate()
            .map(|(i, &std)| {
                let noise = white(&mut rng_for(case, i, 0), W * H, std);
                (
                    frame(&quantize(&clean_y, &noise), &clean_c, &clean_c),
                    frame(&clean_y, &clean_c, &clean_c),
                )
            })
            .collect()
    }

    #[test]
    fn stationary_noise_stays_one_segment_without_boundaries() {
        let est = estimate(&white_pairs(9, &[3.0; 240]), &[]);
        assert_eq!(est.segments.len(), 1, "240 frames of the same noise");
    }

    #[test]
    fn a_noise_change_splits_once_without_boundaries() {
        let stds: Vec<f64> = (0..72).map(|i| if i < 36 { 1.0 } else { 6.0 }).collect();
        let est = estimate(&white_pairs(10, &stds), &[]);
        let starts: Vec<u64> = est.segments.iter().map(|s| s.start_time).collect();
        assert_eq!(
            starts,
            vec![0, 36 * 10_000_000 / 24],
            "split where the noise jumps"
        );
    }

    #[test]
    fn textured_frames_give_no_estimate() {
        // Pixel-level texture has no flat block, so nothing may enter the noise model.
        let clean_y = grid(W, H, |x, y| {
            if (x / 2 + y / 2) % 2 == 0 {
                40.0
            } else {
                200.0
            }
        });
        let clean_c = flat(CW * CH, 128);
        let pairs: Vec<Pair> = (0..8)
            .map(|i| {
                let noise = white(&mut rng_for(11, i, 0), W * H, 3.0);
                (
                    frame(&quantize(&clean_y, &noise), &clean_c, &clean_c),
                    frame(&clean_y, &clean_c, &clean_c),
                )
            })
            .collect();
        let est = estimate(&pairs, &[]);
        assert_eq!(est.frames_without_estimate, 8);
        assert_eq!(est.segments.len(), 1);
    }

    #[test]
    fn colour_grain_on_one_plane_drops_both() {
        // AV1 4:2:0 forbids scaling points on only one chroma plane.
        let clean_y = clean_luma(false);
        let clean_c = flat(CW * CH, 128);
        let pairs: Vec<Pair> = (0..8)
            .map(|i| {
                let noise = white(&mut rng_for(14, i, 0), W * H, 3.0);
                let cb_noise = white(&mut rng_for(14, i, 1), CW * CH, 2.0);
                (
                    frame(
                        &quantize(&clean_y, &noise),
                        &quantize(&clean_c, &cb_noise),
                        &clean_c,
                    ),
                    frame(&clean_y, &clean_c, &clean_c),
                )
            })
            .collect();
        let est = estimate(&pairs, &[]);
        let seg = first_segment(&est);
        assert_eq!(
            seg.scaling_points_cb.is_empty(),
            seg.scaling_points_cr.is_empty(),
            "cb {:?} cr {:?}",
            seg.scaling_points_cb,
            seg.scaling_points_cr
        );
    }
}
