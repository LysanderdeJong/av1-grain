use anyhow::{Result, ensure};
use num_rational::Rational64;
use rayon::join;
use v_frame::{frame::Frame, pixel::Pixel};

use self::solver::{FlatBlockFinder, NoiseModel};
use crate::{
    GrainTableSegment,
    profile::{self, MetricId},
    util::frame_into_u8,
};

mod solver;

const BLOCK_SIZE: usize = 32;
const BLOCK_SIZE_SQUARED: usize = BLOCK_SIZE * BLOCK_SIZE;
/// Without boundaries, a segment holds at least this many frames before a noise change may
/// split it, so single-frame estimates can't flip the table every frame.
const MIN_SEGMENT_FRAMES: usize = 24;

pub struct DiffGenerator {
    fps: Rational64,
    source_bit_depth: usize,
    denoised_bit_depth: usize,
    frame_count: usize,
    prev_timestamp: u64,
    segment_frames: usize,
    flat_block_finder: FlatBlockFinder,
    noise_model: NoiseModel,
    grain_table: Vec<GrainTableSegment>,
    boundaries: Vec<usize>,
    frames_without_estimate: usize,
}

impl DiffGenerator {
    #[must_use]
    #[inline]
    pub fn new(fps: Rational64, source_bit_depth: usize, denoised_bit_depth: usize) -> Self {
        Self {
            frame_count: 0,
            fps,
            flat_block_finder: FlatBlockFinder::new(),
            noise_model: NoiseModel::new(),
            grain_table: Vec::new(),
            prev_timestamp: 0,
            segment_frames: 0,
            source_bit_depth,
            denoised_bit_depth,
            boundaries: Vec::new(),
            frames_without_estimate: 0,
        }
    }

    /// Start a new grain segment at each of these frame indices (for example shot starts)
    /// and nowhere else. Without boundaries, segments split where the noise changes.
    #[inline]
    pub fn set_boundaries(&mut self, mut frames: Vec<usize>) {
        frames.sort_unstable();
        frames.dedup();
        self.boundaries = frames;
    }

    /// Frames that added nothing to the noise model: too few flat blocks, or a failed solve.
    #[must_use]
    #[inline]
    pub const fn frames_without_estimate(&self) -> usize {
        self.frames_without_estimate
    }

    /// Processes the next frame and adds the results to the state of this
    /// `DiffGenerator`.
    ///
    /// # Errors
    /// - If the frames do not have the same resolution
    /// - If the frames do not have the same chroma subsampling
    #[inline]
    pub fn diff_frame<T: Pixel + Send + Sync, U: Pixel + Send + Sync>(
        &mut self,
        source: &Frame<T>,
        denoised: &Frame<U>,
    ) -> Result<()> {
        let (source, denoised) = profile::time(MetricId::FrameConvert, || {
            join(
                || frame_into_u8(source, self.source_bit_depth),
                || frame_into_u8(denoised, self.denoised_bit_depth),
            )
        });

        self.diff_frame_internal(&source, &denoised)
    }

    /// Finalize the state of this `DiffGenerator` and return the resulting
    /// grain table segments.
    #[must_use]
    #[inline]
    pub fn finish(mut self) -> Vec<GrainTableSegment> {
        let grain_table = profile::time(MetricId::Finish, || {
            log::debug!("Updating final parameters");
            self.push_segment(i64::MAX as u64);
            self.grain_table
        });
        profile::print_report();
        grain_table
    }

    /// Close the current segment at `end`. A segment whose frames gave no estimate repeats
    /// the previous segment's grain instead of writing none.
    fn push_segment(&mut self, end: u64) {
        let segment = match self.grain_table.last() {
            Some(prev) if !self.noise_model.has_estimate() => GrainTableSegment {
                random_seed: 0,
                start_time: self.prev_timestamp,
                end_time: end,
                ..prev.clone()
            },
            _ => profile::time(MetricId::GrainParameters, || {
                self.noise_model
                    .get_grain_parameters(self.prev_timestamp, end)
            }),
        };
        self.grain_table.push(segment);
    }

    fn diff_frame_internal(&mut self, source: &Frame<u8>, denoised: &Frame<u8>) -> Result<()> {
        profile::time(MetricId::VerifyDimensions, || {
            verify_dimensions_match(source, denoised)
        })?;

        let (flat_blocks, num_flat_blocks) = profile::time(MetricId::FlatBlockFinder, || {
            self.flat_block_finder
                .run(&source.y_plane, &denoised.y_plane)
        });
        log::debug!("Num flat blocks: {num_flat_blocks}");

        let cur_timestamp = self.frame_count as u64 * 10_000_000u64 * *self.fps.denom() as u64
            / *self.fps.numer() as u64;
        let has_boundaries = !self.boundaries.is_empty();
        if has_boundaries
            && self.segment_frames > 0
            && self.boundaries.binary_search(&self.frame_count).is_ok()
        {
            self.push_segment(cur_timestamp);
            self.noise_model.reset();
            self.prev_timestamp = cur_timestamp;
            self.segment_frames = 0;
        }

        log::debug!("Updating noise model");
        let allow_split = !has_boundaries && self.segment_frames >= MIN_SEGMENT_FRAMES;
        let status = profile::time(MetricId::NoiseModelUpdate, || {
            self.noise_model
                .update(source, denoised, &flat_blocks, allow_split)
        });

        match status {
            NoiseStatus::DifferentType => {
                log::debug!(
                    "Updating parameters for times {} to {}",
                    self.prev_timestamp,
                    cur_timestamp
                );
                self.push_segment(cur_timestamp);
                profile::time(MetricId::SaveLatest, || self.noise_model.save_latest());
                self.prev_timestamp = cur_timestamp;
                self.segment_frames = 0;
            }
            NoiseStatus::Error(err) => {
                log::debug!("No noise estimate for frame {}: {err:#}", self.frame_count);
                self.frames_without_estimate += 1;
            }
            NoiseStatus::Ok => {}
        }
        self.segment_frames += 1;
        log::debug!("Noise model updated for frame {}", self.frame_count);
        self.frame_count += 1;

        Ok(())
    }
}

#[derive(Debug)]
enum NoiseStatus {
    Ok,
    DifferentType,
    #[allow(dead_code)]
    Error(anyhow::Error),
}

impl PartialEq for NoiseStatus {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (&Self::Error(_), &Self::Error(_)) => true,
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

fn verify_dimensions_match(source: &Frame<u8>, denoised: &Frame<u8>) -> Result<()> {
    let res_1 = (source.y_plane.width(), source.y_plane.height());
    let res_2 = (denoised.y_plane.width(), denoised.y_plane.height());
    ensure!(
        res_1 == res_2,
        "Luma resolutions were not equal, {}x{} != {}x{}",
        res_1.0,
        res_1.1,
        res_2.0,
        res_2.1
    );

    let res_1 = (
        source.u_plane.as_ref().map(|p| p.width()),
        source.u_plane.as_ref().map(|p| p.height()),
    );
    let res_2 = (
        denoised.u_plane.as_ref().map(|p| p.width()),
        denoised.u_plane.as_ref().map(|p| p.height()),
    );
    ensure!(
        res_1 == res_2,
        "Chroma resolutions were not equal, {}x{} != {}x{}",
        res_1.0.unwrap_or(0),
        res_1.1.unwrap_or(0),
        res_2.0.unwrap_or(0),
        res_2.1.unwrap_or(0)
    );

    Ok(())
}
