#[derive(Clone, Copy)]
pub(crate) enum MetricId {
    FrameConvert,
    VerifyDimensions,
    FlatBlockFinder,
    NoiseModelUpdate,
    UpdateLatestY,
    UpdateLatestCb,
    UpdateLatestCr,
    AddBlockObservationsY,
    AddBlockObservationsCb,
    AddBlockObservationsCr,
    AddNoiseStdY,
    AddNoiseStdCb,
    AddNoiseStdCr,
    CombineState,
    GrainParameters,
    SaveLatest,
    Finish,
}

#[cfg(feature = "profile-diff")]
mod enabled {
    use std::{
        sync::atomic::{AtomicU64, Ordering},
        time::Instant,
    };

    use super::MetricId;

    struct Metric {
        nanos: AtomicU64,
        count: AtomicU64,
    }

    impl Metric {
        const fn new() -> Self {
            Self {
                nanos: AtomicU64::new(0),
                count: AtomicU64::new(0),
            }
        }
    }

    const METRIC_COUNT: usize = 17;
    static METRICS: [Metric; METRIC_COUNT] = [
        Metric::new(),
        Metric::new(),
        Metric::new(),
        Metric::new(),
        Metric::new(),
        Metric::new(),
        Metric::new(),
        Metric::new(),
        Metric::new(),
        Metric::new(),
        Metric::new(),
        Metric::new(),
        Metric::new(),
        Metric::new(),
        Metric::new(),
        Metric::new(),
        Metric::new(),
    ];

    const LABELS: [&str; METRIC_COUNT] = [
        "frame_convert",
        "verify_dimensions",
        "flat_block_finder",
        "noise_model_update",
        "update_latest_y",
        "update_latest_cb",
        "update_latest_cr",
        "add_block_observations_y",
        "add_block_observations_cb",
        "add_block_observations_cr",
        "add_noise_std_y",
        "add_noise_std_cb",
        "add_noise_std_cr",
        "combine_state",
        "grain_parameters",
        "save_latest",
        "finish",
    ];

    fn metric_index(id: MetricId) -> usize {
        id as usize
    }

    pub(crate) fn time<T>(id: MetricId, f: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let result = f();
        let elapsed = start.elapsed().as_nanos();
        let elapsed = elapsed.min(u128::from(u64::MAX)) as u64;
        let metric = &METRICS[metric_index(id)];
        metric.nanos.fetch_add(elapsed, Ordering::Relaxed);
        metric.count.fetch_add(1, Ordering::Relaxed);
        result
    }

    pub(crate) fn print_report() {
        eprintln!("av1-grain diff profile:");
        eprintln!(
            "{:<30} {:>10} {:>12} {:>12}",
            "metric", "count", "total_ms", "avg_ms"
        );
        for (label, metric) in LABELS.iter().zip(METRICS.iter()) {
            let count = metric.count.load(Ordering::Relaxed);
            if count == 0 {
                continue;
            }
            let nanos = metric.nanos.load(Ordering::Relaxed);
            let total_ms = nanos as f64 / 1_000_000.0;
            let avg_ms = total_ms / count as f64;
            eprintln!("{label:<30} {count:>10} {total_ms:>12.3} {avg_ms:>12.3}");
        }
    }
}

#[cfg(feature = "profile-diff")]
pub(crate) use enabled::{print_report, time};

#[cfg(not(feature = "profile-diff"))]
#[inline(always)]
pub(crate) fn time<T>(_: MetricId, f: impl FnOnce() -> T) -> T {
    f()
}

#[cfg(not(feature = "profile-diff"))]
#[inline(always)]
pub(crate) fn print_report() {}
