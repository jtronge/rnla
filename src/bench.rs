/// Benchmark abstraction.
use std::time::Instant;

pub struct BenchOptions {
    /// Number of warmup iterations.
    warmup: usize,

    /// Number of trials
    trial_count: usize,
}

/// Benchmark the function and return the average time.
pub fn bench<S, C, A>(opts: BenchOptions, startup: S, critical_code: C) -> f64
where
    S: Fn() -> A,
    C: Fn(A),
{
    let mut total_time = 0.0;
    let mut total_count = 0;
    for i in 0..opts.warmup + opts.trial_count {
        let args = startup();

        let timer = Instant::now();
        critical_code(args);
        if i >= opts.warmup {
            total_time += timer.elapsed().as_secs_f64();
            total_count += 1;
        }
    }
    assert_eq!(total_count, opts.trial_count);

    total_time / (opts.trial_count as f64)
}
