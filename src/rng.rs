use std::time::SystemTime;

/// Basic pseudorandom number generator.
///
/// Implements LCG (https://en.wikipedia.org/wiki/Linear_congruential_generator)
pub struct RNG {
    /// Last value
    last: i64,

    /// Multiplier
    a: i64,

    /// Increment
    c: i64,

    /// Modulus
    m: i64,
}

impl RNG {
    pub fn new(_seed: i64) -> RNG {
        RNG {
            last: _seed,
            a: 1664525,
            c: 1013904223,
            m: 4294967296,
        }
    }

    pub fn new_time_seed() -> RNG {
        match SystemTime::now().duration_since(SystemTime::UNIX_EPOCH) {
            Ok(n) => RNG::new(n.as_secs().try_into().expect("failed to convert time integers")),
            Err(_) => panic!("failed to get system time"),
        }
    }

    /// Return a random i64.
    pub fn rand_i64(&mut self) -> i64 {
        self.last = (self.a * self.last + self.c) % self.m;
        self.last
    }

    /// Return a random double predcision number.
    pub fn rand_f64(&mut self) -> f64 {
        let exp = self.rand_i64() % 16;
        let den = 2_i64.pow(exp.try_into().expect("failed to unwrap exponent value"));
        let num = self.rand_i64() % den;
        num as f64 / den as f64
    }
}
