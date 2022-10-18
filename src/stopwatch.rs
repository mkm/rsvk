use std::time::{Instant, Duration};

pub struct Stopwatch<T> {
    current: Instant,
    periods: Vec<(T, Duration)>,
}

impl<T> Stopwatch<T> {
    pub fn new() -> Self {
        Stopwatch {
            current: Instant::now(),
            periods: Vec::new(),
        }
    }

    pub fn tick(&mut self, info: T) {
        let now = Instant::now();
        self.periods.push((info, now - self.current));
        self.current = now;
    }

    pub fn periods(self) -> Vec<(T, Duration)> {
        self.periods
    }
}
