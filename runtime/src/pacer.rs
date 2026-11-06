//! Wall-clock pacing of the control loop on an absolute schedule (WP4-16).
//!
//! Deadlines are `t0 + k * period`. Sleeping until the next absolute deadline
//! means the time spent outside the runtime (waiting for the state, the sim
//! step, the network) is part of the period instead of being added to it, so
//! the loop holds its rate. A period that ends after its deadline is a miss;
//! after a miss the schedule catches up if it is less than one period late and
//! restarts from now otherwise, so one long stall does not cause a burst.

use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct Pacer {
    period: Duration,
    next: Option<Instant>,
    pub misses: u64,
}

impl Pacer {
    pub fn new(period: Duration) -> Self {
        Self {
            period,
            next: None,
            misses: 0,
        }
    }

    /// Call once per period with the period's start time (first call) and the
    /// current time; returns how long to sleep, or `None` on a miss.
    pub fn wait_time(&mut self, start: Instant, now: Instant) -> Option<Duration> {
        let deadline = self.next.unwrap_or(start) + self.period;
        if now <= deadline {
            self.next = Some(deadline);
            Some(deadline - now)
        } else {
            self.misses += 1;
            let late = now - deadline;
            self.next = Some(if late < self.period { deadline } else { now });
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: Duration = Duration::from_millis(20);

    #[test]
    fn time_outside_the_loop_is_part_of_the_period() {
        let t0 = Instant::now();
        let mut p = Pacer::new(P);
        // Compute took 1 ms: sleep 19 ms to the first deadline.
        assert_eq!(
            p.wait_time(t0, t0 + Duration::from_millis(1)),
            Some(Duration::from_millis(19))
        );
        // Next period: 5 ms waiting for the sim + 1 ms compute after the deadline
        // at t0+20 ms -> sleep 14 ms to t0+40 ms (relative pacing would sleep 19).
        let s = t0 + Duration::from_millis(25);
        assert_eq!(
            p.wait_time(s, t0 + Duration::from_millis(26)),
            Some(Duration::from_millis(14))
        );
        assert_eq!(p.misses, 0);
    }

    #[test]
    fn small_overrun_is_a_miss_and_the_schedule_catches_up() {
        let t0 = Instant::now();
        let mut p = Pacer::new(P);
        assert_eq!(p.wait_time(t0, t0 + Duration::from_millis(25)), None);
        assert_eq!(p.misses, 1);
        // Deadline stays t0+40 ms.
        let now = t0 + Duration::from_millis(30);
        assert_eq!(p.wait_time(now, now), Some(Duration::from_millis(10)));
    }

    #[test]
    fn long_stall_restarts_the_schedule_without_a_burst() {
        let t0 = Instant::now();
        let mut p = Pacer::new(P);
        let late = t0 + Duration::from_millis(100);
        assert_eq!(p.wait_time(t0, late), None);
        assert_eq!(
            p.wait_time(late, late + Duration::from_millis(1)),
            Some(Duration::from_millis(19))
        );
        assert_eq!(p.misses, 1);
    }

    #[test]
    fn holds_fifty_hertz_when_the_sim_takes_five_ms_per_step() {
        // Model of the MAC-09 failure: 5 ms outside the runtime + 0.1 ms compute per
        // period. Relative sleeping gave ~25 ms periods; the absolute schedule keeps 20.
        let t0 = Instant::now();
        let mut p = Pacer::new(P);
        let mut now = t0;
        for _ in 0..250 {
            now += Duration::from_micros(5100);
            let start = now - Duration::from_micros(100);
            now += p.wait_time(start, now).unwrap_or_default();
        }
        // First deadline is anchored at the first start (t0 + 5 ms) + 20 ms.
        let total = now - t0;
        assert!(total <= Duration::from_millis(250 * 20 + 5), "{total:?}");
        assert_eq!(p.misses, 0);
    }
}
