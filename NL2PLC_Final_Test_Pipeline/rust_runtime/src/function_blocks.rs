use std::time::Duration;

fn saturating_add_duration(left: Duration, right: Duration) -> Duration {
    left.checked_add(right).unwrap_or(Duration::MAX)
}

#[derive(Debug, Default, Clone)]
pub struct RisingEdge {
    previous: bool,
}

impl RisingEdge {
    pub fn update(&mut self, input: bool) -> bool {
        let edge = input && !self.previous;
        self.previous = input;
        edge
    }
}

#[derive(Debug, Default, Clone)]
pub struct FallingEdge {
    previous: bool,
}

impl FallingEdge {
    pub fn update(&mut self, input: bool) -> bool {
        let edge = !input && self.previous;
        self.previous = input;
        edge
    }
}

#[derive(Debug, Default, Clone)]
pub struct OnDelay {
    elapsed: Duration,
    output: bool,
}

impl OnDelay {
    pub fn update(&mut self, input: bool, preset: Duration, dt: Duration) -> bool {
        if !input {
            self.elapsed = Duration::ZERO;
            self.output = false;
        } else if !self.output {
            self.elapsed = saturating_add_duration(self.elapsed, dt);
            self.output = self.elapsed >= preset;
        }
        self.output
    }

    pub fn elapsed(&self) -> Duration {
        self.elapsed
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[derive(Debug, Default, Clone)]
pub struct OffDelay {
    elapsed: Duration,
    output: bool,
}

impl OffDelay {
    pub fn update(&mut self, input: bool, preset: Duration, dt: Duration) -> bool {
        if input {
            self.elapsed = Duration::ZERO;
            self.output = true;
        } else if self.output {
            self.elapsed = saturating_add_duration(self.elapsed, dt);
            if self.elapsed >= preset {
                self.output = false;
            }
        }
        self.output
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[derive(Debug, Default, Clone)]
pub struct PulseTimer {
    previous: bool,
    elapsed: Duration,
    output: bool,
}

impl PulseTimer {
    pub fn update(&mut self, input: bool, preset: Duration, dt: Duration) -> bool {
        let rising = input && !self.previous;
        self.previous = input;
        if rising && !self.output {
            self.output = true;
            self.elapsed = Duration::ZERO;
        }
        if self.output {
            self.elapsed = saturating_add_duration(self.elapsed, dt);
            if self.elapsed >= preset {
                self.output = false;
            }
        }
        self.output
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[derive(Debug, Default, Clone)]
pub struct CounterUp {
    previous: bool,
    current: u32,
}

impl CounterUp {
    pub fn update(&mut self, count: bool, reset: bool, preset: u32) -> bool {
        if reset {
            self.current = 0;
        } else if count && !self.previous {
            self.current = self.current.saturating_add(1);
        }
        self.previous = count;
        self.current >= preset
    }

    pub fn current(&self) -> u32 {
        self.current
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[derive(Debug, Default, Clone)]
pub struct CounterDown {
    previous: bool,
    current: u32,
}

impl CounterDown {
    pub fn update(&mut self, count: bool, load: bool, preset: u32) -> bool {
        if load {
            self.current = preset;
        } else if count && !self.previous {
            self.current = self.current.saturating_sub(1);
        }
        self.previous = count;
        self.current == 0
    }

    pub fn current(&self) -> u32 {
        self.current
    }
}

#[derive(Debug, Default, Clone)]
pub struct UpDownCounter {
    previous_up: bool,
    previous_down: bool,
    current: i32,
}

impl UpDownCounter {
    pub fn update(
        &mut self,
        up: bool,
        down: bool,
        reset: bool,
        load: Option<i32>,
    ) -> i32 {
        if reset {
            self.current = 0;
        } else if let Some(value) = load {
            self.current = value;
        } else {
            if up && !self.previous_up {
                self.current = self.current.saturating_add(1);
            }
            if down && !self.previous_down {
                self.current = self.current.saturating_sub(1);
            }
        }
        self.previous_up = up;
        self.previous_down = down;
        self.current
    }

    pub fn current(&self) -> i32 {
        self.current
    }
}

