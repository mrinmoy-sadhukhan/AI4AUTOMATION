use std::time::Duration;

use crate::plc_io::{PlcIo, PlcResult};

#[derive(Default)]
pub struct Program;

impl Program {
    pub fn scan<I: PlcIo>(
        &mut self,
        _io: &mut I,
        _dt: Duration,
    ) -> PlcResult<()> {
        Ok(())
    }
}

