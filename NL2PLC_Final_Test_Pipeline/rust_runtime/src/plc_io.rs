use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlcIoError {
    message: String,
}

impl PlcIoError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for PlcIoError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for PlcIoError {}

pub type PlcResult<T> = Result<T, PlcIoError>;

pub trait PlcIo {
    fn read_discrete_input(&mut self, address: &str) -> PlcResult<bool>;
    fn write_coil(&mut self, address: &str, value: bool) -> PlcResult<()>;
    fn read_input_register(&mut self, address: &str) -> PlcResult<i16>;
    fn write_holding_register(&mut self, address: &str, value: i16)
        -> PlcResult<()>;
}

/// Compilation-only runtime. Replace it with the competition/runtime adapter.
pub struct NullIo;

impl PlcIo for NullIo {
    fn read_discrete_input(&mut self, _address: &str) -> PlcResult<bool> {
        Ok(false)
    }

    fn write_coil(&mut self, _address: &str, _value: bool) -> PlcResult<()> {
        Ok(())
    }

    fn read_input_register(&mut self, _address: &str) -> PlcResult<i16> {
        Ok(0)
    }

    fn write_holding_register(
        &mut self,
        _address: &str,
        _value: i16,
    ) -> PlcResult<()> {
        Ok(())
    }
}

