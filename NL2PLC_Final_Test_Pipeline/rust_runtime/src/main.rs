use std::time::Duration;

use nl2plc_runtime::plc_io::NullIo;
use nl2plc_runtime::program::Program;

fn main() {
    let mut io = NullIo;
    let mut program = Program::default();
    if let Err(error) = program.scan(&mut io, Duration::from_millis(20)) {
        eprintln!("PLC scan failed: {error}");
    }
}

