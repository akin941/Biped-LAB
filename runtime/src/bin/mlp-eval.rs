//! `mlp-eval MODEL INPUT OUTPUT`: run an MLP file on a batch for parity tests.
//!
//! INPUT is raw little-endian f32 rows of the model's input size; OUTPUT gets
//! raw f32 rows of its output size.

use std::process::ExitCode;

use biped_runtime::mlp::Mlp;

fn run(args: &[String]) -> Result<(), String> {
    let [model, input, output] = args else {
        return Err("usage: mlp-eval MODEL INPUT OUTPUT".into());
    };
    let m = Mlp::from_bytes(&std::fs::read(model).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let raw = std::fs::read(input).map_err(|e| e.to_string())?;
    if raw.len() % (4 * m.in_dim) != 0 {
        return Err("input size is not a multiple of the model input".into());
    }
    let xs: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
        .collect();
    let mut out = Vec::with_capacity(xs.len() / m.in_dim * m.out_dim * 4);
    for row in xs.chunks_exact(m.in_dim) {
        for v in m.forward(row) {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(output, out).map_err(|e| e.to_string())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("mlp-eval: {e}");
            ExitCode::from(2)
        }
    }
}
