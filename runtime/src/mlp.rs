//! Feed-forward policy interpreter (WP4-09, ADR-0013).
//!
//! Runs the MLP file written by `bipedlab.mlp_export` (format documented there).
//! std only: no ONNX runtime library on the robot. Every size in the file is
//! bounded before anything is allocated, and the op chain must keep the tensor
//! sizes consistent, so a corrupt file is an error, never a panic.

use std::io;

pub const MAGIC: &[u8; 4] = b"BPML";
pub const VERSION: u16 = 1;
/// Upper bounds so a corrupt file cannot make us allocate much.
pub const MAX_DIM: usize = 4096;
pub const MAX_OPS: usize = 64;

#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    /// `y = x @ w + b`, `w` row-major `rows x cols`.
    Dense {
        rows: usize,
        cols: usize,
        w: Vec<f32>,
        b: Vec<f32>,
    },
    Scale(Vec<f32>),
    Shift(Vec<f32>),
    Relu,
    Elu(f32),
    Tanh,
    Clip(f32, f32),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Mlp {
    pub in_dim: usize,
    pub out_dim: usize,
    pub ops: Vec<Op>,
}

fn bad(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> io::Result<&[u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.buf.len());
        let end = end.ok_or_else(|| bad("truncated MLP file"))?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> io::Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> io::Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().expect("2 bytes")))
    }
    fn u32(&mut self) -> io::Result<usize> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("4 bytes")) as usize)
    }
    fn dim(&mut self) -> io::Result<usize> {
        let d = self.u32()?;
        if d == 0 || d > MAX_DIM {
            return Err(bad(format!("dimension {d} out of range")));
        }
        Ok(d)
    }
    fn f32s(&mut self, n: usize) -> io::Result<Vec<f32>> {
        let raw = self.take(n.checked_mul(4).ok_or_else(|| bad("size overflow"))?)?;
        let v: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
            .collect();
        if v.iter().all(|x| x.is_finite()) {
            Ok(v)
        } else {
            Err(bad("non-finite parameter"))
        }
    }
}

impl Mlp {
    pub fn from_bytes(buf: &[u8]) -> io::Result<Self> {
        let mut r = Reader { buf, pos: 0 };
        if r.take(4)? != MAGIC {
            return Err(bad("bad magic"));
        }
        if r.u16()? != VERSION {
            return Err(bad("unsupported MLP file version"));
        }
        let in_dim = r.dim()?;
        let n_ops = r.u32()?;
        if n_ops > MAX_OPS {
            return Err(bad("too many ops"));
        }
        let mut dim = in_dim;
        let mut ops = Vec::with_capacity(n_ops);
        for _ in 0..n_ops {
            let op = match r.u8()? {
                1 => {
                    let rows = r.dim()?;
                    let cols = r.dim()?;
                    if rows != dim {
                        return Err(bad(format!("dense layer expects {rows} inputs, tensor has {dim}")));
                    }
                    let w = r.f32s(rows * cols)?;
                    let b = r.f32s(cols)?;
                    dim = cols;
                    Op::Dense { rows, cols, w, b }
                }
                k @ (2 | 3) => {
                    let n = r.dim()?;
                    if n != dim {
                        return Err(bad("elementwise constant size mismatch"));
                    }
                    let v = r.f32s(n)?;
                    if k == 2 {
                        Op::Scale(v)
                    } else {
                        Op::Shift(v)
                    }
                }
                4 => Op::Relu,
                5 => Op::Elu(r.f32s(1)?[0]),
                6 => Op::Tanh,
                7 => {
                    // Bounds may be +-inf in the file, so read them unchecked.
                    let raw = r.take(8)?;
                    let lo = f32::from_le_bytes(raw[..4].try_into().expect("4 bytes"));
                    let hi = f32::from_le_bytes(raw[4..].try_into().expect("4 bytes"));
                    if lo.is_nan() || hi.is_nan() || lo > hi {
                        return Err(bad("invalid clip bounds"));
                    }
                    Op::Clip(lo, hi)
                }
                k => return Err(bad(format!("unknown op {k}"))),
            };
            ops.push(op);
        }
        if r.pos != buf.len() {
            return Err(bad("trailing bytes after the last op"));
        }
        Ok(Mlp {
            in_dim,
            out_dim: dim,
            ops,
        })
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.in_dim, "input size");
        let mut y = x.to_vec();
        for op in &self.ops {
            match op {
                Op::Dense { rows, cols, w, b } => {
                    let mut out = b.clone();
                    for (i, &xi) in y.iter().enumerate().take(*rows) {
                        let row = &w[i * cols..(i + 1) * cols];
                        for (o, &wij) in out.iter_mut().zip(row) {
                            *o += xi * wij;
                        }
                    }
                    y = out;
                }
                Op::Scale(v) => y.iter_mut().zip(v).for_each(|(a, s)| *a *= s),
                Op::Shift(v) => y.iter_mut().zip(v).for_each(|(a, s)| *a += s),
                Op::Relu => y.iter_mut().for_each(|a| *a = a.max(0.0)),
                Op::Elu(alpha) => y.iter_mut().for_each(|a| {
                    if *a <= 0.0 {
                        *a = alpha * (a.exp() - 1.0)
                    }
                }),
                Op::Tanh => y.iter_mut().for_each(|a| *a = a.tanh()),
                Op::Clip(lo, hi) => y.iter_mut().for_each(|a| *a = a.clamp(*lo, *hi)),
            }
        }
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(ops: &[u8], in_dim: u32, n_ops: u32) -> Vec<u8> {
        let mut v = MAGIC.to_vec();
        v.extend_from_slice(&VERSION.to_le_bytes());
        v.extend_from_slice(&in_dim.to_le_bytes());
        v.extend_from_slice(&n_ops.to_le_bytes());
        v.extend_from_slice(ops);
        v
    }

    fn f32s(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn two_layer() -> Vec<u8> {
        // 2 -> 2 dense, ELU(1), shift, 2 -> 1 dense, clip.
        let mut ops = vec![1u8];
        ops.extend_from_slice(&2u32.to_le_bytes());
        ops.extend_from_slice(&2u32.to_le_bytes());
        ops.extend(f32s(&[1.0, -1.0, 2.0, 0.5, 0.1, -0.1]));
        ops.push(5);
        ops.extend(f32s(&[1.0]));
        ops.push(3);
        ops.extend_from_slice(&2u32.to_le_bytes());
        ops.extend(f32s(&[0.5, 0.5]));
        ops.push(1);
        ops.extend_from_slice(&2u32.to_le_bytes());
        ops.extend_from_slice(&1u32.to_le_bytes());
        ops.extend(f32s(&[1.0, 1.0, 0.0]));
        ops.push(7);
        ops.extend(f32s(&[-1.0, 1.0]));
        file(&ops, 2, 5)
    }

    #[test]
    fn forward_matches_hand_computation() {
        let m = Mlp::from_bytes(&two_layer()).unwrap();
        assert_eq!((m.in_dim, m.out_dim), (2, 1));
        // x = [1, 0]: dense -> [1.1, -1.1]; elu -> [1.1, e^-1.1 - 1]; +0.5; sum; clip.
        let h = [1.1f32 + 0.5, ((-1.1f32).exp() - 1.0) + 0.5];
        let want = (h[0] + h[1]).clamp(-1.0, 1.0);
        assert!((m.forward(&[1.0, 0.0])[0] - want).abs() < 1e-6);
        // x = [-0.5, -0.5]: dense -> [-1.4, 0.15] ...
        let d = [-0.5f32 * 1.0 - 0.5 * 2.0 + 0.1, -0.5f32 * -1.0 - 0.5 * 0.5 - 0.1];
        let e = [(d[0].exp() - 1.0) + 0.5, d[1] + 0.5];
        assert!((m.forward(&[-0.5, -0.5])[0] - (e[0] + e[1])).abs() < 1e-6);
    }

    #[test]
    fn rejects_corrupt_files() {
        let good = two_layer();
        assert!(Mlp::from_bytes(&good[..good.len() - 1]).is_err(), "truncated");
        let mut extra = good.clone();
        extra.push(0);
        assert!(Mlp::from_bytes(&extra).is_err(), "trailing");
        let mut magic = good.clone();
        magic[0] = b'X';
        assert!(Mlp::from_bytes(&magic).is_err());
        let mut huge = good.clone();
        huge[6..10].copy_from_slice(&(MAX_DIM as u32 + 1).to_le_bytes());
        assert!(Mlp::from_bytes(&huge).is_err(), "in_dim bound");
        let mut mismatch = good.clone();
        mismatch[6..10].copy_from_slice(&3u32.to_le_bytes());
        assert!(Mlp::from_bytes(&mismatch).is_err(), "dense rows != in_dim");
        let mut nan = good.clone();
        nan[23..27].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(Mlp::from_bytes(&nan).is_err(), "NaN weight");
    }
}
