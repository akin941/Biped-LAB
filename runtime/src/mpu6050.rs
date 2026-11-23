//! MPU6050 IMU driver over I2C (WP5-07, G4.5 low-budget build).
//!
//! The owner's BOM uses an MPU6050 (GY-521) breakout. This module reads it and
//! turns it into what `hwstate::Imu` reports: the body orientation (w, x, y, z)
//! and the gyro (rad/s, body frame, x forward, y left, z up).
//!
//! - `I2c` is the bus: `LinuxI2c` talks to `/dev/i2c-N` (i2c-dev, Linux only),
//!   `SimI2c` is a register map for tests.
//! - `Mpu6050::open` checks WHO_AM_I, wakes the chip with the gyro clock, sets
//!   the 44 Hz low-pass filter, +-500 deg/s and +-4 g, then averages the gyro
//!   while the robot is still to remove its bias.
//! - `Mount` maps the chip axes to the body axes (the breakout's mounting).
//! - Orientation comes from a Mahony-style complementary filter: the gyro is
//!   integrated and the accelerometer pulls roll and pitch back towards
//!   gravity. Yaw drifts (no magnetometer); the policy only uses projected
//!   gravity and the gyro, so that does not matter for walking.
//!
//! Nothing here has run on a real chip (`env: hw` needs the owner's approval).

use crate::hwstate::Imu;
use std::collections::HashMap;
use std::time::Instant;

pub const DEFAULT_ADDR: u16 = 0x68;

/// Register addresses (MPU-6000/6050 register map, rev. 4.2).
pub mod reg {
    pub const SMPLRT_DIV: u8 = 0x19;
    pub const CONFIG: u8 = 0x1A;
    pub const GYRO_CONFIG: u8 = 0x1B;
    pub const ACCEL_CONFIG: u8 = 0x1C;
    pub const ACCEL_XOUT_H: u8 = 0x3B;
    pub const PWR_MGMT_1: u8 = 0x6B;
    pub const WHO_AM_I: u8 = 0x75;
}

/// WHO_AM_I values accepted: MPU6050 (0x68) and the register-compatible
/// MPU6500 / MPU9250 parts often sold as "MPU6050" breakouts (0x70, 0x71).
pub const WHO_AM_I_OK: [u8; 3] = [0x68, 0x70, 0x71];

/// +-4 g full scale: 8192 LSB per g.
pub const ACCEL_LSB_PER_G: f64 = 8192.0;
/// +-500 deg/s full scale: 65.5 LSB per deg/s.
pub const GYRO_LSB_PER_DPS: f64 = 65.5;

pub trait I2c {
    /// Writes `bytes` to the device at `addr`.
    fn write(&mut self, addr: u16, bytes: &[u8]) -> Result<(), String>;
    /// Writes the register address `reg`, then reads `buf.len()` bytes.
    fn write_read(&mut self, addr: u16, reg: u8, buf: &mut [u8]) -> Result<(), String>;
}

/// Register-map I2C device for tests: each address has 256 registers;
/// a read of n bytes returns registers reg..reg+n, a write sets them.
#[derive(Default)]
pub struct SimI2c {
    pub regs: HashMap<u16, [u8; 256]>,
    /// Every write, in order (address, bytes).
    pub writes: Vec<(u16, Vec<u8>)>,
}

impl SimI2c {
    /// A simulated MPU6050 at `addr` with the given WHO_AM_I.
    pub fn mpu(addr: u16, who_am_i: u8) -> Self {
        let mut s = SimI2c::default();
        let mut r = [0u8; 256];
        r[reg::WHO_AM_I as usize] = who_am_i;
        r[reg::PWR_MGMT_1 as usize] = 0x40; // sleep bit set after power-up
        s.regs.insert(addr, r);
        s
    }

    /// Sets the raw sample registers (accel xyz, temperature, gyro xyz).
    pub fn set_sample(&mut self, addr: u16, accel: [i16; 3], gyro: [i16; 3]) {
        let r = self.regs.get_mut(&addr).expect("no such device");
        let words = [accel[0], accel[1], accel[2], 0, gyro[0], gyro[1], gyro[2]];
        for (k, w) in words.iter().enumerate() {
            let b = w.to_be_bytes();
            r[reg::ACCEL_XOUT_H as usize + 2 * k] = b[0];
            r[reg::ACCEL_XOUT_H as usize + 2 * k + 1] = b[1];
        }
    }
}

impl I2c for SimI2c {
    fn write(&mut self, addr: u16, bytes: &[u8]) -> Result<(), String> {
        let r = self.regs.get_mut(&addr).ok_or(format!("no device at {addr:#x}"))?;
        if let Some((&first, rest)) = bytes.split_first() {
            for (k, &b) in rest.iter().enumerate() {
                r[(first as usize + k) & 0xff] = b;
            }
        }
        self.writes.push((addr, bytes.to_vec()));
        Ok(())
    }

    fn write_read(&mut self, addr: u16, reg: u8, buf: &mut [u8]) -> Result<(), String> {
        let r = self.regs.get(&addr).ok_or(format!("no device at {addr:#x}"))?;
        for (k, b) in buf.iter_mut().enumerate() {
            *b = r[(reg as usize + k) & 0xff];
        }
        Ok(())
    }
}

/// `/dev/i2c-N` through the Linux i2c-dev interface (I2C_SLAVE ioctl, then
/// plain read / write). Only built on Linux; elsewhere `open` returns an error.
pub struct LinuxI2c {
    #[cfg(target_os = "linux")]
    file: std::fs::File,
    #[cfg(target_os = "linux")]
    current: Option<u16>,
}

#[cfg(target_os = "linux")]
mod sys {
    extern "C" {
        pub fn ioctl(fd: i32, request: std::ffi::c_ulong, ...) -> i32;
    }
    pub const I2C_SLAVE: std::ffi::c_ulong = 0x0703;
}

impl LinuxI2c {
    #[cfg(target_os = "linux")]
    pub fn open(path: &str) -> Result<Self, String> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| format!("{path}: {e}"))?;
        Ok(LinuxI2c { file, current: None })
    }

    #[cfg(not(target_os = "linux"))]
    pub fn open(path: &str) -> Result<Self, String> {
        Err(format!("{path}: i2c-dev is Linux only"))
    }

    #[cfg(target_os = "linux")]
    fn select(&mut self, addr: u16) -> Result<(), String> {
        use std::os::fd::AsRawFd;
        if self.current != Some(addr) {
            // SAFETY: I2C_SLAVE takes the 7-bit address as an integer argument
            // on an open i2c-dev file descriptor; no memory is passed.
            let rc = unsafe { sys::ioctl(self.file.as_raw_fd(), sys::I2C_SLAVE, std::ffi::c_ulong::from(addr)) };
            if rc < 0 {
                return Err(format!("I2C_SLAVE {addr:#x}: {}", std::io::Error::last_os_error()));
            }
            self.current = Some(addr);
        }
        Ok(())
    }
}

#[cfg(target_os = "linux")]
impl I2c for LinuxI2c {
    fn write(&mut self, addr: u16, bytes: &[u8]) -> Result<(), String> {
        use std::io::Write;
        self.select(addr)?;
        self.file.write_all(bytes).map_err(|e| format!("i2c write: {e}"))
    }

    fn write_read(&mut self, addr: u16, reg: u8, buf: &mut [u8]) -> Result<(), String> {
        use std::io::{Read, Write};
        self.select(addr)?;
        self.file.write_all(&[reg]).map_err(|e| format!("i2c write: {e}"))?;
        self.file.read_exact(buf).map_err(|e| format!("i2c read: {e}"))
    }
}

#[cfg(not(target_os = "linux"))]
impl I2c for LinuxI2c {
    fn write(&mut self, _addr: u16, _bytes: &[u8]) -> Result<(), String> {
        Err("i2c-dev is Linux only".into())
    }
    fn write_read(&mut self, _addr: u16, _reg: u8, _buf: &mut [u8]) -> Result<(), String> {
        Err("i2c-dev is Linux only".into())
    }
}

/// Chip-to-body axis map: body[i] = sign[i] * chip[axis[i]].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mount {
    pub axis: [usize; 3],
    pub sign: [f64; 3],
}

impl Mount {
    pub const IDENTITY: Mount = Mount {
        axis: [0, 1, 2],
        sign: [1.0, 1.0, 1.0],
    };

    /// Parses e.g. "x,y,z" (identity) or "-y,x,z": the chip axis that gives
    /// body x, y and z, each with an optional minus sign.
    pub fn parse(s: &str) -> Result<Mount, String> {
        let parts: Vec<&str> = s.split(',').map(str::trim).collect();
        if parts.len() != 3 {
            return Err(format!("mount {s:?}: expected three axes like x,y,z"));
        }
        let mut m = Mount::IDENTITY;
        for (i, p) in parts.iter().enumerate() {
            let (sign, name) = match p.strip_prefix('-') {
                Some(n) => (-1.0, n),
                None => (1.0, *p),
            };
            m.axis[i] = match name {
                "x" => 0,
                "y" => 1,
                "z" => 2,
                _ => return Err(format!("mount {s:?}: bad axis {p:?}")),
            };
            m.sign[i] = sign;
        }
        let mut seen = m.axis;
        seen.sort_unstable();
        if seen != [0, 1, 2] {
            return Err(format!("mount {s:?}: each chip axis must be used once"));
        }
        Ok(m)
    }

    pub fn apply(&self, v: [f64; 3]) -> [f64; 3] {
        [
            self.sign[0] * v[self.axis[0]],
            self.sign[1] * v[self.axis[1]],
            self.sign[2] * v[self.axis[2]],
        ]
    }
}

/// One sample in body axes: specific force (m/s^2, +z up when level and
/// still) and angular rate (rad/s).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    pub accel: [f64; 3],
    pub gyro: [f64; 3],
}

const G: f64 = 9.80665;

/// Decodes the 14 bytes from ACCEL_XOUT_H (big-endian i16: accel xyz,
/// temperature, gyro xyz) into chip-axis SI units.
pub fn decode(raw: &[u8; 14]) -> Sample {
    let w = |k: usize| i16::from_be_bytes([raw[2 * k], raw[2 * k + 1]]) as f64;
    let a = |k: usize| w(k) / ACCEL_LSB_PER_G * G;
    let g = |k: usize| (w(k) / GYRO_LSB_PER_DPS).to_radians();
    Sample {
        accel: [a(0), a(1), a(2)],
        gyro: [g(4), g(5), g(6)],
    }
}

pub struct Mpu6050<B: I2c> {
    pub bus: B,
    pub addr: u16,
    pub mount: Mount,
    /// Gyro bias in body axes (rad/s), removed from every sample.
    pub gyro_bias: [f64; 3],
}

impl<B: I2c> Mpu6050<B> {
    /// Checks the chip, configures it and measures the gyro bias over
    /// `bias_samples` reads (the robot must be still; 0 skips it).
    pub fn open(mut bus: B, addr: u16, mount: Mount, bias_samples: usize) -> Result<Self, String> {
        let mut who = [0u8];
        bus.write_read(addr, reg::WHO_AM_I, &mut who)?;
        if !WHO_AM_I_OK.contains(&who[0]) {
            return Err(format!(
                "WHO_AM_I at {addr:#x} is {:#04x}, not an MPU6050-compatible part",
                who[0]
            ));
        }
        // Wake up with the X gyro PLL as clock, then 1 kHz / (1 + 4) = 200 Hz
        // samples, 44 Hz DLPF, +-500 deg/s, +-4 g.
        for (r, v) in [
            (reg::PWR_MGMT_1, 0x01),
            (reg::SMPLRT_DIV, 4),
            (reg::CONFIG, 0x03),
            (reg::GYRO_CONFIG, 0x08),
            (reg::ACCEL_CONFIG, 0x08),
        ] {
            bus.write(addr, &[r, v])?;
        }
        let mut imu = Mpu6050 {
            bus,
            addr,
            mount,
            gyro_bias: [0.0; 3],
        };
        if bias_samples > 0 {
            let mut sum = [0.0; 3];
            for _ in 0..bias_samples {
                let s = imu.sample()?;
                for (acc, g) in sum.iter_mut().zip(s.gyro) {
                    *acc += g;
                }
            }
            imu.gyro_bias = sum.map(|v| v / bias_samples as f64);
        }
        Ok(imu)
    }

    /// One sample in body axes with the gyro bias removed.
    pub fn sample(&mut self) -> Result<Sample, String> {
        let mut raw = [0u8; 14];
        self.bus.write_read(self.addr, reg::ACCEL_XOUT_H, &mut raw)?;
        let s = decode(&raw);
        let g = self.mount.apply(s.gyro);
        Ok(Sample {
            accel: self.mount.apply(s.accel),
            gyro: [
                g[0] - self.gyro_bias[0],
                g[1] - self.gyro_bias[1],
                g[2] - self.gyro_bias[2],
            ],
        })
    }
}

/// Mahony-style complementary filter (proportional correction only).
pub struct Attitude {
    /// Orientation (w, x, y, z), body to world.
    pub q: [f64; 4],
    /// Accelerometer correction gain (1/s); larger trusts gravity more.
    pub kp: f64,
}

impl Default for Attitude {
    fn default() -> Self {
        Attitude {
            q: [1.0, 0.0, 0.0, 0.0],
            kp: 2.0,
        }
    }
}

impl Attitude {
    /// Advances by `dt` seconds with one body-frame sample.
    pub fn update(&mut self, s: &Sample, dt: f64) {
        let [w, x, y, z] = self.q;
        let mut omega = s.gyro;
        let n = (s.accel[0].powi(2) + s.accel[1].powi(2) + s.accel[2].powi(2)).sqrt();
        // Only trust the accelerometer near 1 g (not during impacts).
        if n > 0.5 * G && n < 1.5 * G {
            let a = [s.accel[0] / n, s.accel[1] / n, s.accel[2] / n];
            // World up (0, 0, 1) in the body frame.
            let v = [
                2.0 * (x * z - w * y),
                2.0 * (w * x + y * z),
                w * w - x * x - y * y + z * z,
            ];
            // e = a x v: rotation that brings the estimate onto the measurement.
            let e = [
                a[1] * v[2] - a[2] * v[1],
                a[2] * v[0] - a[0] * v[2],
                a[0] * v[1] - a[1] * v[0],
            ];
            for (o, ei) in omega.iter_mut().zip(e) {
                *o += self.kp * ei;
            }
        }
        let [gx, gy, gz] = omega;
        let h = 0.5 * dt;
        let q = [
            w + h * (-x * gx - y * gy - z * gz),
            x + h * (w * gx + y * gz - z * gy),
            y + h * (w * gy - x * gz + z * gx),
            z + h * (w * gz + x * gy - y * gx),
        ];
        let norm = q.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm.is_finite() && norm > 0.0 {
            self.q = q.map(|v| v / norm);
        }
    }

    /// Starts from the roll and pitch of one accelerometer sample (yaw 0).
    pub fn from_gravity(accel: [f64; 3]) -> Attitude {
        let roll = accel[1].atan2(accel[2]);
        let pitch = (-accel[0]).atan2((accel[1].powi(2) + accel[2].powi(2)).sqrt());
        let (sr, cr) = (0.5 * roll).sin_cos();
        let (sp, cp) = (0.5 * pitch).sin_cos();
        Attitude {
            q: [cr * cp, sr * cp, cr * sp, -sr * sp],
            ..Attitude::default()
        }
    }
}

/// `hwstate::Imu` from an MPU6050 and the attitude filter. The filter step is
/// the measured time between reads, clamped to [1, 100] ms.
pub struct Mpu6050Imu<B: I2c> {
    pub chip: Mpu6050<B>,
    pub attitude: Attitude,
    last: Option<Instant>,
}

impl<B: I2c> Mpu6050Imu<B> {
    pub fn new(mut chip: Mpu6050<B>) -> Result<Self, String> {
        let s = chip.sample()?;
        Ok(Mpu6050Imu {
            attitude: Attitude::from_gravity(s.accel),
            chip,
            last: None,
        })
    }
}

impl<B: I2c> Imu for Mpu6050Imu<B> {
    fn read(&mut self) -> Result<([f64; 4], [f64; 3]), String> {
        let s = self.chip.sample()?;
        let now = Instant::now();
        let dt = self
            .last
            .map_or(0.02, |t| now.duration_since(t).as_secs_f64())
            .clamp(0.001, 0.1);
        self.last = Some(now);
        self.attitude.update(&s, dt);
        Ok((self.attitude.q, s.gyro))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::projected_gravity;

    fn level(g_lsb: i16) -> SimI2c {
        let mut b = SimI2c::mpu(DEFAULT_ADDR, 0x68);
        b.set_sample(DEFAULT_ADDR, [0, 0, g_lsb], [0, 0, 0]);
        b
    }

    #[test]
    fn open_checks_who_am_i_and_configures() {
        let chip = Mpu6050::open(level(8192), DEFAULT_ADDR, Mount::IDENTITY, 0).unwrap();
        let r = chip.bus.regs[&DEFAULT_ADDR];
        assert_eq!(r[reg::PWR_MGMT_1 as usize], 0x01); // awake, gyro clock
        assert_eq!(r[reg::GYRO_CONFIG as usize], 0x08);
        assert_eq!(r[reg::ACCEL_CONFIG as usize], 0x08);
        assert_eq!(r[reg::CONFIG as usize], 0x03);
        let bad = SimI2c::mpu(DEFAULT_ADDR, 0x12);
        let e = Mpu6050::open(bad, DEFAULT_ADDR, Mount::IDENTITY, 0).err().unwrap();
        assert!(e.contains("WHO_AM_I"), "{e}");
        let e = Mpu6050::open(level(8192), 0x69, Mount::IDENTITY, 0).err().unwrap();
        assert!(e.contains("no device"), "{e}");
    }

    #[test]
    fn decode_scales_and_signs() {
        let mut raw = [0u8; 14];
        raw[4..6].copy_from_slice(&8192i16.to_be_bytes()); // az = 1 g
        raw[0..2].copy_from_slice(&(-4096i16).to_be_bytes()); // ax = -0.5 g
        raw[8..10].copy_from_slice(&655i16.to_be_bytes()); // gx = 10 deg/s
        let s = decode(&raw);
        assert!((s.accel[2] - G).abs() < 1e-9);
        assert!((s.accel[0] + 0.5 * G).abs() < 1e-9);
        assert!((s.gyro[0] - 10f64.to_radians()).abs() < 1e-9);
    }

    #[test]
    fn gyro_bias_is_removed() {
        let mut b = level(8192);
        b.set_sample(DEFAULT_ADDR, [0, 0, 8192], [131, -66, 0]); // ~2, ~-1 deg/s
        let mut chip = Mpu6050::open(b, DEFAULT_ADDR, Mount::IDENTITY, 10).unwrap();
        let s = chip.sample().unwrap();
        assert!(s.gyro.iter().all(|v| v.abs() < 1e-12));
    }

    #[test]
    fn mount_parse_and_apply() {
        let m = Mount::parse("-y, x, z").unwrap();
        assert_eq!(m.apply([1.0, 2.0, 3.0]), [-2.0, 1.0, 3.0]);
        assert_eq!(Mount::parse("x,y,z").unwrap(), Mount::IDENTITY);
        assert!(Mount::parse("x,x,z").is_err());
        assert!(Mount::parse("x,y").is_err());
        assert!(Mount::parse("x,y,w").is_err());
    }

    #[test]
    fn attitude_converges_to_a_tilt_and_tracks_the_gyro() {
        // Robot pitched forward by 0.3 rad, still: gravity seen in the body frame.
        let pitch: f64 = 0.3;
        let accel = [-pitch.sin() * G, 0.0, pitch.cos() * G];
        let mut att = Attitude::default();
        let s = Sample { accel, gyro: [0.0; 3] };
        for _ in 0..1000 {
            att.update(&s, 0.02);
        }
        // projected_gravity is world down in the body frame: (sin p, 0, -cos p).
        let g = projected_gravity(att.q);
        assert!((g[0] - pitch.sin()).abs() < 1e-3, "{g:?}");
        assert!((g[2] + pitch.cos()).abs() < 1e-3, "{g:?}");
        // from_gravity starts there at once.
        let g0 = projected_gravity(Attitude::from_gravity(accel).q);
        assert!((g0[0] - pitch.sin()).abs() < 1e-9 && g0[1].abs() < 1e-9, "{g0:?}");
        // Pure gyro (no accel trust): a roll rate of 1 rad/s for 0.2 s.
        let mut att = Attitude {
            kp: 0.0,
            ..Attitude::default()
        };
        let s = Sample {
            accel: [0.0; 3],
            gyro: [1.0, 0.0, 0.0],
        };
        for _ in 0..200 {
            att.update(&s, 0.001);
        }
        let g = projected_gravity(att.q);
        assert!((g[1] + 0.2f64.sin()).abs() < 1e-3, "{g:?}");
    }

    #[test]
    fn imu_trait_reports_level_and_rates() {
        let mut b = level(8192);
        b.set_sample(DEFAULT_ADDR, [0, 0, 8192], [0, 0, 0]);
        let chip = Mpu6050::open(b, DEFAULT_ADDR, Mount::IDENTITY, 0).unwrap();
        let mut imu = Mpu6050Imu::new(chip).unwrap();
        imu.chip.bus.set_sample(DEFAULT_ADDR, [0, 0, 8192], [0, 0, 655]);
        let (q, gyro) = imu.read().unwrap();
        assert!((gyro[2] - 10f64.to_radians()).abs() < 1e-9);
        let g = projected_gravity(q);
        assert!((g[2] + 1.0).abs() < 1e-6, "{g:?}");
    }
}
