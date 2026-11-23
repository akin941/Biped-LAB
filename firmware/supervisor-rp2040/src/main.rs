//! RP2040 firmware of the safety supervisor for the DS3218MG build
//! (ADR-0015, WP5-10). Everything that decides is in `supervisor-core`
//! (host-tested); this file only wires it to the peripherals.
//!
//! Raspberry Pi Pico wiring:
//!
//! | Pico pin  | Use                                                   |
//! |-----------|-------------------------------------------------------|
//! | GP0 / GP1 | UART0 TX → Pi RX, RX ← Pi TX, 115200 8N1, 3.3 V logic |
//! | GP2–GP11  | servo signals, joints 0–9 in the runtime's order       |
//! | GP26      | battery through a 22 kΩ / 10 kΩ divider (ADC0)         |
//! | GP25      | on-board LED: on while a fault is latched              |
//!
//! Servo power never goes through the Pico: the 6 V UBEC feeds the servos,
//! behind the e-stop switch (S-3), with a common ground.
//!
//! Loop: every received targets frame goes to the supervisor and is answered
//! with one status frame; every 20 ms the supervisor also ticks on its own, so
//! its watchdog holds and then stops the pulses when the Pi goes silent. The
//! RP2040's hardware watchdog resets the chip if this loop ever stalls; after
//! a reset the outputs stay low until the Pi clears and enables again.
//! Nothing here has run on a chip yet (`env: hw`).

#![no_std]
#![no_main]

use embedded_hal::digital::OutputPin;
use embedded_hal::pwm::SetDutyCycle;
use embedded_hal_nb::nb;
use panic_halt as _;
use rp2040_hal::{
    self as hal,
    adc::{Adc, AdcPin},
    clocks::init_clocks_and_plls,
    fugit::{ExtU32, RateExtU32},
    pac,
    uart::{DataBits, StopBits, UartConfig, UartPeripheral},
    Clock, Sio, Timer, Watchdog,
};
use supervisor_core::frame::{Reader, Status, Targets, TARGETS_LEN};
use supervisor_core::{joints, Limits, Output, Supervisor, N_JOINTS};

#[link_section = ".boot2"]
#[used]
pub static BOOT2_FIRMWARE: [u8; 256] = rp2040_boot2::BOOT_LOADER_W25Q080;

const XTAL_HZ: u32 = 12_000_000;
const BAUD: u32 = 115_200;
/// Servo frame: 1 us PWM ticks, 20 ms period (50 Hz).
const PWM_PERIOD_US: u16 = 20_000;
const TICK_MS: u32 = 20;
/// Battery divider: V_bat = V_adc * (22k + 10k) / 10k.
const DIVIDER: f32 = 3.2;
const ADC_VREF: f32 = 3.3;

/// Sets one slice to 1 MHz / 20 ms and routes both its channels to pins.
macro_rules! servo_slice {
    ($slice:expr, $div:expr, $pin_a:expr, $pin_b:expr) => {{
        let mut s = $slice;
        s.set_div_int($div);
        s.set_div_frac(0);
        s.set_top(PWM_PERIOD_US - 1);
        s.enable();
        s.channel_a.output_to($pin_a);
        s.channel_b.output_to($pin_b);
        s
    }};
}

#[hal::entry]
fn main() -> ! {
    let mut pac = pac::Peripherals::take().unwrap();
    let mut watchdog = Watchdog::new(pac.WATCHDOG);
    let clocks = init_clocks_and_plls(
        XTAL_HZ,
        pac.XOSC,
        pac.CLOCKS,
        pac.PLL_SYS,
        pac.PLL_USB,
        &mut pac.RESETS,
        &mut watchdog,
    )
    .ok()
    .unwrap();
    let sio = Sio::new(pac.SIO);
    let pins = hal::gpio::Pins::new(
        pac.IO_BANK0,
        pac.PADS_BANK0,
        sio.gpio_bank0,
        &mut pac.RESETS,
    );
    let timer = Timer::new(pac.TIMER, &mut pac.RESETS, &clocks);

    let uart = UartPeripheral::new(
        pac.UART0,
        (pins.gpio0.into_function(), pins.gpio1.into_function()),
        &mut pac.RESETS,
    )
    .enable(
        UartConfig::new(BAUD.Hz(), DataBits::Eight, None, StopBits::One),
        clocks.peripheral_clock.freq(),
    )
    .unwrap();

    // 125 MHz system clock / 125 = 1 MHz PWM counter.
    let div = (clocks.system_clock.freq().to_Hz() / 1_000_000) as u8;
    let slices = hal::pwm::Slices::new(pac.PWM, &mut pac.RESETS);
    let mut s1 = servo_slice!(slices.pwm1, div, pins.gpio2, pins.gpio3);
    let mut s2 = servo_slice!(slices.pwm2, div, pins.gpio4, pins.gpio5);
    let mut s3 = servo_slice!(slices.pwm3, div, pins.gpio6, pins.gpio7);
    let mut s4 = servo_slice!(slices.pwm4, div, pins.gpio8, pins.gpio9);
    let mut s5 = servo_slice!(slices.pwm5, div, pins.gpio10, pins.gpio11);
    let mut set_pulses = |p: &[u16; N_JOINTS]| {
        // Duty in counter ticks = pulse width in us; 0 = no pulse.
        let _ = s1.channel_a.set_duty_cycle(p[0]);
        let _ = s1.channel_b.set_duty_cycle(p[1]);
        let _ = s2.channel_a.set_duty_cycle(p[2]);
        let _ = s2.channel_b.set_duty_cycle(p[3]);
        let _ = s3.channel_a.set_duty_cycle(p[4]);
        let _ = s3.channel_b.set_duty_cycle(p[5]);
        let _ = s4.channel_a.set_duty_cycle(p[6]);
        let _ = s4.channel_b.set_duty_cycle(p[7]);
        let _ = s5.channel_a.set_duty_cycle(p[8]);
        let _ = s5.channel_b.set_duty_cycle(p[9]);
    };
    set_pulses(&[0; N_JOINTS]);

    let mut adc = Adc::new(pac.ADC, &mut pac.RESETS);
    let mut battery_pin = AdcPin::new(pins.gpio26.into_floating_input()).unwrap();
    let mut led = pins.gpio25.into_push_pull_output();

    let mut sup = Supervisor::new(joints::biped_ds3218(), Limits::default());
    let mut reader = Reader::<TARGETS_LEN>::default();
    let now_ms = || (timer.get_counter().ticks() / 1000) as u32;
    let mut next_tick = now_ms();
    let mut battery_v = 0.0f32;

    watchdog.start(100.millis());
    loop {
        watchdog.feed();
        let mut apply = |out: &Output, battery_v: f32| {
            set_pulses(&out.pulses_us.unwrap_or([0; N_JOINTS]));
            let _ = if sup_latched(out) {
                led.set_high()
            } else {
                led.set_low()
            };
            Status {
                seq: 0,
                bits: out.bits,
                battery_mv: (battery_v * 1000.0) as u16,
                pulses_us: out.pulses_us.unwrap_or([0; N_JOINTS]),
            }
        };

        if now_ms().wrapping_sub(next_tick) < u32::MAX / 2 {
            next_tick = next_tick.wrapping_add(TICK_MS);
            let raw: u16 = adc.read(&mut battery_pin).unwrap_or(0);
            battery_v = raw as f32 * (ADC_VREF / 4095.0) * DIVIDER;
            let out = sup.tick(now_ms(), battery_v);
            apply(&out, battery_v);
        }

        let mut buf = [0u8; 32];
        let n = match uart.read_raw(&mut buf) {
            Ok(n) => n,
            Err(nb::Error::WouldBlock) => 0,
            Err(nb::Error::Other(_)) => {
                sup.bad_frame(); // framing, parity, break or overrun
                0
            }
        };
        for &b in &buf[..n] {
            let Some(raw) = reader.push(b) else { continue };
            match Targets::decode(&raw) {
                Ok(t) => {
                    let now = now_ms();
                    sup.on_targets(&t, now);
                    let out = sup.tick(now, battery_v);
                    let mut status = apply(&out, battery_v);
                    status.seq = sup.last_seq;
                    uart.write_full_blocking(&status.encode());
                }
                Err(_) => sup.bad_frame(),
            }
        }
    }
}

fn sup_latched(out: &Output) -> bool {
    out.bits & supervisor_core::bits::LATCHED != 0
}
