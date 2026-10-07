//! The game's number type, represented as an IEEE `f64` behind the Q32.32 API.
//!
//! Unity's BoplFixedMath (`Assets/Scripts/Assembly-CSharp/BoplFixedMath/Fix.cs`)
//! is a Q32.32 fixed-point type chosen for lockstep multiplayer determinism.
//! The rollout backend only needs determinism within one binary, which IEEE
//! doubles on one machine already provide, so this module keeps every method
//! name and semantic contract of the fixed-point type while storing an `f64`.
//! The cargo feature `q32` swaps in the bit-exact reference arithmetic
//! (`fixed_q32.rs`) for the golden-verification tooling.
//!
//! What is kept exact:
//! - Constants are the game's truncated Q32.32 values (`PI`, `HALF_PI`, `TAU`,
//!   `DT`), not the mathematical ones, so every comparison against them lands
//!   on the same side as in the reference.
//! - `from_f32`/`from_f64`/`from_raw` snap to the Q32.32 grid exactly like the
//!   reference's `(Fix)0.95m` conversions, so every authored threshold is the
//!   same number in both representations.
//! - Sums and differences of grid values with < 2^53 raw magnitude are exact
//!   (`timer += DT` lands exactly where the fixed-point timer does), and the
//!   product of an integer with a grid value is exact (`tick * DT`).
//! - Trig arguments are reduced by the reference's `ClampSinValue` quadrant
//!   logic, so `sin(TAU)`, `cos(HALF_PI)`, and every other multiple of the
//!   game's `HALF_PI` are exactly zero or one, as in the table lookup.
//! - `Atan2` keeps the reference's rational approximation (`x / (1 + 0.28 x²)`,
//!   including its non-standard `|ratio| >= 1` branch and saturation
//!   fallback) because the game's perimeter arithmetic genuinely uses it.
//!
//! What differs: general multiplication/division round to the nearest double
//! instead of truncating to 2^-32; `sqrt`, `sin`, `cos`, `tan`, `atan`, `acos`
//! use libm rather than the interpolated tables and the Euler series;
//! saturating "slow" arithmetic never saturates.
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::cmp::Ordering;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

const SCALE: f64 = 4_294_967_296.0;

/// `v.round() as i64` (half away from zero) without the libm call. `v as i64`
/// truncates toward zero exactly (saturating beyond i64), and `v - t` is exact:
/// a double with a fractional part has |v| < 2^52, where the integer part
/// converts back exactly; a double at or beyond 2^52 is already an integer.
#[inline]
fn round_i64(v: f64) -> i64 {
    // The common case: `round` lowers to a truncate of `v + copysign(0.5 -
    // ulp, v)` and the conversion of an in-range integral double needs no
    // saturation or NaN handling. Every observation field goes through here,
    // so the checked path only handles what the fast one excludes.
    if v.abs() < 4_611_686_018_427_387_904.0 {
        // SAFETY: |v| < 2^62, so the rounded value is finite and within i64.
        return unsafe { v.round().to_int_unchecked::<i64>() };
    }
    let t = v as i64;
    let fraction = v - t as f64;
    // Saturating like the `as` conversion for |v| >= 2^63 (fraction is then
    // the whole overshoot, never a panic).
    t.saturating_add((fraction >= 0.5) as i64)
        .saturating_sub((fraction <= -0.5) as i64)
}

/// `v.floor() as i64` without the libm call, by the same argument.
#[inline]
const fn floor_i64(v: f64) -> i64 {
    let t = v as i64;
    t.saturating_sub((t as f64 > v) as i64)
}

/// `sin(x)` for `0 <= x <= HALF_PI` (already quadrant-reduced) without the
/// libm call: fdlibm's `__kernel_sin`/`__kernel_cos` polynomials, which are
/// accurate to under one ulp on `|x| <= pi/4`, with the upper half of the
/// interval folded through `cos(pi/2 - x)`. libm's `sin` spends most of its
/// time on the general range reduction this caller has already done. The
/// result differs from libm by at most one ulp (~1e-16), which the
/// `fast_sin` table floor to the 2^-32 grid absorbs except when the exact
/// value lies within 1e-16 of a grid boundary.
#[inline]
// The coefficients are fdlibm's literals (k_sin.c, k_cos.c) as published, so
// they can be checked against the source; they round to the same doubles.
#[allow(clippy::excessive_precision)]
fn sin_reduced(x: f64) -> f64 {
    const S1: f64 = -1.666_666_666_666_663_243_48e-01;
    const S2: f64 = 8.333_333_333_322_489_461_24e-03;
    const S3: f64 = -1.984_126_982_985_794_931_34e-04;
    const S4: f64 = 2.755_731_370_707_006_767_89e-06;
    const S5: f64 = -2.505_076_025_340_686_341_95e-08;
    const S6: f64 = 1.589_690_995_211_550_102_21e-10;
    const C1: f64 = 4.166_666_666_666_660_190_37e-02;
    const C2: f64 = -1.388_888_888_887_410_957_49e-03;
    const C3: f64 = 2.480_158_728_947_672_941_78e-05;
    const C4: f64 = -2.755_731_435_139_066_330_35e-07;
    const C5: f64 = 2.087_572_321_298_174_827_90e-09;
    const C6: f64 = -1.135_964_755_778_819_482_65e-11;
    if x <= std::f64::consts::FRAC_PI_4 {
        let z = x * x;
        let w = z * z;
        let r = S2 + z * (S3 + z * S4) + z * w * (S5 + z * S6);
        x + z * x * (S1 + z * r)
    } else {
        // Exact by Sterbenz: x and pi/2 are within a factor of two.
        let y = std::f64::consts::FRAC_PI_2 - x;
        let z = y * y;
        let w = z * z;
        let r = z * (C1 + z * (C2 + z * C3)) + w * w * (C4 + z * (C5 + z * C6));
        let hz = 0.5 * z;
        let w = 1.0 - hz;
        w + ((1.0 - w) - hz + z * r)
    }
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(transparent)]
pub struct Fix(f64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MathError {
    DivisionByZero,
    Domain,
    Overflow,
}

impl Fix {
    pub const ZERO: Self = Self(0.0);
    pub const ONE: Self = Self(1.0);
    pub const MIN: Self = Self::from_raw(i64::MIN);
    pub const MAX: Self = Self::from_raw(i64::MAX);
    pub const PI: Self = Self::from_raw(13_493_037_704);
    pub const HALF_PI: Self = Self::from_raw(6_746_518_852);
    pub const TAU: Self = Self::from_raw(26_986_075_409);
    /// Fix.cs `Cos`: `Sin(x + (x > 0 ? -3π/2 : π/2))` with the raw constant
    /// -20239556556.
    const COS_SHIFT_POSITIVE: Self = Self::from_raw(-20_239_556_556);
    /// Fix.cs `Atan2`'s `(Fix)0.28m`.
    const ATAN2_C: Self = Self::from_raw(1_202_590_842);

    /// Construct from a raw Q32.32 integer (the ABI and extracted-data encoding).
    pub const fn from_raw(raw: i64) -> Self {
        Self(raw as f64 / SCALE)
    }
    /// The nearest raw Q32.32 integer (the ABI and extracted-data encoding).
    pub fn raw(self) -> i64 {
        round_i64(self.0 * SCALE)
    }
    /// `raw` of each value. Inside +-2^30 every lane takes `round_i64`'s fast
    /// path, which then vectorizes; a row's dozen fields convert in a few
    /// packed instructions instead of a dozen scalar sequences.
    #[inline]
    pub fn raw_array<const N: usize>(values: [Self; N]) -> [i64; N] {
        let scaled = values.map(|value| value.0 * SCALE);
        let raws = if scaled.iter().fold(true, |fast, v| fast & (v.abs() < 4_611_686_018_427_387_904.0)) {
            // SAFETY: |v| < 2^62, so the rounded value is finite and within i64.
            scaled.map(|v| unsafe { v.round().to_int_unchecked::<i64>() })
        } else {
            scaled.map(round_i64)
        };
        debug_assert_eq!(raws, values.map(Self::raw));
        raws
    }
    pub const fn from_int(value: i64) -> Self {
        Self(value as f64)
    }
    /// Floor to an integer, like the reference's arithmetic shift by 32.
    pub const fn to_int(self) -> i64 {
        floor_i64(self.0)
    }
    /// `(Fix)0.2f` in the reference: scale and truncate toward zero onto the
    /// Q32.32 grid, so authored thresholds are the same number in both
    /// representations.
    pub const fn from_f32(value: f32) -> Self {
        Self::from_raw((value * SCALE as f32) as i64)
    }
    /// `(Fix)0.95m` in the reference: scale and truncate toward zero onto the
    /// Q32.32 grid.
    pub const fn from_f64(value: f64) -> Self {
        Self::from_raw((value * SCALE) as i64)
    }
    pub const fn to_f64(self) -> f64 {
        self.0
    }
    pub fn sign(self) -> i64 {
        if self.0 > 0.0 {
            1
        } else if self.0 < 0.0 {
            -1
        } else {
            0
        }
    }
    pub fn sign2(self) -> i64 {
        if self.0 < 0.0 { -1 } else { 1 }
    }
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }
    pub fn slow_abs(self) -> Self {
        Self(self.0.abs())
    }
    pub fn floor(self) -> Self {
        Self(self.0.floor())
    }
    pub fn ceil(self) -> Self {
        Self(self.0.ceil())
    }
    /// Fix.cs `Round`: half-way values round to the even integer.
    pub fn round(self) -> Self {
        Self(self.0.round_ties_even())
    }
    pub fn slow_add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
    pub fn slow_sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
    pub fn slow_mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }
    pub fn checked_div(self, rhs: Self) -> Result<Self, MathError> {
        if rhs.0 == 0.0 {
            return Err(MathError::DivisionByZero);
        }
        Ok(Self(self.0 / rhs.0))
    }
    /// C# `%`: the remainder takes the dividend's sign, exactly `f64::rem`.
    pub fn checked_rem(self, rhs: Self, _slow: bool) -> Result<Self, MathError> {
        if rhs.0 == 0.0 {
            return Err(MathError::DivisionByZero);
        }
        // fmod is a software loop here. Below one period it is the identity,
        // and below two it is |a| - m with a's sign, exact (Sterbenz): the
        // remainders the game takes (an orientation by TAU, a phase by ONE)
        // are nearly all one of these. NaN and infinities fail both tests.
        let (a, m) = (self.0, rhs.0.abs());
        Ok(Self(if a.abs() < m {
            a
        } else if a.abs() < 2.0 * m {
            (a.abs() - m).copysign(a)
        } else {
            a % rhs.0
        }))
    }
    /// `checked_rem(rhs, true)`, with the fmod beyond two periods done
    /// exactly by a fused multiply-add where it can be: fmod's result is
    /// always representable, so for the true quotient `q` the fused
    /// `a - q * m` (one rounding of the exact value) *is* it. `trunc(a / m)`
    /// is that quotient unless the division rounded across an integer, and
    /// then the exact `a - q * m` falls outside `(0, m)`, as does its
    /// rounding; a zero remainder (whose sign fmod takes from `a`), a
    /// negative `a`, NaN and infinities take the fmod. Kept apart from
    /// `checked_rem`, which inlines into the integrator.
    pub fn checked_rem_fused(self, rhs: Self) -> Result<Self, MathError> {
        let (a, m) = (self.0, rhs.0.abs());
        if a > 0.0 && a >= 2.0 * m {
            let r = (-(a / m).trunc()).mul_add(m, a);
            if r > 0.0 && r < m {
                debug_assert_eq!(r.to_bits(), (a % rhs.0).to_bits());
                return Ok(Self(r));
            }
        }
        self.checked_rem(rhs, true)
    }
    pub fn sqrt(self) -> Result<Self, MathError> {
        if self.0 < 0.0 {
            return Err(MathError::Domain);
        }
        Ok(Self(self.0.sqrt()))
    }
    /// Fix.cs `ClampSinValue`: reduce into `[0, HALF_PI)` with the quadrant
    /// flags, using the game's truncated `TAU`/`PI`/`HALF_PI`. Reducing here
    /// (rather than letting libm reduce by the true 2π) keeps the game's
    /// multiples of `HALF_PI` exactly on the axes.
    fn sin_quadrant(self) -> (f64, bool, bool) {
        let tau = Self::TAU.0;
        let pi = Self::PI.0;
        let half_pi = Self::HALF_PI.0;
        let mut angle = self.0;
        if !(0.0..tau).contains(&angle) {
            // One period off (a stored rotation plus `cos`'s shift, or just
            // past TAU) is the only case that occurs; `fmod` is the identity
            // on (-tau, 0) and exactly `angle - tau` on [tau, 2 tau)
            // (Sterbenz), so these branches give the same double as the
            // general remainder without its cost.
            if (-tau..0.0).contains(&angle) {
                angle += tau;
            } else if (tau..2.0 * tau).contains(&angle) {
                angle -= tau;
            } else {
                angle %= tau;
                if angle < 0.0 {
                    angle += tau;
                }
            }
        }
        let vertical = angle >= pi;
        if vertical {
            angle -= pi;
        }
        let horizontal = angle >= half_pi;
        if horizontal {
            angle -= half_pi;
        }
        (angle, horizontal, vertical)
    }
    pub fn sin(self) -> Self {
        let (angle, horizontal, vertical) = self.sin_quadrant();
        // The mirrored quadrant reads the table from its far end,
        // `SinLut[LUT_SIZE - 1 - i]`, i.e. sin(HALF_PI - angle).
        let value = if horizontal {
            sin_reduced(Self::HALF_PI.0 - angle)
        } else {
            sin_reduced(angle)
        };
        Self(if vertical { -value } else { value })
    }
    /// Fix.cs `FastSin`: the un-interpolated lookup `SinLut[angle >> 15]`
    /// (mirrored from the far end in the second quadrant). Reproduced
    /// exactly rather than approximated by `sin`: the table's entry `i` is
    /// `floor(sin(i * (PI/2) / 205886) * 2^32)` (all 205887 entries of
    /// `data/sinlut.bin` match this formula), so the lookup carries both the
    /// index truncation (up to 7.6e-6 rad) and the table's 1.0000185 angle
    /// scale. `Vec2(angle)` (Vec2.cs:31) builds every FixTransform basis
    /// with it: map platform bases (FixTransform.cs:45) and slime colliders
    /// sit at the table's angle, not the exact one, and a body standing on a
    /// tilted platform settles where that basis puts it.
    pub fn fast_sin(self) -> Self {
        const LUT_LAST: f64 = 205886.0;
        let (angle, horizontal, vertical) = self.sin_quadrant();
        // `angle` is in units; raw >> 15 is floor(angle * 2^17).
        let index = (angle * 131_072.0).floor().min(LUT_LAST);
        let index = if horizontal { LUT_LAST - index } else { index };
        let value = sin_reduced(index * (std::f64::consts::FRAC_PI_2 / LUT_LAST));
        let value = (value * 4_294_967_296.0).floor() / 4_294_967_296.0;
        Self(if vertical { -value } else { value })
    }
    /// The table entry `fast_sin` reads, with its sign: `fast_sin` is
    /// `fast_sin_at(fast_sin_entry())`, so two angles with one entry share
    /// the value. The entry is the mirrored index's bits (an integer, or
    /// -0 for a -0 angle, so bit 0 is free) with the sign in bit 0.
    pub fn fast_sin_entry(self) -> u64 {
        const LUT_LAST: f64 = 205886.0;
        let (angle, horizontal, vertical) = self.sin_quadrant();
        let index = (angle * 131_072.0).floor().min(LUT_LAST);
        let index = if horizontal { LUT_LAST - index } else { index };
        index.to_bits() | vertical as u64
    }
    /// `fast_cos`'s entry, as `fast_sin_entry`.
    pub fn fast_cos_entry(self) -> u64 {
        Self(self.0 + if self.0 > 0.0 { Self::COS_SHIFT_POSITIVE.0 } else { Self::HALF_PI.0 }).fast_sin_entry()
    }
    /// The value at a `fast_sin_entry`.
    pub fn fast_sin_at(entry: u64) -> Self {
        const LUT_LAST: f64 = 205886.0;
        let index = f64::from_bits(entry & !1);
        let value = sin_reduced(index * (std::f64::consts::FRAC_PI_2 / LUT_LAST));
        let value = (value * 4_294_967_296.0).floor() / 4_294_967_296.0;
        Self(if entry & 1 != 0 { -value } else { value })
    }
    pub fn cos(self) -> Self {
        Self(
            self.0
                + if self.0 > 0.0 {
                    Self::COS_SHIFT_POSITIVE.0
                } else {
                    Self::HALF_PI.0
                },
        )
        .sin()
    }
    pub fn fast_cos(self) -> Self {
        Self(
            self.0
                + if self.0 > 0.0 {
                    Self::COS_SHIFT_POSITIVE.0
                } else {
                    Self::HALF_PI.0
                },
        )
        .fast_sin()
    }
    pub fn tan(self) -> Self {
        // Fix.cs `Tan`: reduce by PI, fold the second quadrant, then look up.
        let mut angle = self.0 % Self::PI.0;
        let mut negative = angle < 0.0;
        if negative {
            angle = -angle;
        }
        if angle > Self::HALF_PI.0 {
            negative = !negative;
            angle = Self::PI.0 - angle;
        }
        let value = angle.tan();
        Self(if negative { -value } else { value })
    }
    pub fn atan(self) -> Self {
        Self(self.0.atan())
    }
    pub fn acos(self) -> Result<Self, MathError> {
        // A normalized double's component can exceed 1 by rounding; the
        // reference's Q32.32 normalization has the same slack of one raw unit,
        // so accept it here and reject anything beyond as the reference does.
        let slack = 1.0 + 1.0 / SCALE;
        if self.0 < -slack || self.0 > slack {
            return Err(MathError::Domain);
        }
        if self.0 == 0.0 {
            return Ok(Self::HALF_PI);
        }
        Ok(Self(self.0.clamp(-1.0, 1.0).acos()))
    }
    /// Fix.cs `Atan2`: the rational approximation `r / (1 + 0.28 r²)` for
    /// `|r| < 1`, the reference's non-standard `(HALF_PI - r) / (r² + 0.28)`
    /// otherwise, and `±HALF_PI` where the fixed-point denominator saturates.
    pub fn atan2(self, x: Self) -> Self {
        let y = self;
        if x.0 == 0.0 {
            return if y.0 > 0.0 {
                Self::HALF_PI
            } else if y.0 == 0.0 {
                Self::ZERO
            } else {
                -Self::HALF_PI
            };
        }
        let ratio = y / x;
        let c = Self::ATAN2_C.0;
        let denominator = 1.0 + c * ratio.0 * ratio.0;
        if denominator >= Self::MAX.0 {
            return if y.0 < 0.0 {
                -Self::HALF_PI
            } else {
                Self::HALF_PI
            };
        }
        if ratio.0.abs() < 1.0 {
            let angle = Self(ratio.0 / denominator);
            if x.0 < 0.0 {
                if y.0 < 0.0 {
                    angle - Self::PI
                } else {
                    angle + Self::PI
                }
            } else {
                angle
            }
        } else {
            let angle = Self((Self::HALF_PI.0 - ratio.0) / (ratio.0 * ratio.0 + c));
            if y.0 < 0.0 { angle - Self::PI } else { angle }
        }
    }
}

impl PartialEq for Fix {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for Fix {}
impl PartialOrd for Fix {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
    fn lt(&self, other: &Self) -> bool {
        self.0 < other.0
    }
    fn le(&self, other: &Self) -> bool {
        self.0 <= other.0
    }
    fn gt(&self, other: &Self) -> bool {
        self.0 > other.0
    }
    fn ge(&self, other: &Self) -> bool {
        self.0 >= other.0
    }
}
impl Ord for Fix {
    /// A total order: NaN never occurs (every division and square root is
    /// checked), and `-0.0 == 0.0` as in the comparison operators.
    fn cmp(&self, other: &Self) -> Ordering {
        if self.0 < other.0 {
            Ordering::Less
        } else if self.0 > other.0 {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }
}

/// Serialized as the raw Q32.32 integer, like the reference and the fixtures.
impl Serialize for Fix {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_i64(self.raw())
    }
}
impl<'de> Deserialize<'de> for Fix {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        i64::deserialize(deserializer).map(Self::from_raw)
    }
}

impl Add for Fix {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}
impl Sub for Fix {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}
impl Mul for Fix {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }
}
impl Div for Fix {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        self.checked_div(rhs).expect("fixed division by zero")
    }
}
impl Neg for Fix {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0)
    }
}
impl AddAssign for Fix {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl SubAssign for Fix {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl MulAssign for Fix {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl DivAssign for Fix {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_constants_round_trip() {
        for raw in [
            0,
            1,
            -1,
            71_582_788,
            13_493_037_704,
            -6_442_450_944,
            1 << 40,
        ] {
            assert_eq!(Fix::from_raw(raw).raw(), raw);
        }
        assert_eq!(Fix::from_f32(0.2).raw(), (0.2f32 * 4_294_967_296.0) as i64);
        assert_eq!(
            Fix::from_f64(0.95).raw(),
            (0.95f64 * 4_294_967_296.0) as i64
        );
        assert_eq!(
            Fix::from_f64(-11.3).raw(),
            (-11.3f64 * 4_294_967_296.0) as i64
        );
    }

    #[test]
    fn timers_step_exactly() {
        let dt = Fix::from_raw(71_582_788);
        let mut timer = Fix::from_raw(-6_442_450_944);
        for _ in 0..240 {
            timer += dt;
        }
        assert_eq!(timer.raw(), -6_442_450_944 + 240 * 71_582_788);
        assert_eq!((Fix::from_int(7200) * dt).raw(), 7200 * 71_582_788);
        assert!(Fix::from_int(60) * dt < Fix::ONE);
    }

    #[test]
    fn axes_are_exact() {
        assert_eq!(Fix::ZERO.sin(), Fix::ZERO);
        assert_eq!(Fix::ZERO.cos(), Fix::ONE);
        assert_eq!(Fix::HALF_PI.sin(), Fix::ONE);
        // The game's TAU is 2*PI + 1 raw unit, so Cos(HALF_PI) = Sin(-PI)
        // reduces to one raw unit past PI: -2^-32 here, and the reference's
        // interpolated table rounds the same value down to zero.
        assert_eq!(Fix::HALF_PI.cos().raw(), -1);
        assert_eq!(Fix::PI.sin(), Fix::ZERO);
        assert_eq!(Fix::PI.cos(), -Fix::ONE);
        assert_eq!(Fix::TAU.sin(), Fix::ZERO);
        assert_eq!(Fix::TAU.cos(), Fix::ONE);
        assert_eq!((-Fix::HALF_PI).sin(), -Fix::ONE);
        assert_eq!((Fix::PI + Fix::HALF_PI).cos(), Fix::ZERO);
    }

    #[test]
    fn round_i64_is_round_half_away_from_zero() {
        let reference = |v: f64| -> i64 {
            let t = v as i64;
            let fraction = v - t as f64;
            t.saturating_add((fraction >= 0.5) as i64)
                .saturating_sub((fraction <= -0.5) as i64)
        };
        let mut samples = vec![
            0.0,
            -0.0,
            0.5,
            -0.5,
            1.5,
            -1.5,
            2.5,
            0.499_999_999_999_999_94,
            -0.499_999_999_999_999_94,
            4_503_599_627_370_495.5,
            -4_503_599_627_370_495.5,
            4_503_599_627_370_496.0,
            4_611_686_018_427_387_904.0,
            -4_611_686_018_427_387_904.0,
            9.3e18,
            -9.3e18,
            f64::MAX,
            f64::MIN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..200_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let magnitude = 2f64.powi((state % 70) as i32 - 5);
            let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
            samples.push((unit - 0.5) * magnitude);
            samples.push(((state >> 20) as f64 - (1u64 << 43) as f64) * 0.5);
        }
        for v in samples {
            assert_eq!(round_i64(v), reference(v), "{v:e}");
        }
    }

    #[test]
    fn sin_kernel_matches_libm() {
        // Every table index of `fast_sin` (the values the game's bases are
        // built from) floors to the same 2^-32 grid value as libm did, and
        // the kernel is within one ulp of libm across the reduced range.
        const LUT_LAST: f64 = 205886.0;
        let mut floor_mismatches = 0;
        for i in 0..=205886u32 {
            let x = f64::from(i) * (std::f64::consts::FRAC_PI_2 / LUT_LAST);
            let libm = (x.sin() * SCALE).floor();
            let kernel = (sin_reduced(x) * SCALE).floor();
            floor_mismatches += u32::from(libm != kernel);
        }
        assert_eq!(floor_mismatches, 0);
        let mut worst_ulps = 0.0f64;
        for i in 0..1_000_000u32 {
            let x = f64::from(i) / 1_000_000.0 * Fix::HALF_PI.0;
            let libm = x.sin();
            let kernel = sin_reduced(x);
            let ulp = (libm.abs().max(f64::MIN_POSITIVE)).log2().floor().exp2() * f64::EPSILON;
            worst_ulps = worst_ulps.max((libm - kernel).abs() / ulp);
        }
        assert!(worst_ulps <= 1.0, "kernel off by {worst_ulps} ulps");
        // The reduction branches agree with the general remainder.
        for raw in [
            -26_986_075_409i64,
            -20_239_556_556,
            -13_493_037_704,
            -1,
            1,
            26_986_075_408,
            26_986_075_409,
            26_986_075_410,
            40_000_000_000,
            53_972_150_817,
            53_972_150_818,
            80_000_000_000,
            -80_000_000_000,
        ] {
            let angle = Fix::from_raw(raw).0;
            let tau = Fix::TAU.0;
            // The previous implementation: the general remainder every time.
            let mut reference = angle;
            if !(0.0..tau).contains(&reference) {
                reference %= tau;
                if reference < 0.0 {
                    reference += tau;
                }
            }
            let vertical = reference >= Fix::PI.0;
            if vertical {
                reference -= Fix::PI.0;
            }
            let horizontal = reference >= Fix::HALF_PI.0;
            if horizontal {
                reference -= Fix::HALF_PI.0;
            }
            assert_eq!(
                Fix::from_raw(raw).sin_quadrant(),
                (reference, horizontal, vertical),
                "raw {raw}"
            );
        }
    }

    #[test]
    fn rounding_and_signs() {
        assert_eq!(Fix::from_f64(2.5).round(), Fix::from_int(2));
        assert_eq!(Fix::from_f64(3.5).round(), Fix::from_int(4));
        assert_eq!(Fix::from_f64(-1.5).to_int(), -2);
        assert_eq!(Fix::from_f64(-0.25).sign(), -1);
        assert_eq!(Fix::ZERO.sign(), 0);
        assert_eq!(Fix::ZERO.sign2(), 1);
        assert_eq!(
            Fix::from_f64(-7.5).checked_rem(Fix::from_int(2), true),
            Ok(Fix::from_f64(-1.5))
        );
        assert_eq!(
            Fix::ONE.checked_div(Fix::ZERO),
            Err(MathError::DivisionByZero)
        );
        assert_eq!((-Fix::ONE).sqrt(), Err(MathError::Domain));
    }

    #[test]
    fn rem_matches_fmod_bit_for_bit() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let moduli = [Fix::TAU.0, Fix::PI.0, 1.0, 360.0, 0.37, 1e-300, f64::MIN_POSITIVE, 1e300, f64::MAX, f64::INFINITY, -2.0];
        let mut values = vec![0.0, -0.0, f64::NAN, f64::INFINITY, -f64::INFINITY, f64::MIN_POSITIVE / 4.0, f64::MAX];
        for &m in &moduli {
            for scale in [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0] {
                let v: f64 = m * scale;
                values.extend([v, v.next_up(), v.next_down()]);
            }
        }
        for _ in 0..200_000 {
            let m = moduli[(next() % moduli.len() as u64) as usize];
            let unit = (next() >> 11) as f64 / (1u64 << 53) as f64;
            values.push(m * (unit * 6.0 - 3.0));
            values.push(f64::from_bits(next()));
        }
        for &a in &values {
            for &m in &moduli {
                let fast = Fix(a).checked_rem(Fix(m), true).unwrap().0;
                assert_eq!(fast.to_bits(), (a % m).to_bits(), "{a:e} % {m:e}");
            }
        }
    }

    #[test]
    fn integer_conversions_match_libm() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut samples = vec![
            0.0,
            -0.0,
            0.5,
            -0.5,
            1.5,
            -1.5,
            2.5,
            -2.5,
            0.499_999_999_999_999_94,
            -0.499_999_999_999_999_94,
            4_503_599_627_370_495.5,
            -4_503_599_627_370_495.5,
            9.3e18,
            -9.3e18,
            1e30,
            -1e30,
            f64::MAX,
            f64::MIN,
        ];
        for _ in 0..100_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let magnitude = 2f64.powi((state >> 58) as i32 - 6);
            let v = (state as f64 / u64::MAX as f64 - 0.5) * magnitude;
            samples.push(v);
            samples.push(v.trunc() + 0.5);
            samples.push(v.trunc() - 0.5);
            samples.push(v * SCALE);
        }
        for v in samples {
            assert_eq!(round_i64(v), v.round() as i64, "round {v:e}");
            assert_eq!(floor_i64(v), v.floor() as i64, "floor {v:e}");
        }
    }

    #[test]
    fn atan2_is_the_reference_approximation() {
        let one = Fix::ONE;
        let half = Fix::from_f64(0.5);
        let approx = half.atan2(one).to_f64();
        // 0.28 is the game's truncated (Fix)0.28m, 1 raw unit below 0.28.
        assert!((approx - 0.5 / (1.0 + 0.28 * 0.25)).abs() < 1e-9);
        assert_eq!(Fix::ZERO.atan2(Fix::ZERO), Fix::ZERO);
        assert_eq!(one.atan2(Fix::ZERO), Fix::HALF_PI);
        assert_eq!(one.atan2(Fix::from_raw(1)), Fix::HALF_PI);
    }
}
