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
//! What is kept exact (see RUST_ROLLOUTS.md, "Number type"):
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
const fn round_i64(v: f64) -> i64 {
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
    pub const fn raw(self) -> i64 {
        round_i64(self.0 * SCALE)
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
        Ok(Self(self.0 % rhs.0))
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
            angle %= tau;
            if angle < 0.0 {
                angle += tau;
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
            (Self::HALF_PI.0 - angle).sin()
        } else {
            angle.sin()
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
        let value = (index * (std::f64::consts::FRAC_PI_2 / LUT_LAST)).sin();
        let value = (value * 4_294_967_296.0).floor() / 4_294_967_296.0;
        Self(if vertical { -value } else { value })
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
