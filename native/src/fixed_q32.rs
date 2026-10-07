//! The reference game's Q32.32 arithmetic, including its distinct fast/slow
//! overflow rules (cargo feature `q32`; see `fixed.rs` for the default f64
//! representation).
//! Constants and lookup tables are copied as raw integers from the pinned C# source.
use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(transparent)]
#[serde(transparent)]
pub struct Fix(i64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MathError {
    DivisionByZero,
    Domain,
    Overflow,
}

const SIN: &[u8; 205887 * 8] = include_bytes!("../data/sinlut.bin");
const TAN: &[u8; 205887 * 8] = include_bytes!("../data/tanlut.bin");

fn lookup(table: &[u8], index: usize) -> Fix {
    Fix(i64::from_le_bytes(
        table[index * 8..index * 8 + 8].try_into().unwrap(),
    ))
}

impl Fix {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(1 << 32);
    pub const MIN: Self = Self(i64::MIN);
    pub const MAX: Self = Self(i64::MAX);
    pub const PI: Self = Self(13_493_037_704);
    pub const HALF_PI: Self = Self(6_746_518_852);
    pub const TAU: Self = Self(26_986_075_409);

    /// Construct from a raw Q32.32 integer (the ABI and extracted-data encoding).
    pub const fn from_raw(raw: i64) -> Self {
        Self(raw)
    }
    /// The raw Q32.32 integer (the ABI and extracted-data encoding).
    pub const fn raw(self) -> i64 {
        self.0
    }
    /// `raw` of each value, as the f64 type's packed conversion.
    pub fn raw_array<const N: usize>(values: [Self; N]) -> [i64; N] {
        values.map(Self::raw)
    }
    pub const fn from_int(value: i64) -> Self {
        Self(value.wrapping_shl(32))
    }
    pub const fn to_int(self) -> i64 {
        self.0 >> 32
    }
    /// `to_int().clamp(lo, hi)` (the f64 build clamps before the floor).
    pub fn floor_clamp(self, lo: i64, hi: i64) -> i64 {
        self.to_int().clamp(lo, hi)
    }
    pub fn from_f32(value: f32) -> Self {
        Self((value * 4_294_967_296.0_f32) as i64)
    }
    pub fn from_f64(value: f64) -> Self {
        Self((value * 4_294_967_296.0_f64) as i64)
    }
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / 4_294_967_296.0
    }
    /// The value's bits: equal exactly when the two are the same value.
    pub const fn bits(self) -> u64 {
        self.0 as u64
    }
    pub fn sign(self) -> i64 {
        self.0.signum()
    }
    pub fn sign2(self) -> i64 {
        if self.0 < 0 { -1 } else { 1 }
    }
    pub fn abs(self) -> Self {
        Self(self.0.wrapping_abs())
    }
    pub fn slow_abs(self) -> Self {
        Self(self.0.saturating_abs())
    }
    pub fn floor(self) -> Self {
        Self(self.0 & !0xffff_ffff)
    }
    pub fn ceil(self) -> Self {
        if self.0 & 0xffff_ffff == 0 {
            self
        } else {
            self.floor().slow_add(Self::ONE)
        }
    }
    pub fn round(self) -> Self {
        let fraction = self.0 & 0xffff_ffff;
        let floor = self.floor();
        if fraction < 0x8000_0000 || (fraction == 0x8000_0000 && floor.0 & (1 << 32) == 0) {
            floor
        } else {
            floor.slow_add(Self::ONE)
        }
    }
    pub fn slow_add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }
    pub fn slow_sub(self, rhs: Self) -> Self {
        Self(self.0.saturating_sub(rhs.0))
    }
    pub fn slow_mul(self, rhs: Self) -> Self {
        let a = self.0;
        let b = rhs.0;
        let lo_a = a & 0xffff_ffff;
        let hi_a = a >> 32;
        let lo_b = (b & 0xffff_ffff) as u64;
        let hi_b = b >> 32;
        let lo = (lo_a as u64).wrapping_mul(lo_b);
        let cross_a = lo_a.wrapping_mul(hi_b);
        let cross_b = hi_a.wrapping_mul(lo_b as i64);
        let hi = hi_a.wrapping_mul(hi_b);
        let mut overflow = false;
        let mut result = (lo >> 32) as i64;
        for term in [cross_a, cross_b, hi.wrapping_shl(32)] {
            let next = result.wrapping_add(term);
            overflow |= (result ^ term ^ next) < 0;
            result = next;
        }
        let same_sign = (a ^ b) >= 0;
        if same_sign {
            if result < 0 || (overflow && a > 0) {
                return Self::MAX;
            }
        } else if result > 0 {
            return Self::MIN;
        }
        let high_word = hi >> 32;
        if high_word != 0 && high_word != -1 {
            return if same_sign { Self::MAX } else { Self::MIN };
        }
        if !same_sign && result > a.min(b) && a.min(b) < -Self::ONE.0 && a.max(b) > Self::ONE.0 {
            return Self::MIN;
        }
        Self(result)
    }
    /// `self / rhs` for a divisor the caller knows is nonzero (the f64
    /// build drops the zero test; this is the checked division).
    #[inline]
    pub fn div_nonzero(self, rhs: Self) -> Self {
        self / rhs
    }
    pub fn checked_div(self, rhs: Self) -> Result<Self, MathError> {
        if rhs.0 == 0 {
            return Err(MathError::DivisionByZero);
        }
        // This is the same rounded `(abs(lhs) << 32) / abs(rhs)` computed by
        // FixedMath's restoring loop, expressed directly now that Rust gives
        // us a native u128 intermediate. Besides being simpler, it avoids the
        // loop's repeated 64-bit divisions in normalization-heavy physics.
        let numerator = u128::from(self.0.unsigned_abs()) << 32;
        let divisor = u128::from(rhs.0.unsigned_abs());
        let mut quotient = numerator / divisor;
        let remainder = numerator - quotient * divisor;
        if remainder * 2 >= divisor {
            quotient += 1;
        }
        let negative = (self.0 ^ rhs.0) < 0;
        let limit = if negative {
            1_u128 << 63
        } else {
            i64::MAX as u128
        };
        if quotient > limit {
            return Ok(if negative { Self::MIN } else { Self::MAX });
        }
        let result = quotient as i64;
        Ok(Self(if negative {
            result.wrapping_neg()
        } else {
            result
        }))
    }
    /// The orientation wrap `((self % TAU) + TAU) % TAU`.
    pub fn wrap_tau(self) -> Self {
        (self.checked_rem(Self::TAU, true).unwrap() + Self::TAU)
            .checked_rem(Self::TAU, true)
            .unwrap()
    }
    /// The f64 type's fused fmod; here, exactly `checked_rem(rhs, true)`.
    pub fn checked_rem_fused(self, rhs: Self) -> Result<Self, MathError> {
        self.checked_rem(rhs, true)
    }
    pub fn checked_rem(self, rhs: Self, slow: bool) -> Result<Self, MathError> {
        if rhs.0 == 0 {
            return Err(MathError::DivisionByZero);
        }
        if self.0 == i64::MIN && rhs.0 == -1 {
            return if slow {
                Ok(Self::ZERO)
            } else {
                Err(MathError::Overflow)
            };
        }
        Ok(Self(self.0 % rhs.0))
    }
    pub fn sqrt(self) -> Result<Self, MathError> {
        if self.0 < 0 {
            return Err(MathError::Domain);
        }
        // sqrt(raw / 2^32) in Q32 is sqrt(raw << 32). The primitive integer
        // square root produces the same floor as FixedMath's two restoring
        // passes; its remainder test below preserves FixedMath's final
        // nearest-integer rounding.
        let value = (self.0 as u128) << 32;
        // Q32 magnitudes fit below 2^48 after the root, where f64 still has
        // sub-integer precision. Use the hardware root as an estimate, then
        // correct against the integer radicand so the result stays exact.
        let mut result = ((self.0 as f64).sqrt() * 65_536.0) as u64;
        while u128::from(result + 1) * u128::from(result + 1) <= value {
            result += 1;
        }
        while u128::from(result) * u128::from(result) > value {
            result -= 1;
        }
        let remainder = value - u128::from(result) * u128::from(result);
        if remainder > u128::from(result) {
            result += 1;
        }
        Ok(Self(result as i64))
    }
    fn sin_quadrant(self) -> (i64, bool, bool) {
        let mut angle = self.0;
        // FixedMath only needs the overflow-safe descending remainders for
        // unusually large angles. Simulation transforms are already within
        // two turns, so avoid 29 signed divisions on every axis rebuild.
        // Keeping the original cascade in the slow path preserves every raw
        // Q32 result for replay/oracle parity.
        const CASCADE_FLOOR: i64 = 7_244_019_458_077_122_842_i64 >> 28;
        if angle >= CASCADE_FLOOR || angle <= -CASCADE_FLOOR {
            for i in 0..29 {
                angle %= 7_244_019_458_077_122_842_i64 >> i;
            }
        }
        if self.0 < 0 {
            angle += Self::TAU.0;
        }
        let vertical = angle >= Self::PI.0;
        while angle >= Self::PI.0 {
            angle -= Self::PI.0;
        }
        let horizontal = angle >= Self::HALF_PI.0;
        if horizontal {
            angle -= Self::HALF_PI.0;
        }
        (angle, horizontal, vertical)
    }
    pub fn fast_sin(self) -> Self {
        let (angle, horizontal, vertical) = self.sin_quadrant();
        let index = ((angle >> 15) as usize).min(205886);
        let raw = lookup(SIN, if horizontal { 205886 - index } else { index }).0;
        Self(if vertical { -raw } else { raw })
    }
    /// The table entry `fast_sin` reads, with its sign in bit 32:
    /// `fast_sin` is `fast_sin_at(fast_sin_entry())`.
    pub fn fast_sin_entry(self) -> u64 {
        let (angle, horizontal, vertical) = self.sin_quadrant();
        let index = ((angle >> 15) as usize).min(205886);
        (if horizontal { 205886 - index } else { index }) as u64 | (vertical as u64) << 32
    }
    /// `fast_cos`'s entry, as `fast_sin_entry`.
    pub fn fast_cos_entry(self) -> u64 {
        Self(self.0.wrapping_add(if self.0 > 0 { -20_239_556_556 } else { Self::HALF_PI.0 })).fast_sin_entry()
    }
    /// The value at a `fast_sin_entry`.
    pub fn fast_sin_at(entry: u64) -> Self {
        let raw = lookup(SIN, (entry & 0xffff_ffff) as usize).0;
        Self(if entry >> 32 != 0 { -raw } else { raw })
    }
    pub fn fast_cos(self) -> Self {
        Self(self.0.wrapping_add(if self.0 > 0 {
            -20_239_556_556
        } else {
            Self::HALF_PI.0
        }))
        .fast_sin()
    }
    pub fn sin(self) -> Self {
        let (angle, horizontal, vertical) = self.sin_quadrant();
        let scaled = Self(angle) * (Self::from_int(205886) / Self::HALF_PI);
        let rounded = scaled.round();
        let error = scaled - rounded;
        let index = rounded.to_int();
        let a = lookup(
            SIN,
            if horizontal { 205886 - index } else { index } as usize,
        );
        let b = lookup(
            SIN,
            if horizontal {
                205886 - index - error.sign()
            } else {
                index + error.sign()
            } as usize,
        );
        let offset = (error * (a - b).abs()).0;
        let raw = a.0.wrapping_add(if horizontal {
            offset.wrapping_neg()
        } else {
            offset
        });
        Self(if vertical { raw.wrapping_neg() } else { raw })
    }
    pub fn cos(self) -> Self {
        Self(self.0.wrapping_add(if self.0 > 0 {
            -20_239_556_556
        } else {
            Self::HALF_PI.0
        }))
        .sin()
    }
    pub fn tan(self) -> Self {
        let mut angle = self.0 % Self::PI.0;
        let mut negative = angle < 0;
        if negative {
            angle = -angle;
        }
        if angle > Self::HALF_PI.0 {
            negative = !negative;
            angle = Self::PI.0 - angle;
        }
        let scaled = Self(angle) * (Self::from_int(205886) / Self::HALF_PI);
        let rounded = scaled.round();
        let error = scaled - rounded;
        let a = lookup(TAN, rounded.to_int() as usize);
        let b = lookup(TAN, (rounded.to_int() + error.sign()) as usize);
        let raw = a.0.wrapping_add((error * (a - b).abs()).0);
        Self(if negative { raw.wrapping_neg() } else { raw })
    }
    pub fn atan(self) -> Self {
        if self == Self::ZERO {
            return self;
        }
        let negative = self.0 < 0;
        let mut z = if negative { -self } else { self };
        let inverted = z > Self::ONE;
        if inverted {
            z = Self::ONE / z;
        }
        let mut sum = Self::ONE;
        let mut term = Self::ONE;
        let square = z.slow_mul(z);
        let twice = square.slow_mul(Self::from_int(2));
        let base = square.slow_add(Self::ONE);
        let increment = base.slow_mul(Self::from_int(2));
        let mut numerator = twice;
        let mut denominator = base.slow_mul(Self::from_int(3));
        for _ in 2..30 {
            term = term.slow_mul(numerator / denominator);
            sum = sum.slow_add(term);
            numerator = numerator.slow_add(twice);
            denominator = denominator.slow_add(increment);
            if term == Self::ZERO {
                break;
            }
        }
        sum = sum.slow_mul(z) / base;
        if inverted {
            sum = Self::HALF_PI.slow_sub(sum);
        }
        if negative { -sum } else { sum }
    }
    pub fn acos(self) -> Result<Self, MathError> {
        if self < -Self::ONE || self > Self::ONE {
            return Err(MathError::Domain);
        }
        if self == Self::ZERO {
            return Ok(Self::HALF_PI);
        }
        let angle = (Self::ONE.slow_sub(self.slow_mul(self)).sqrt()? / self).atan();
        Ok(if self.0 >= 0 {
            angle
        } else {
            angle.slow_add(Self::PI)
        })
    }
    pub fn atan2(self, x: Self) -> Self {
        let y = self;
        if x.0 == 0 {
            return match y.0.cmp(&0) {
                std::cmp::Ordering::Less => -Self::HALF_PI,
                std::cmp::Ordering::Equal => Self::ZERO,
                std::cmp::Ordering::Greater => Self::HALF_PI,
            };
        }
        let ratio = y / x;
        let c = Self(1_202_590_842); // (Fix)0.28m
        let denominator = Self::ONE.slow_add(c.slow_mul(ratio.slow_mul(ratio)));
        if denominator == Self::MAX {
            return if y.0 < 0 {
                -Self::HALF_PI
            } else {
                Self::HALF_PI
            };
        }
        if ratio.slow_abs() < Self::ONE {
            let angle = ratio / denominator;
            if x.0 < 0 {
                if y.0 < 0 {
                    angle.slow_sub(Self::PI)
                } else {
                    angle.slow_add(Self::PI)
                }
            } else {
                angle
            }
        } else {
            // Preserve the reference's parentheses; this is not standard atan2.
            let angle = Self::HALF_PI.slow_sub(ratio) / ratio.slow_mul(ratio).slow_add(c);
            if y.0 < 0 {
                angle.slow_sub(Self::PI)
            } else {
                angle
            }
        }
    }
}

impl Add for Fix {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0.wrapping_add(rhs.0))
    }
}
impl Sub for Fix {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0.wrapping_sub(rhs.0))
    }
}
impl Mul for Fix {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self(((self.0 as i128 * rhs.0 as i128) >> 32) as i64)
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
        Self(if self.0 == i64::MIN {
            i64::MAX
        } else {
            -self.0
        })
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
