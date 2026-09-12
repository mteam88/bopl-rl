use crate::fixed::Fix;
use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct Vec2 {
    pub x: Fix,
    pub y: Fix,
}

impl Vec2 {
    pub const ZERO: Self = Self::new(Fix::ZERO, Fix::ZERO);
    pub const RIGHT: Self = Self::new(Fix::ONE, Fix::ZERO);
    pub const UP: Self = Self::new(Fix::ZERO, Fix::ONE);
    pub const fn new(x: Fix, y: Fix) -> Self {
        Self { x, y }
    }
    pub fn angle(angle: Fix) -> Self {
        Self::new(angle.fast_cos(), angle.fast_sin())
    }
    pub fn dot(self, other: Self) -> Fix {
        self.x * other.x + self.y * other.y
    }
    pub fn cross(self, other: Self) -> Fix {
        self.x * other.y - self.y * other.x
    }
    pub fn length_squared(self) -> Fix {
        self.dot(self)
    }
    pub fn length(self) -> Fix {
        self.length_squared()
            .sqrt()
            .expect("negative squared length")
    }
    pub fn normalized(self) -> Self {
        self / self.length()
    }
    pub fn normalized_safe(self) -> Self {
        let length = self.length();
        if length == Fix::ZERO {
            self
        } else {
            self / length
        }
    }
    pub fn perp(self) -> Self {
        Self::new(-self.y, self.x)
    }
    pub fn perp_cw(self) -> Self {
        Self::new(self.y, -self.x)
    }
    pub fn complex_mul(self, other: Self) -> Self {
        Self::new(
            self.x * other.x - self.y * other.y,
            self.y * other.x + self.x * other.y,
        )
    }
    pub fn normalized_angle(self) -> Fix {
        // The reference's `x % (One + 1 raw)` only ever sees a normalized
        // component, for which the remainder is the identity; keep the
        // general remainder for anything larger.
        let modulus = Fix::from_raw(Fix::ONE.raw() + 1);
        let x = if self.x.abs() < modulus {
            self.x
        } else {
            self.x.checked_rem(modulus, true).unwrap()
        };
        let angle = x.acos().expect("normalized vector outside cosine domain");
        if self.y > Fix::ZERO {
            angle
        } else {
            Fix::TAU - angle
        }
    }
}
impl Add for Vec2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}
impl Sub for Vec2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}
impl Neg for Vec2 {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y)
    }
}
impl Mul<Fix> for Vec2 {
    type Output = Self;
    fn mul(self, rhs: Fix) -> Self {
        Self::new(rhs * self.x, rhs * self.y)
    }
}
impl Mul<Vec2> for Fix {
    type Output = Vec2;
    fn mul(self, rhs: Vec2) -> Vec2 {
        rhs * self
    }
}
impl Div<Fix> for Vec2 {
    type Output = Self;
    fn div(self, rhs: Fix) -> Self {
        Self::new(self.x / rhs, self.y / rhs)
    }
}
impl AddAssign for Vec2 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl SubAssign for Vec2 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl MulAssign<Fix> for Vec2 {
    fn mul_assign(&mut self, rhs: Fix) {
        *self = *self * rhs;
    }
}
