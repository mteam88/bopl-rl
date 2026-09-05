//! AnimationCurveFixed's fixed-point Hermite interpolation and Unity wrap modes.
use crate::fixed::Fix;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Key {
    pub time: Fix,
    pub value: Fix,
    pub in_tangent: Fix,
    pub out_tangent: Fix,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Curve {
    pub keys: Vec<Key>,
    pub pre_wrap: i32,
    pub post_wrap: i32,
}
fn modulo(a: Fix, b: Fix) -> Fix {
    a.checked_rem(b, true)
        .expect("curve modulo division by zero")
}
impl Curve {
    pub fn evaluate(&self, mut t: Fix) -> Fix {
        if self.keys.len() >= 2 {
            let first = self.keys[0];
            let last = *self.keys.last().unwrap();
            let span = last.time - first.time;
            if t <= first.time {
                match self.pre_wrap {
                    2 => {
                        t -= first.time;
                        t = span - modulo(t.slow_abs(), span);
                        t += first.time;
                    }
                    4 => {
                        t -= first.time;
                        let cycles = (t / span).slow_abs().to_int() as i32;
                        t = span - modulo(t.slow_abs(), span);
                        t = modulo(t, span);
                        if cycles % 2 == 0 {
                            t = span - t;
                        }
                        t += first.time;
                    }
                    _ => return first.value,
                }
            } else if t >= last.time {
                match self.post_wrap {
                    2 => {
                        t -= first.time;
                        t = modulo(t, span);
                        t += first.time;
                    }
                    4 => {
                        t -= first.time;
                        let cycles = (t / span).slow_abs().to_int() as i32;
                        t = modulo(t, span);
                        if cycles % 2 != 0 {
                            t = span - t;
                        }
                        t += first.time;
                    }
                    _ => return last.value,
                }
            }
        }
        for pair in self.keys.windows(2) {
            if t <= pair[1].time {
                return interpolate(t, pair[0], pair[1]);
            }
        }
        // The original returns zero for both empty and one-key curves.
        Fix::ZERO
    }
}
fn interpolate(t: Fix, a: Key, b: Key) -> Fix {
    let span = b.time - a.time;
    let t = (t - a.time) / span;
    let out_tangent = a.out_tangent * span;
    let in_tangent = b.in_tangent * span;
    let square = t * t;
    let cube = square * t;
    let two = Fix::from_int(2);
    let three = Fix::from_int(3);
    let h1 = two * cube - three * square + Fix::ONE;
    let h2 = cube - two * square + t;
    let h3 = cube - square;
    let h4 = -two * cube + three * square;
    h1 * a.value + h2 * out_tangent + h3 * in_tangent + h4 * b.value
}
