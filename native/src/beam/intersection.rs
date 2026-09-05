//! Beam intersections preserve the source's strict endpoints and tie rules.
use crate::{fixed::Fix, vector::Vec2};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Ray {
    pub position: Vec2,
    pub direction: Vec2,
    pub distance: Fix,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ReflectedRay {
    #[serde(flatten)]
    pub ray: Ray,
    pub reflected: bool,
    pub was_reflected: bool,
    pub reflection_distance: Fix,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Intersection {
    pub hit: bool,
    pub distances: Vec2,
}

pub fn lines(a: Ray, b: Ray) -> Intersection {
    let position = b.position - a.position;
    let normal = b.direction.perp();
    let projection = position.dot(normal) * normal;
    let denominator = projection.dot(a.direction);
    if denominator == Fix::ZERO {
        return Intersection {
            hit: false,
            distances: Vec2::ZERO,
        };
    }
    let x = projection.dot(projection) / denominator;
    let y = (a.direction * x - position).dot(b.direction);
    Intersection {
        hit: x > Fix::ZERO && x < a.distance && y > Fix::ZERO && y < b.distance,
        distances: Vec2::new(x, y),
    }
}
fn reflect(ray: Ray, distance: Fix) -> Ray {
    Ray {
        position: ray.position + ray.direction * ray.distance,
        direction: Vec2::new(ray.direction.x, -ray.direction.y),
        distance,
    }
}
pub fn beams(a: ReflectedRay, b: ReflectedRay, minimize_b: bool) -> Intersection {
    let ar = Ray {
        distance: a.ray.distance
            - if a.reflected {
                a.reflection_distance
            } else {
                Fix::ZERO
            },
        ..a.ray
    };
    let br = Ray {
        distance: b.ray.distance
            - if b.reflected {
                b.reflection_distance
            } else {
                Fix::ZERO
            },
        ..b.ray
    };
    let along_a =
        !minimize_b && ((b.was_reflected && b.reflected) || !(a.was_reflected && a.reflected));
    let mut best = Fix::MAX;
    let mut result = Intersection {
        hit: false,
        distances: Vec2::ZERO,
    };
    let direct = lines(ar, br);
    if direct.hit {
        best = if along_a {
            direct.distances.x
        } else {
            direct.distances.y
        };
        result = direct;
    }
    if a.reflected {
        let mut hit = lines(reflect(ar, a.reflection_distance), br);
        if hit.hit {
            hit.distances.x += ar.distance;
            result.hit = true;
            let at = if along_a {
                hit.distances.x
            } else {
                hit.distances.y
            };
            if at < best {
                best = at;
                result.distances = hit.distances;
            }
        }
    }
    if b.reflected {
        let mut hit = lines(ar, reflect(br, b.reflection_distance));
        if hit.hit {
            hit.distances.y += br.distance;
            result.hit = true;
            // Both Unity methods compare the original direct intersection's y
            // here, including its output when that direct intersection missed.
            let comparison = if along_a {
                hit.distances.x
            } else {
                direct.distances.y
            };
            if comparison < best {
                best = if along_a {
                    hit.distances.x
                } else {
                    hit.distances.y
                };
                result.distances = hit.distances;
            }
        }
    }
    if a.reflected && b.reflected {
        let mut hit = lines(
            reflect(ar, a.reflection_distance),
            reflect(br, b.reflection_distance),
        );
        if hit.hit {
            hit.distances.x += ar.distance;
            hit.distances.y += br.distance;
            result.hit = true;
            let at = if along_a {
                hit.distances.x
            } else {
                hit.distances.y
            };
            if at < best {
                result.distances = hit.distances;
            }
        }
    }
    result
}
