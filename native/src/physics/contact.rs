//! Exact manifold construction, including the source's axis order and tie rules.
use super::Contact;
use crate::{
    fixed::Fix,
    geometry::{Circle, OrientedBox, RoundedRect},
    vector::Vec2,
};

pub fn circle_circle(a: Circle, b: Circle) -> Option<Contact> {
    let radius = a.radius + b.radius;
    let delta = b.center - a.center;
    let squared = delta.length_squared();
    if squared > radius * radius {
        return None;
    }
    let distance = squared.sqrt().unwrap();
    if distance == Fix::ZERO {
        Some(Contact {
            normal: Vec2::RIGHT,
            penetration: radius,
            point: a.center,
        })
    } else {
        let normal = delta / distance;
        Some(Contact {
            normal,
            penetration: radius - distance,
            point: a.center + normal * a.radius,
        })
    }
}

fn corners(b: OrientedBox) -> [Vec2; 4] {
    [
        b.center + b.right + b.up,
        b.center + b.right - b.up,
        b.center - b.right + b.up,
        b.center - b.right - b.up,
    ]
}
fn projection(points: &[Vec2], axis: Vec2) -> [Fix; 2] {
    let values: Vec<_> = points.iter().map(|p| axis.dot(*p)).collect();
    [*values.iter().min().unwrap(), *values.iter().max().unwrap()]
}
fn sat(a: [Fix; 2], b: [Fix; 2], axis: Vec2, inverse: Option<Fix>, contact: &mut Contact) -> bool {
    let mut forward = a[1] - b[0];
    let mut backward = b[1] - a[0];
    if forward < Fix::ZERO || backward < Fix::ZERO {
        return false;
    }
    let direction = if let Some(inverse) = inverse {
        forward *= inverse;
        backward *= inverse;
        axis * inverse
    } else {
        axis
    };
    if forward < contact.penetration && (inverse.is_some() || forward < backward) {
        contact.penetration = forward;
        contact.normal = direction;
    }
    if backward < contact.penetration {
        contact.penetration = backward;
        contact.normal = if let Some(inverse) = inverse {
            -axis * inverse
        } else {
            -direction
        };
    }
    true
}
fn collision_face(c: [Vec2; 4], direction: Vec2, inverse: Vec2) -> ([Vec2; 2], Vec2) {
    let inverse = inverse * Fix::from_f32(0.5);
    let scores = c.map(|p| direction.dot(p));
    // First maximum wins: top-right, bottom-right, top-left, bottom-left.
    let index = (0..4)
        .find(|i| scores.iter().all(|s| scores[*i] >= *s))
        .unwrap();
    let (vertical, horizontal) = match index {
        0 => ([c[0], c[1]], [c[2], c[0]]),
        1 => ([c[0], c[1]], [c[1], c[3]]),
        2 => ([c[3], c[2]], [c[2], c[0]]),
        _ => ([c[3], c[2]], [c[1], c[3]]),
    };
    let vnormal = (vertical[1] - vertical[0]).perp() * inverse.y;
    let hnormal = (horizontal[1] - horizontal[0]).perp() * inverse.x;
    if vnormal.dot(direction).abs() > hnormal.dot(direction).abs() {
        (vertical, vnormal)
    } else {
        (horizontal, hnormal)
    }
}
fn clip(reference: [Vec2; 2], mut incident: [Vec2; 2], normal: Vec2, penetration: Vec2) -> Vec2 {
    let tangent = normal.perp_cw();
    let min = tangent.dot(reference[0]);
    let max = tangent.dot(reference[1]);
    for point in &mut incident {
        let distance = tangent.dot(*point);
        if distance < min {
            *point += (min - distance) * tangent;
        } else if distance > max {
            *point -= (distance - max) * tangent;
        }
    }
    if penetration.dot(incident[0]) > penetration.dot(incident[1]) {
        incident[0]
    } else {
        incident[1]
    }
}
fn contact_point(a: [Vec2; 2], b: [Vec2; 2], normal_a: Vec2, normal_b: Vec2, normal: Vec2) -> Vec2 {
    if normal_a.dot(normal).abs() > normal_b.dot(normal).abs() {
        clip(a, b, normal_a, -normal)
    } else {
        clip(b, a, normal_b, normal)
    }
}
pub fn box_box(a: OrientedBox, b: OrientedBox) -> Option<Contact> {
    let ca = corners(a);
    let cb = corners(b);
    let mut contact = Contact {
        penetration: Fix::MAX,
        normal: a.right,
        ..Contact::default()
    };
    let tests = [
        (
            projection(&[ca[0], ca[2]], a.right),
            projection(&cb, a.right),
            a.right,
            a.inverse_extents.x,
        ),
        (
            projection(&[ca[0], ca[1]], a.up),
            projection(&cb, a.up),
            a.up,
            a.inverse_extents.y,
        ),
        (
            projection(&ca, b.right),
            projection(&[cb[0], cb[2]], b.right),
            b.right,
            b.inverse_extents.x,
        ),
        (
            projection(&ca, b.up),
            projection(&[cb[0], cb[1]], b.up),
            b.up,
            b.inverse_extents.y,
        ),
    ];
    for (x, y, axis, inverse) in tests {
        if !sat(x, y, axis, Some(inverse), &mut contact) {
            return None;
        }
    }
    contact.penetration = contact.penetration.abs();
    let (fa, na) = collision_face(ca, contact.normal, a.inverse_extents);
    let (fb, nb) = collision_face(cb, -contact.normal, b.inverse_extents);
    contact.point = contact_point(fa, fb, na, nb, contact.normal);
    Some(contact)
}
pub fn rounded_rounded(a: RoundedRect, b: RoundedRect) -> Option<Contact> {
    let ca = corners(a.inner);
    let cb = corners(b.inner);
    let axes = [
        a.inner.right * a.inner.inverse_extents.x,
        a.inner.up * a.inner.inverse_extents.y,
        b.inner.right * b.inner.inverse_extents.x,
        b.inner.up * b.inner.inverse_extents.y,
    ];
    let mut contact = Contact {
        penetration: Fix::MAX,
        normal: axes[0],
        ..Contact::default()
    };
    let inflate = |p: [Fix; 2], r: Fix| [p[0] - r, p[1] + r];
    for (i, axis) in axes.into_iter().enumerate() {
        let pa = match i {
            0 => projection(&[ca[0], ca[2]], axis),
            1 => projection(&[ca[0], ca[1]], axis),
            _ => projection(&ca, axis),
        };
        let pb = match i {
            2 => projection(&[cb[0], cb[2]], axis),
            3 => projection(&[cb[0], cb[1]], axis),
            _ => projection(&cb, axis),
        };
        if !sat(
            inflate(pa, a.radius),
            inflate(pb, b.radius),
            axis,
            None,
            &mut contact,
        ) {
            return None;
        }
    }
    let (mut fa, na) = collision_face(ca, contact.normal, a.inner.inverse_extents);
    let (mut fb, nb) = collision_face(cb, -contact.normal, b.inner.inverse_extents);
    let mut smallest = Fix::MAX;
    let mut nearest = Vec2::ZERO;
    for delta in [fa[0] - fb[0], fa[0] - fb[1], fa[1] - fb[1], fa[1] - fb[0]] {
        let squared = delta.length_squared();
        if squared < smallest {
            smallest = squared;
            nearest = delta;
        }
    }
    // Identical corners divide by zero in the source; do not silently normalize safely.
    let axis = nearest / smallest.sqrt().unwrap();
    if !sat(
        inflate(projection(&ca, axis), a.radius),
        inflate(projection(&cb, axis), b.radius),
        axis,
        None,
        &mut contact,
    ) {
        return None;
    }
    for point in &mut fa {
        *point += contact.normal * a.radius;
    }
    for point in &mut fb {
        *point -= contact.normal * b.radius;
    }
    contact.point = contact_point(fa, fb, na, nb, contact.normal);
    Some(contact)
}
fn line_closest(a: Vec2, b: Vec2, point: Vec2, inverse: Fix) -> Vec2 {
    let mut line = b - a;
    let start = a.dot(line);
    let end = b.dot(line);
    let at = point.dot(line);
    if at > end {
        return b;
    }
    if at < start {
        return a;
    }
    line *= inverse;
    a + line * (at - start) * inverse
}
fn circle_edges(c: Circle, b: OrientedBox) -> ([Vec2; 4], [Vec2; 4], bool) {
    let corners = corners(b);
    let right = line_closest(
        corners[1],
        corners[0],
        c.center,
        Fix::from_f64(0.5) * b.inverse_extents.y,
    );
    let top = line_closest(
        corners[2],
        corners[0],
        c.center,
        Fix::from_f64(0.5) * b.inverse_extents.x,
    );
    let inside =
        right != corners[0] && right != corners[1] && top != corners[2] && top != corners[0];
    let edges = [right, right - b.right - b.right, top, top - b.up - b.up];
    (corners, edges, inside)
}
pub fn circle_box(c: Circle, b: OrientedBox) -> Option<Contact> {
    let (_, edges, inside) = circle_edges(c, b);
    let distances = edges.map(|p| (p - c.center).length_squared());
    let mut distance = distances[0];
    let mut normal = b.right * b.inverse_extents.x;
    if distances[1] < distance {
        distance = distances[1];
        normal = -normal;
    }
    if distances[2] < distance {
        distance = distances[2];
        normal = b.up * b.inverse_extents.y;
    }
    if distances[3] < distance {
        distance = distances[3];
        normal = -b.up * b.inverse_extents.y;
    }
    if !inside && distance > c.radius * c.radius {
        return None;
    }
    let distance = distance.sqrt().unwrap();
    Some(Contact {
        normal: -normal,
        penetration: if inside {
            distance + c.radius
        } else {
            c.radius - distance
        },
        point: c.center + (-normal) * c.radius,
    })
}
pub fn circle_rounded(mut c: Circle, r: RoundedRect) -> Option<Contact> {
    c.radius += r.radius;
    let (_, edges, inside) = circle_edges(c, r.inner);
    let mut delta = edges[0] - c.center;
    let mut distance = delta.length_squared();
    for point in &edges[1..] {
        let other = *point - c.center;
        let squared = other.length_squared();
        if squared < distance {
            delta = other;
            distance = squared;
        }
    }
    if !inside && distance > c.radius * c.radius {
        return None;
    }
    let (normal, penetration) = if distance == Fix::ZERO {
        let normal = (r.inner.center - c.center).normalized();
        (if inside { -normal } else { normal }, c.radius)
    } else {
        let distance = distance.sqrt().unwrap();
        if inside {
            (-delta / distance, distance + c.radius)
        } else {
            (delta / distance, c.radius - distance)
        }
    };
    Some(Contact {
        normal,
        penetration,
        point: c.center + normal * (c.radius - r.radius),
    })
}
