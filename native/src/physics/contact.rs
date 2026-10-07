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
    let mut values = points.iter().map(|point| axis.dot(*point));
    let first = values.next().expect("collision projection is non-empty");
    values.fold([first, first], |[minimum, maximum], value| {
        [minimum.min(value), maximum.max(value)]
    })
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
/// `Ord::min(acc, value)` as `projection`'s fold takes it: the accumulated
/// value on a tie (so a signed zero stays where the fold leaves it) and on a
/// NaN `value`.
#[inline(always)]
fn fold_min(acc: Fix, value: Fix) -> Fix {
    std::hint::select_unpredictable(acc > value, value, acc)
}
/// `Ord::max(acc, value)`: the new value on a tie or a NaN.
#[inline(always)]
fn fold_max(acc: Fix, value: Fix) -> Fix {
    std::hint::select_unpredictable(acc > value, acc, value)
}
#[inline(always)]
fn dot(ax: Fix, ay: Fix, p: Vec2) -> Fix {
    ax * p.x + ay * p.y
}
/// `projection` of two points, given their dot products in order.
#[inline(always)]
fn span2(d0: Fix, d1: Fix) -> (Fix, Fix) {
    (fold_min(d0, d1), fold_max(d0, d1))
}
/// `projection` of four points, given their dot products in order.
#[inline(always)]
fn span4(d0: Fix, d1: Fix, d2: Fix, d3: Fix) -> (Fix, Fix) {
    (
        fold_min(fold_min(fold_min(d0, d1), d2), d3),
        fold_max(fold_max(fold_max(d0, d1), d2), d3),
    )
}
/// `sat`'s forward and backward overlaps of `inflate`d projections.
#[inline(always)]
fn overlaps(pa: (Fix, Fix), ra: Fix, pb: (Fix, Fix), rb: Fix) -> (Fix, Fix) {
    ((pa.1 + ra) - (pb.0 - rb), (pb.1 + rb) - (pa.0 - ra))
}
/// `collision_face` of corners `c` along `direction` whose corner scores
/// (`direction.dot(c[i])`) are `s`.
#[inline(always)]
fn collision_face_scored(c: [Vec2; 4], s: [Fix; 4], direction: Vec2, inverse: Vec2) -> ([Vec2; 2], Vec2) {
    // The original's `find` panics when no score is >= all (a NaN).
    #[allow(clippy::eq_op)]
    let nan = (s[0] != s[0]) | (s[1] != s[1]) | (s[2] != s[2]) | (s[3] != s[3]);
    if nan {
        collision_face_nan();
    }
    // First maximum: the pairwise firsts, the later pair only if strictly
    // greater, as the in-order search picks without a NaN.
    let low = s[1] > s[0];
    let high = s[3] > s[2];
    let low_score = if low { s[1] } else { s[0] };
    let high_score = if high { s[3] } else { s[2] };
    let upper = high_score > low_score;
    // index = 2 * upper + (upper ? high : low); vertical on bit 1, horizontal on bit 0.
    let bottom = if upper { high } else { low };
    let (v0, v1) = if upper { (c[3], c[2]) } else { (c[0], c[1]) };
    let (h0, h1) = if bottom { (c[1], c[3]) } else { (c[2], c[0]) };
    let half = Fix::from_f32(0.5);
    let (ix, iy) = (inverse.x * half, inverse.y * half);
    let vnormal = (v1 - v0).perp() * iy;
    let hnormal = (h1 - h0).perp() * ix;
    if vnormal.dot(direction).abs() > hnormal.dot(direction).abs() {
        ([v0, v1], vnormal)
    } else {
        ([h0, h1], hnormal)
    }
}
/// Keeps the enclosing `if` a branch rather than a select: the choice is
/// usually predictable, and a predicted branch lets the work after it start
/// before the comparison resolves (an empty block LLVM cannot speculate).
#[inline(always)]
fn branch() {
    // SAFETY: no instructions; only an optimization barrier for if-conversion.
    unsafe { std::arch::asm!("", options(nomem, nostack, preserves_flags)) }
}
#[cold]
#[inline(never)]
fn collision_face_nan() -> ! {
    panic!("called `Option::unwrap()` on a `None` value")
}
/// PhysTools' RoundedRect/RoundedRect SAT, bit for bit as
/// `rounded_rounded_reference` computes it, with the same arithmetic in the
/// same order:
/// - any failing axis returns None, so the four face axes are tested
///   most-often-failing first (up of `a`, up of `b`, right of `a`, right of
///   `b`) and the contact takes their overlaps in the original order after;
/// - projections, `collision_face` and `contact_point` are inlined on scalars.
pub fn rounded_rounded(a: RoundedRect, b: RoundedRect) -> Option<Contact> {
    let result = rounded_rounded_fast(a, b);
    #[cfg(debug_assertions)]
    {
        let bits = |c: &Option<Contact>| {
            c.map(|c| [c.penetration, c.normal.x, c.normal.y, c.point.x, c.point.y].map(|v| v.bits()))
        };
        debug_assert_eq!(bits(&result), bits(&rounded_rounded_reference(a, b)));
    }
    result
}
#[inline(always)]
fn rounded_rounded_fast(a: RoundedRect, b: RoundedRect) -> Option<Contact> {
    let ca = corners(a.inner);
    let cb = corners(b.inner);
    let (ra, rb) = (a.radius, b.radius);
    let axes = [
        a.inner.right * a.inner.inverse_extents.x,
        a.inner.up * a.inner.inverse_extents.y,
        b.inner.right * b.inner.inverse_extents.x,
        b.inner.up * b.inner.inverse_extents.y,
    ];
    let failed = |o: (Fix, Fix)| o.0 < Fix::ZERO || o.1 < Fix::ZERO;
    let (x, y) = (axes[1].x, axes[1].y);
    let o1 = overlaps(
        span2(dot(x, y, ca[0]), dot(x, y, ca[1])),
        ra,
        span4(dot(x, y, cb[0]), dot(x, y, cb[1]), dot(x, y, cb[2]), dot(x, y, cb[3])),
        rb,
    );
    if failed(o1) {
        return None;
    }
    let (x, y) = (axes[3].x, axes[3].y);
    let o3 = overlaps(
        span4(dot(x, y, ca[0]), dot(x, y, ca[1]), dot(x, y, ca[2]), dot(x, y, ca[3])),
        ra,
        span2(dot(x, y, cb[0]), dot(x, y, cb[1])),
        rb,
    );
    if failed(o3) {
        return None;
    }
    let (x, y) = (axes[0].x, axes[0].y);
    let o0 = overlaps(
        span2(dot(x, y, ca[0]), dot(x, y, ca[2])),
        ra,
        span4(dot(x, y, cb[0]), dot(x, y, cb[1]), dot(x, y, cb[2]), dot(x, y, cb[3])),
        rb,
    );
    if failed(o0) {
        return None;
    }
    let (x, y) = (axes[2].x, axes[2].y);
    let o2 = overlaps(
        span4(dot(x, y, ca[0]), dot(x, y, ca[1]), dot(x, y, ca[2]), dot(x, y, ca[3])),
        ra,
        span2(dot(x, y, cb[0]), dot(x, y, cb[2])),
        rb,
    );
    if failed(o2) {
        return None;
    }
    // `sat`'s updates in axis order, from MAX along axes[0].
    let mut penetration = Fix::MAX;
    let mut normal = axes[0];
    for (o, axis) in [(o0, axes[0]), (o1, axes[1]), (o2, axes[2]), (o3, axes[3])] {
        if o.0 < penetration && o.0 < o.1 {
            branch();
            penetration = o.0;
            normal = axis;
        }
        if o.1 < penetration {
            branch();
            penetration = o.1;
            normal = -axis;
        }
    }
    let (x, y) = (normal.x, normal.y);
    let sa = [dot(x, y, ca[0]), dot(x, y, ca[1]), dot(x, y, ca[2]), dot(x, y, ca[3])];
    let (x, y) = (-normal.x, -normal.y);
    let sb = [dot(x, y, cb[0]), dot(x, y, cb[1]), dot(x, y, cb[2]), dot(x, y, cb[3])];
    let (mut fa, na) = collision_face_scored(ca, sa, normal, a.inner.inverse_extents);
    let (mut fb, nb) = collision_face_scored(cb, sb, -normal, b.inner.inverse_extents);
    // The first strictly least squared length below MAX, as a tournament.
    let deltas = [fa[0] - fb[0], fa[0] - fb[1], fa[1] - fb[1], fa[1] - fb[0]];
    let squared = deltas.map(|d| {
        let s = d.length_squared();
        if s < Fix::MAX { s } else { Fix::MAX }
    });
    let pick = |l: (Fix, usize), r: (Fix, usize)| if r.0 < l.0 { r } else { l };
    let (mut smallest, closest) = pick(pick((squared[0], 0), (squared[1], 1)), pick((squared[2], 2), (squared[3], 3)));
    let mut nearest = deltas[closest];
    if !(smallest < Fix::MAX) {
        smallest = Fix::MAX;
        nearest = Vec2::ZERO;
    }
    // Identical corners divide by zero in the source; do not silently normalize safely.
    let axis = nearest / smallest.sqrt().unwrap();
    let (x, y) = (axis.x, axis.y);
    let (forward, backward) = overlaps(
        span4(dot(x, y, ca[0]), dot(x, y, ca[1]), dot(x, y, ca[2]), dot(x, y, ca[3])),
        ra,
        span4(dot(x, y, cb[0]), dot(x, y, cb[1]), dot(x, y, cb[2]), dot(x, y, cb[3])),
        rb,
    );
    if forward < Fix::ZERO || backward < Fix::ZERO {
        return None;
    }
    if forward < penetration && forward < backward {
        branch();
        penetration = forward;
        normal = axis;
    }
    if backward < penetration {
        branch();
        penetration = backward;
        normal = -axis;
    }
    for point in &mut fa {
        *point += normal * a.radius;
    }
    for point in &mut fb {
        *point -= normal * b.radius;
    }
    Some(Contact {
        penetration,
        normal,
        point: contact_point_inline(fa, fb, na, nb, normal),
    })
}
#[inline(always)]
fn clip_inline(reference: [Vec2; 2], mut incident: [Vec2; 2], normal: Vec2, penetration: Vec2) -> Vec2 {
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
#[inline(always)]
fn contact_point_inline(a: [Vec2; 2], b: [Vec2; 2], normal_a: Vec2, normal_b: Vec2, normal: Vec2) -> Vec2 {
    if normal_a.dot(normal).abs() > normal_b.dot(normal).abs() {
        clip_inline(a, b, normal_a, -normal)
    } else {
        clip_inline(b, a, normal_b, normal)
    }
}
/// The source's RoundedRect/RoundedRect SAT, step for step: what
/// `rounded_rounded` reproduces bit for bit.
#[cfg(any(debug_assertions, test))]
pub fn rounded_rounded_reference(a: RoundedRect, b: RoundedRect) -> Option<Contact> {
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
/// Boolean Circle/Box test uses a strict radius boundary, unlike the manifold overload.
pub fn circle_box_overlaps(c: Circle, b: OrientedBox) -> bool {
    let (_, edges, inside) = circle_edges(c, b);
    inside
        || edges
            .iter()
            .map(|&p| (p - c.center).length_squared())
            .min()
            .unwrap()
            < c.radius * c.radius
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
