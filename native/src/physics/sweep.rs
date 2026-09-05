//! Stable sweep ordering, including inactive bodies in the variance calculation.
use super::scene::{Scene, group};
use crate::{fixed::Fix, vector::Vec2};
type Pairs = [Vec<[usize; 2]>; 6];
pub(super) fn pairs(scene: &Scene) -> (bool, bool, Pairs) {
    let mut mean = Vec2::ZERO;
    let mut active = Vec::new();
    for kind in [2, 0, 1] {
        for (i, o) in scene.objects.iter().enumerate() {
            if o.collider.kind() == kind && o.active {
                active.push(i);
                mean += o.collider.center();
            }
        }
    }
    mean = mean / Fix::from_int(active.len().max(1) as i64);
    let mut variance = Vec2::ZERO;
    let mut rounded_sweep_x = false;
    for kind in [2, 0, 1] {
        for o in &scene.objects {
            if o.collider.kind() != kind {
                continue;
            }
            let delta = mean - o.collider.center();
            variance += Vec2::new(delta.x * delta.x, delta.y * delta.y);
        }
        if kind == 2 {
            rounded_sweep_x = variance.x > variance.y;
        }
    }
    let sweep_x = variance.x > variance.y;
    let axis = |v: Vec2| if sweep_x { v.x } else { v.y };
    let other = |v: Vec2| if sweep_x { v.y } else { v.x };
    let bounds: Vec<_> = scene.objects.iter().map(|o| o.collider.bounds()).collect();
    active.sort_by(|&a, &b| {
        (axis(bounds[a][0]) - axis(bounds[b][0]))
            .sign()
            .cmp(&0)
            .then_with(|| {
                scene.objects[a]
                    .collider
                    .kind()
                    .cmp(&scene.objects[b].collider.kind())
            })
            .then(a.cmp(&b))
    });
    let mut groups: Pairs = std::array::from_fn(|_| Vec::new());
    let mut window = Vec::new();
    for b in active {
        window.retain(|&a: &usize| axis(bounds[a][1]) > axis(bounds[b][0]));
        for &a in &window {
            let oa = &scene.objects[a];
            let ob = &scene.objects[b];
            if scene.layers[oa.layer] & (1 << ob.layer) == 0
                || other(bounds[a][1]) <= other(bounds[b][0])
                || other(bounds[a][0]) >= other(bounds[b][1])
            {
                continue;
            }
            let (a, b) = if oa.collider.kind() > ob.collider.kind() {
                (b, a)
            } else {
                (a, b)
            };
            groups[group(
                scene.objects[a].collider.kind(),
                scene.objects[b].collider.kind(),
            )]
            .push([a, b]);
        }
        window.push(b);
    }
    (sweep_x, rounded_sweep_x, groups)
}
