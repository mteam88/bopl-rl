//! Platform attachments and stable composite-slot reuse from DetPhysics.
use super::Body;
use super::scene::{Composite, Scene};
use crate::{fixed::Fix, vector::Vec2};

impl Scene {
    fn allocate_composite(&mut self) -> usize {
        if let Some(i) = self.composites.iter().position(|c| c.destroyed) {
            self.composites[i].destroyed = false;
            i
        } else {
            self.composites.push(Composite {
                body: Body::default(),
                position: Vec2::ZERO,
                destroyed: false,
                circles: Vec::new(),
                boxes: Vec::new(),
                rounded_rects: Vec::new(),
            });
            self.composites.len() - 1
        }
    }
    fn reset_composite(&mut self, c: usize) {
        let c = &mut self.composites[c];
        c.body = Body::default();
        c.position = Vec2::ZERO;
        c.destroyed = true;
        c.circles.clear();
        c.boxes.clear();
        c.rounded_rects.clear();
    }
    fn add_member(&mut self, child: usize, c: usize) {
        let id = self.objects[child].id;
        match self.objects[child].collider.kind() {
            0 => self.composites[c].circles.push(id),
            1 => self.composites[c].boxes.push(id),
            _ => self.composites[c].rounded_rects.push(id),
        }
        self.objects[child].composite = Some(c);
    }
    /// `root=None` means the big body is the combined body itself. Preserve that
    /// alias: clearing big.force then also clears combined.force in the source.
    fn combine(&mut self, child: usize, root: Option<usize>, c: usize, combined_has_parent: bool) {
        let small = self.objects[child].body;
        let (big, pos) = match root {
            Some(i) => (self.objects[i].body, self.objects[i].collider.center()),
            None => (self.composites[c].body, self.composites[c].position),
        };
        let mass_small = Fix::ONE / small.inverse_mass;
        let mass_big = Fix::ONE / big.inverse_mass;
        let inertia_small = Fix::ONE / small.inverse_inertia;
        let inertia_big = Fix::ONE / big.inverse_inertia;
        let inverse_mass = Fix::ONE / (mass_small + mass_big);
        let small_pos = self.objects[child].collider.center();
        self.objects[child].relative_orientation = small.orientation - big.orientation;
        self.objects[child].relative_pos =
            (small_pos - pos).complex_mul(Vec2::angle(-big.orientation));
        if let Some(i) = root {
            self.objects[i].relative_orientation = Fix::ZERO;
            self.objects[i].relative_pos = Vec2::ZERO;
        }
        self.composites[c].position = pos;
        self.composites[c].body = Body {
            inverse_mass,
            inverse_inertia: Fix::ONE / (inertia_small + inertia_big),
            force: if root.is_none() && !combined_has_parent {
                Vec2::ZERO
            } else {
                big.force
            },
            ..big
        };
    }
    fn attach_to(&mut self, child: usize, c: usize) {
        self.add_member(child, c);
        self.combine(child, None, c, false);
    }
    /// Public AddCircleToRR/AddBoxToRR behavior, including their different guards.
    pub fn attach(&mut self, child_id: u32, root_id: u32) {
        let child = self.index(child_id);
        let root = self.index(root_id);
        assert!(self.objects[root].collider.kind() == 2 && self.objects[child].collider.kind() < 2);
        if let Some(c) = self.composite(root) {
            self.attach_to(child, c);
        } else if self.composite(child).is_none() {
            let c = self.allocate_composite();
            self.add_member(child, c);
            self.add_member(root, c);
            self.combine(child, Some(root), c, false);
        }
    }
    fn create_multi(
        &mut self,
        circles: Vec<u32>,
        boxes: Vec<u32>,
        rounded: Vec<u32>,
    ) -> Option<usize> {
        match circles.len() + boxes.len() + rounded.len() {
            0 => None,
            1 => {
                let id = circles
                    .first()
                    .or(boxes.first())
                    .or(rounded.first())
                    .unwrap();
                let i = self.index(*id);
                self.objects[i].composite = None;
                None
            }
            _ => {
                let c = self.allocate_composite();
                let root = self.index(rounded[0]);
                let mut combined_has_parent = self.composite(root).is_some();
                self.composites[c].body = self.objects[root].body;
                self.composites[c].position = self.objects[root].collider.center();
                self.add_member(root, c);
                for id in circles.into_iter().chain(boxes) {
                    let i = self.index(id);
                    self.add_member(i, c);
                    self.combine(i, None, c, combined_has_parent);
                    combined_has_parent = false;
                }
                Some(c)
            }
        }
    }
    pub fn detach(&mut self, id: u32) {
        let i = self.index(id);
        let c = self.objects[i].composite.expect("detach without composite");
        let old = &mut self.composites[c];
        let members = match self.objects[i].collider.kind() {
            0 => &mut old.circles,
            1 => &mut old.boxes,
            _ => &mut old.rounded_rects,
        };
        if let Some(index) = members.iter().position(|&member| member == id) {
            members.remove(index);
        }
        let old = old.clone();
        if let Some(new) = self.create_multi(old.circles, old.boxes, old.rounded_rects) {
            self.composites[new].body.velocity = old.body.velocity;
            self.composites[new].body.angular_velocity = old.body.angular_velocity;
        }
        self.reset_composite(c);
        self.objects[i].composite = None;
    }
    pub fn destroy(&mut self, id: u32) {
        let i = self.index(id);
        if let Some(c) = self.composite(i) {
            if self.objects[i].collider.kind() == 2 {
                let comp = &self.composites[c];
                let children: Vec<_> = comp.boxes.iter().chain(&comp.circles).copied().collect();
                self.objects.retain(|o| !children.contains(&o.id));
                self.reset_composite(c);
            } else {
                self.synchronize_composite(c);
                self.detach(id);
            }
        }
        self.objects.retain(|o| o.id != id);
    }
}
