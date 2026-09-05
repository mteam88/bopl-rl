//! Player.SetPosition arbitration between SlimeController identities. Ability
//! count determines priority; an equal-priority previous winner can reclaim it.
use crate::vector::Vec2;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct State {
    pub value: Vec2,
    pub current: Option<usize>,
    pub previous: Option<usize>,
}
impl State {
    pub fn begin_frame(&mut self) {
        self.previous = self.current;
        self.current = None;
    }
    pub fn set(&mut self, value: Vec2, candidate: Option<usize>, priorities: &[Option<i32>]) {
        let Some(candidate) = candidate else { return };
        let Some(priority) = priorities.get(candidate).copied().flatten() else {
            return;
        };
        let comparison = self
            .current
            .and_then(|i| priorities.get(i).copied().flatten())
            .map_or(1, |other| priority.wrapping_sub(other));
        if comparison > 0 || (comparison == 0 && self.previous == Some(candidate)) {
            self.current = Some(candidate);
            self.value = value;
        }
    }
}
