//! Player-level death accounting from Player.Kill, after a body death path has
//! been selected. Body lifecycle and elimination listeners are separate stages.
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Cause {
    Other = 0,
    Squished = 1,
    Accident = 2,
    Eaten = 3,
    Beamed = 4,
    NotDeadYet = 5,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Player {
    pub id: i32,
    pub team: i32,
    pub bodies_alive: i32,
    pub won_round: bool,
    pub kills: i32,
    pub deaths: i32,
    pub still_alive_round: bool,
    pub killed_at: i64,
    pub cause: i32,
    pub rumble: i32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Death {
    pub player: i32,
    pub cause: i32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Book {
    pub players: Vec<Player>,
    pub registered_kills: [i32; 6],
}
impl Book {
    /// Source callback order is represented by the returned final-death event.
    /// The caller dispatches it after the accounting mutations have completed.
    pub fn kill(
        &mut self,
        victim: usize,
        killer: i32,
        timestamp: i64,
        cause: i32,
    ) -> Option<Death> {
        let player = &mut self.players[victim];
        if player.won_round {
            return None;
        }
        player.rumble = 2;
        let final_body = player.bodies_alive == 1;
        let id = player.id;
        player.deaths = player.deaths.wrapping_add(1);
        player.bodies_alive = player.bodies_alive.wrapping_sub(1);
        if let Some(killer) = self.players.iter_mut().find(|p| p.id == killer) {
            killer.kills = killer.kills.wrapping_add(1);
            killer.rumble = 1;
        }
        if !final_body {
            return None;
        }
        if killer != id && (1..=6).contains(&killer) {
            let count = &mut self.registered_kills[(killer - 1) as usize];
            *count = count.wrapping_add(1);
        }
        let player = &mut self.players[victim];
        player.still_alive_round = false;
        player.killed_at = timestamp;
        player.cause = cause;
        Some(Death { player: id, cause })
    }
}
