//! Decoding a BOPLREP2 archive (the format `RolloutWorld::encode_replay`
//! writes and the game's ReplayArchive reads) back into a reset config and
//! per-tick actions, so a recording can be re-stepped in this sim. The header
//! has no warmup or episode limit; callers supply those.
use super::{CAction, CConfig, CPlayerConfig};

pub struct Replay {
    pub config: CConfig,
    /// One row per simulated tick from boot, warmup included.
    pub ticks: Vec<Vec<CAction>>,
}

pub fn decode(raw: &[u8], warmup_ticks: u32, max_episode_ticks: u32) -> Result<Replay, String> {
    if !raw.starts_with(b"BOPLREP2") {
        return Err("not a BOPLREP2 archive".into());
    }
    let mut at = 136;
    let byte = |at: &mut usize| -> Result<u8, String> {
        let value = *raw.get(*at).ok_or("truncated archive")?;
        *at += 1;
        Ok(value)
    };
    let word = |at: &mut usize, width: usize| -> Result<u32, String> {
        let bytes = raw.get(*at..*at + width).ok_or("truncated archive")?;
        *at += width;
        let mut value = 0u32;
        for (i, b) in bytes.iter().enumerate() {
            value |= (*b as u32) << (8 * i);
        }
        Ok(value)
    };
    let seed = word(&mut at, 4)?;
    let map = byte(&mut at)? as u32;
    let mode = byte(&mut at)? as u32;
    let stocks = word(&mut at, 4)?;
    let players = byte(&mut at)? as usize;
    if !(1..=6).contains(&players) {
        return Err("invalid player count".into());
    }
    let mut config = CConfig {
        seed,
        map,
        players: players as u32,
        stocks,
        mode,
        warmup_ticks,
        max_episode_ticks,
        ..CConfig::default()
    };
    for player in 0..players {
        let _id = byte(&mut at)?;
        let color = byte(&mut at)? as u32;
        let team = byte(&mut at)? as u32;
        let keyboard = byte(&mut at)? as u32;
        let slots = byte(&mut at)? as usize;
        if !(1..=3).contains(&slots) {
            return Err("unsupported slot count".into());
        }
        let mut abilities = [0u32; 3];
        for ability in abilities.iter_mut().take(slots) {
            *ability = word(&mut at, 2)?;
        }
        config.slots = slots as u32;
        config.player[player] = CPlayerConfig { color, team, keyboard, abilities };
    }
    let count = word(&mut at, 4)? as usize;
    let mut ticks = Vec::with_capacity(count);
    for _ in 0..count {
        let mut actions = Vec::with_capacity(players);
        for _ in 0..players {
            let _tick = word(&mut at, 4)?;
            let angle = byte(&mut at)?;
            let buttons = word(&mut at, 2)? as u16;
            let _delay = byte(&mut at)?;
            actions.push(CAction { angle, reserved: 0, buttons });
        }
        ticks.push(actions);
    }
    Ok(Replay { config, ticks })
}
