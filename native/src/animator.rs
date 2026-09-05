//! SpriteAnimator's simulation clock and callback state machine. Sprite IDs are
//! opaque asset identities; immutable frame data is stored by value.
use crate::{fixed::Fix, vector::Vec2};
use serde::{Deserialize, Serialize};
pub type Result<T> = std::result::Result<T, &'static str>;
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Frame {
    pub sprite: Option<u32>,
    pub duration: i32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VelocityFrame {
    pub sprite: Option<u32>,
    pub from: Fix,
    pub to: Fix,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Track {
    pub frames: Option<Vec<Frame>>,
    pub duration: i32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Speed {
    pub velocity: Vec2,
    pub rotation: Fix,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Render {
    pub sprite: Option<u32>,
    pub flip_x: bool,
    pub flip_y: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct State {
    pub render: Option<Render>,
    pub current: Option<Vec<Frame>>,
    pub next: Option<Vec<Frame>>,
    pub index: i32,
    pub position: i32,
    pub kind: i32,
    pub swapped: bool,
    pub callback: Option<u32>,
    pub body: Option<usize>,
    pub velocity_maps: Option<Vec<Option<Vec<VelocityFrame>>>>,
    pub tracks: Option<Vec<Track>>,
    pub position_y: i32,
    pub going_right: bool,
    pub target: i32,
    pub x_speeds: Option<Vec<Fix>>,
    pub min_x: Fix,
    pub swap_timing: i32,
    pub callback_last: bool,
    pub always_stop: bool,
    pub position_2d: i32,
    pub animation_index: i32,
    pub child_suffix: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Start {
    /// 0=beginAnimation, 1=thenPlay, 2=beginAnim(callback), 3=thenDoAction.
    Begin {
        frames: Option<Vec<Frame>>,
        next: Option<Vec<Frame>>,
        callback: Option<u32>,
        style: i32,
    },
    Y {
        map: Option<Vec<VelocityFrame>>,
        looping: Option<Vec<Frame>>,
        anticipation: Option<u32>,
        body: Option<usize>,
    },
    Xy {
        maps: Option<Vec<Option<Vec<VelocityFrame>>>>,
        speeds: Option<Vec<Fix>>,
        min_x: Fix,
        timing: i32,
        anticipation: Option<u32>,
        body: Option<usize>,
        flip: bool,
        immediate: bool,
    },
    TwoD {
        tracks: Option<Vec<Track>>,
        start_x: i32,
        start_y: i32,
        right: bool,
        target: i32,
        override_sprite: bool,
        anticipation: Option<u32>,
    },
    Rotation {
        frames: Option<Vec<Frame>>,
        body: Option<usize>,
    },
    Skip {
        index: i32,
    },
    Target {
        index: i32,
    },
    Child {
        suffix: Option<String>,
    },
}
fn at<T>(items: &[T], index: i32) -> Result<&T> {
    items.get(index as usize).ok_or("index_out_of_range")
}
fn array<T>(items: &Option<Vec<T>>) -> Result<&[T]> {
    items.as_deref().ok_or("null_reference")
}
fn abs(value: i32) -> Result<i32> {
    value.checked_abs().ok_or("overflow")
}
impl State {
    /// Serialized defaults after SpriteAnimator.Awake on a new prefab.
    pub fn idle(sprite: Option<u32>) -> Self {
        Self {
            render: Some(Render {
                sprite,
                flip_x: false,
                flip_y: false,
            }),
            current: None,
            next: None,
            index: 0,
            position: 0,
            kind: 0,
            swapped: false,
            callback: None,
            body: None,
            velocity_maps: None,
            tracks: None,
            position_y: 0,
            going_right: false,
            target: 0,
            x_speeds: None,
            min_x: Fix::ZERO,
            swap_timing: 0,
            callback_last: false,
            always_stop: false,
            position_2d: 0,
            animation_index: 0,
            child_suffix: Some(String::new()),
        }
    }
    fn renderer(&mut self) -> Result<&mut Render> {
        self.render.as_mut().ok_or("null_reference")
    }
    fn speed<'a>(&self, bodies: &'a [Speed]) -> Result<&'a Speed> {
        self.body
            .and_then(|i| bodies.get(i))
            .ok_or("null_reference")
    }
    fn frame(&self) -> Result<&Frame> {
        at(array(&self.current)?, self.index)
    }
    fn track(&self) -> Result<&Track> {
        at(array(&self.tracks)?, self.animation_index)
    }
    fn track_frame(&self) -> Result<&Frame> {
        at(array(&self.track()?.frames)?, self.index)
    }
    fn invoke(
        &mut self,
        clear: bool,
        notify: &mut impl FnMut(&mut Self, u32) -> Result<()>,
    ) -> Result<()> {
        if let Some(id) = self.callback {
            if clear {
                self.callback = None;
            }
            notify(self, id)?;
        }
        Ok(())
    }
    pub fn update(
        &mut self,
        scale: Fix,
        protected: Option<bool>,
        bodies: &[Speed],
        notify: &mut impl FnMut(&mut Self, u32) -> Result<()>,
    ) -> Result<()> {
        if scale < Fix::from_f32(0.01) && (self.always_stop || protected != Some(true)) {
            return Ok(());
        }
        self.animate(bodies, notify)
    }
    pub fn animate(
        &mut self,
        bodies: &[Speed],
        notify: &mut impl FnMut(&mut Self, u32) -> Result<()>,
    ) -> Result<()> {
        match self.kind {
            0 => self.looped(notify),
            2 => self.speed_y(bodies, notify),
            3 => self.speed_xy(bodies, notify),
            4 => self.rotation(bodies),
            5 => self.two_d(),
            _ => Ok(()),
        }
    }
    fn first(
        &mut self,
        bodies: &[Speed],
        notify: &mut impl FnMut(&mut Self, u32) -> Result<()>,
    ) -> Result<()> {
        match self.kind {
            0 => {
                let frame = at(array(&self.current)?, 0)?.clone();
                self.renderer()?.sprite = frame.sprite;
                self.renderer()?.flip_x = frame.duration < 0;
                Ok(())
            }
            2 => self.speed_y(bodies, notify),
            4 => self.rotation(bodies),
            5 => {
                let sprite = self.track_frame()?.sprite;
                self.renderer()?.sprite = sprite;
                Ok(())
            }
            _ => Ok(()),
        }
    }
    fn looped(&mut self, notify: &mut impl FnMut(&mut Self, u32) -> Result<()>) -> Result<()> {
        if self.current.is_none() {
            return Ok(());
        }
        self.position = self.position.wrapping_add(1);
        self.swapped = false;
        let mut wrapped = false;
        let mut i = self.index;
        while i < (array(&self.current)?.len() as i32) {
            let duration = abs(self.frame()?.duration)?;
            if self.position <= duration {
                break;
            }
            self.position = self.position.wrapping_sub(duration);
            self.index = self.index.wrapping_add(1);
            if self.index >= array(&self.current)?.len() as i32 {
                wrapped = true;
                self.index = 0;
                if self.next.is_some() {
                    self.current = self.next.take();
                    self.swapped = true;
                }
            }
            i = i.wrapping_add(1);
        }
        if !(self.callback_last && wrapped) {
            let sprite = self.frame()?.sprite;
            self.renderer()?.sprite = sprite;
            let flip = self.frame()?.duration < 0;
            self.renderer()?.flip_x = flip;
        }
        if wrapped {
            self.invoke(true, notify)?;
        }
        Ok(())
    }
    fn rotation(&mut self, bodies: &[Speed]) -> Result<()> {
        let rotation = (self.speed(bodies)?.rotation + Fix::from_int(45))
            .checked_rem(Fix::from_int(360), false)
            .map_err(|_| "division_by_zero")?;
        let frames = array(&self.current)?;
        let mut selected = 0i32;
        let mut total = Fix::ZERO;
        let mut i = 0;
        while i < frames.len() {
            total += Fix::from_int(i64::from(frames[i].duration)).abs();
            if total > rotation {
                selected = i as i32 - 1;
                if selected < 0 {
                    selected = frames.len() as i32 - 1;
                }
                break;
            }
            i += 1;
        }
        if i == frames.len() {
            selected = frames.len() as i32 - 1;
        }
        let frame = at(frames, selected)?.clone();
        self.renderer()?.sprite = frame.sprite;
        self.renderer()?.flip_y = frame.duration < 0;
        Ok(())
    }
    fn speed_y(
        &mut self,
        bodies: &[Speed],
        notify: &mut impl FnMut(&mut Self, u32) -> Result<()>,
    ) -> Result<()> {
        self.animation_index = 0;
        let y = self.speed(bodies)?.velocity.y;
        let frames = array(at(array(&self.velocity_maps)?, self.animation_index)?)?;
        let mut selected = 0;
        let mut i = 0;
        while i < frames.len() {
            if y >= frames[i].from && y <= frames[i].to {
                selected = i;
                break;
            }
            i += 1;
        }
        if y > at(frames, 0)?.to {
            selected = 0;
        } else if i >= frames.len() {
            return self.looped(notify);
        }
        if selected as i32 >= frames.len() as i32 - 1 {
            let sprite = at(frames, frames.len() as i32 - 1)?.sprite;
            self.renderer()?.sprite = sprite;
            self.invoke(false, notify)?;
        } else {
            let sprite = frames[selected].sprite;
            self.renderer()?.sprite = sprite;
        }
        Ok(())
    }
    fn speed_xy(
        &mut self,
        bodies: &[Speed],
        notify: &mut impl FnMut(&mut Self, u32) -> Result<()>,
    ) -> Result<()> {
        self.position_2d = self.position_2d.wrapping_add(1);
        let swap = self.position_2d >= self.swap_timing;
        let velocity = self.speed(bodies)?.velocity;
        if swap {
            let mut lower = self.min_x;
            for (i, upper) in array(&self.x_speeds)?.iter().enumerate() {
                if velocity.x >= lower && velocity.x <= *upper {
                    self.animation_index = i as i32;
                    break;
                }
                lower = *upper;
            }
            self.animation_index = self
                .animation_index
                .min(array(&self.velocity_maps)?.len() as i32 - 1)
                .max(0);
            self.position_2d = 0;
        }
        let frames = array(at(array(&self.velocity_maps)?, self.animation_index)?)?;
        for (i, f) in frames.iter().enumerate() {
            if velocity.y >= f.from && velocity.y <= f.to {
                self.index = i as i32;
                break;
            }
        }
        if self.index >= frames.len() as i32 - 1 {
            let sprite = at(frames, frames.len() as i32 - 1)?.sprite;
            self.renderer()?.sprite = sprite;
            self.invoke(false, notify)?;
        } else {
            self.index = self.index.min(frames.len() as i32 - 1).max(0);
            let sprite = at(frames, self.index)?.sprite;
            self.renderer()?.sprite = sprite;
        }
        Ok(())
    }
    fn two_d(&mut self) -> Result<()> {
        if self.tracks.is_none() {
            return Ok(());
        }
        if self.animation_index < self.target {
            self.position_y = self.position_y.wrapping_add(1);
            let mut i = self.animation_index;
            while i < self.target {
                let duration = abs(self.track()?.duration)?;
                if self.position_y <= duration {
                    break;
                }
                self.position_y = self.position_y.wrapping_sub(duration);
                self.animation_index = self.animation_index.wrapping_add(1);
                if self.animation_index >= array(&self.tracks)?.len() as i32 {
                    self.animation_index = array(&self.tracks)?.len() as i32 - 1;
                }
                i = i.wrapping_add(1);
            }
        } else if self.animation_index > self.target {
            self.position_y = self.position_y.wrapping_add(1);
            while self.animation_index > self.target {
                let duration = abs(self.track()?.duration)?;
                if self.position_y <= duration {
                    break;
                }
                self.position_y = self.position_y.wrapping_sub(duration);
                self.animation_index = self.animation_index.wrapping_sub(1);
                if self.animation_index >= array(&self.tracks)?.len() as i32 {
                    self.animation_index = array(&self.tracks)?.len() as i32 - 1;
                }
            }
        }
        if self.index != array(&self.track()?.frames)?.len() as i32 - 1 {
            self.position = self.position.wrapping_add(1);
        }
        let mut j = self.index;
        while j < array(&self.track()?.frames)?.len() as i32 {
            let duration = abs(self.track_frame()?.duration)?;
            if self.position <= duration {
                break;
            }
            self.position = self.position.wrapping_sub(duration);
            self.index = self.index.wrapping_add(1);
            if self.index >= array(&self.track()?.frames)?.len() as i32 {
                self.index = array(&self.track()?.frames)?.len() as i32 - 1;
            }
            j = j.wrapping_add(1);
        }
        let flip = self.track()?.duration < 0 || self.track_frame()?.duration < 0;
        self.renderer()?.flip_x = flip;
        let sprite = self.track_frame()?.sprite;
        self.renderer()?.sprite = sprite;
        Ok(())
    }
    pub fn start(
        &mut self,
        start: Start,
        bodies: &[Speed],
        notify: &mut impl FnMut(&mut Self, u32) -> Result<()>,
    ) -> Result<()> {
        match start {
            Start::Begin {
                frames,
                next,
                callback,
                style,
            } => {
                self.kind = 0;
                self.position = 0;
                self.index = 0;
                self.current = frames;
                self.next = if style == 1 { next } else { None };
                if style != 1 {
                    self.callback = None;
                    self.callback_last = false;
                }
                self.renderer()?.flip_x = false;
                if style == 1 {
                    self.callback = callback;
                }
                self.first(bodies, notify)?;
                if style == 2 || style == 3 {
                    if style == 3 {
                        self.callback_last = true;
                    }
                    self.callback = callback;
                }
            }
            Start::Y {
                map,
                looping,
                anticipation,
                body,
            } => {
                self.velocity_maps = Some(vec![map]);
                self.position = 0;
                self.index = 0;
                self.current = looping;
                self.kind = 2;
                self.body = body;
                self.renderer()?.flip_x = false;
                self.next = None;
                self.callback = None;
                if anticipation.is_none() {
                    self.first(bodies, notify)?;
                } else {
                    self.renderer()?.sprite = anticipation;
                }
            }
            Start::Xy {
                maps,
                speeds,
                min_x,
                timing,
                anticipation,
                body,
                flip,
                immediate,
            } => {
                self.animation_index = 0;
                self.velocity_maps = maps;
                self.position = 0;
                self.index = 0;
                self.current = None;
                self.kind = 3;
                self.body = body;
                self.callback = None;
                self.swap_timing = timing;
                if !immediate {
                    self.renderer()?.sprite = anticipation;
                    self.renderer()?.flip_x = flip;
                }
                self.min_x = min_x;
                self.x_speeds = speeds;
                if immediate {
                    self.renderer()?.flip_x = flip;
                    self.speed_xy(bodies, notify)?;
                }
            }
            Start::TwoD {
                tracks,
                start_x,
                start_y,
                right,
                target,
                override_sprite,
                anticipation,
            } => {
                self.tracks = tracks;
                self.going_right = right;
                self.target = target;
                self.index = start_x;
                self.animation_index = start_y;
                self.current = None;
                self.kind = 5;
                self.next = None;
                self.callback = None;
                self.callback_last = false;
                self.renderer()?.flip_x = false;
                self.position = 0;
                self.position_y = 0;
                self.first(bodies, notify)?;
                if override_sprite {
                    self.renderer()?.sprite = anticipation;
                }
            }
            Start::Rotation { frames, body } => {
                self.position = 0;
                self.index = 0;
                self.current = frames;
                self.next = None;
                self.kind = 4;
                self.body = body;
                self.renderer()?.flip_x = false;
                self.callback = None;
                self.first(bodies, notify)?;
            }
            Start::Skip { index } => {
                self.index = index;
                self.position = 0;
            }
            Start::Target { index } => self.target = index,
            Start::Child { suffix } => self.child_suffix = suffix,
        }
        Ok(())
    }
    pub fn current_sprite(&self) -> Result<Option<u32>> {
        Ok(self.frame()?.sprite)
    }
    pub fn left_of_midpoint(&self) -> Result<bool> {
        Ok(self.animation_index >= array(&self.velocity_maps)?.len() as i32 / 2)
    }
}

/// SlimeController.SpeedUpAnimation, used by ReSpawnFast before its callback.
pub fn speed_up(frames: &[Frame], factor: i32) -> Result<Vec<Frame>> {
    frames
        .iter()
        .map(|frame| {
            let duration = abs(frame.duration)?
                .checked_div(factor)
                .ok_or("division_by_zero")?
                .max(1);
            Ok(Frame {
                sprite: frame.sprite,
                duration: if frame.duration < 0 {
                    -duration
                } else {
                    duration
                },
            })
        })
        .collect()
}
