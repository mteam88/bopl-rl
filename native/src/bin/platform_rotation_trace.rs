//! Internal companion to `research/rust_ab/platform_rotation_ab.c`: exposes
//! the platform spring's private home rotation without extending the RL ABI.
use bopl_sim::rollout::{CAction, CConfig, CPlayerConfig, RolloutWorld};
use bopl_sim::{fixed::Fix, physics::contact::rounded_rounded, vector::Vec2};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        eprintln!("usage: platform_rotation_trace <map> <angle> <release_tick> <end_tick>");
        std::process::exit(2);
    }
    let map: u32 = args[1].parse().expect("map");
    let angle: u8 = args[2].parse().expect("angle");
    let release_tick: u32 = args[3].parse().expect("release tick");
    let end_tick: u32 = args[4].parse().expect("end tick");
    let jump_tick: u32 = std::env::var("PLATFORM_ROTATION_JUMP_TICK")
        .ok()
        .map(|value| value.parse().expect("jump tick"))
        .unwrap_or(0);
    let player = CPlayerConfig {
        color: 0,
        team: 0,
        keyboard: 0,
        abilities: [14, 2, 4],
    };
    let mut config = CConfig {
        seed: 7,
        map,
        players: 2,
        slots: 3,
        stocks: 10,
        mode: 0,
        warmup_ticks: 240,
        max_episode_ticks: 7200,
        player: [player; 6],
    };
    config.player[1].team = 1;
    let mut world = RolloutWorld::new();
    world.reset(&config).expect("reset");
    println!(
        "map,angle,release_tick,tick,phase,orientation,angular_velocity,home_rotation,width,height,contact_other,penetration,normal_x,normal_y,point_x,point_y,arm_cross_normal"
    );
    for tick in 1..=end_tick {
        let buttons =
            u16::from(tick == jump_tick) | (u16::from(tick >= 276 && tick < release_tick) * 2);
        world
            .step(
                &[
                    CAction {
                        angle,
                        reserved: 0,
                        buttons,
                    },
                    CAction::default(),
                ],
                1,
            )
            .expect("step");
        if let Some(spawn) = world.spawned.iter().rev().find(|spawn| spawn.active) {
            let index = spawn.platform_index;
            let body = &world.platform_dynamics[index];
            let surface = &world.platforms[index].surface;
            let contact = world
                .platforms
                .iter()
                .enumerate()
                .filter(|(other, _)| *other != index)
                .find_map(|(other, platform)| {
                    rounded_rounded(platform.collider, world.platforms[index].collider)
                        .map(|contact| (other, contact))
                });
            let (other, penetration, normal, point, arm_cross_normal) = contact
                .map(|(other, contact)| {
                    let arm = (contact.point - surface.center).normalized_safe();
                    (
                        other as i64,
                        contact.penetration,
                        contact.normal,
                        contact.point,
                        arm.cross(contact.normal),
                    )
                })
                .unwrap_or((-1, Fix::ZERO, Vec2::ZERO, Vec2::ZERO, Fix::ZERO));
            println!(
                "{map},{angle},{release_tick},{tick},{},{:.12},{:.12},{:.12},{:.12},{:.12},{other},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
                world.players[0].platform.phase,
                body.body.orientation.to_f64(),
                body.body.angular_velocity.to_f64(),
                body.home_rotation.to_f64(),
                surface.extents.x.to_f64(),
                surface.extents.y.to_f64(),
                penetration.to_f64(),
                normal.x.to_f64(),
                normal.y.to_f64(),
                point.x.to_f64(),
                point.y.to_f64(),
                arm_cross_normal.to_f64(),
            );
        }
    }
}
