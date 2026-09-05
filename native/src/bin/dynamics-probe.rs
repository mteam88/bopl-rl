use bopl_sim::{
    fixed::Fix,
    geometry::RoundedRect,
    physics::{self, contact},
    vector::Vec2,
};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use std::io::{self, BufRead, BufWriter, Write};
fn read<T: DeserializeOwned>(v: &Value, k: &str) -> T {
    serde_json::from_value(v[k].clone()).unwrap()
}
fn animator_start(
    state: &mut bopl_sim::animator::State,
    command: &Value,
    bodies: &[bopl_sim::animator::Speed],
    callbacks: &Value,
    events: &mut Vec<Value>,
) -> bopl_sim::animator::Result<()> {
    state.start(
        serde_json::from_value(command.clone()).unwrap(),
        bodies,
        &mut |state, id| animator_callback(state, id, bodies, callbacks, events),
    )
}
fn animator_callback(
    state: &mut bopl_sim::animator::State,
    id: u32,
    bodies: &[bopl_sim::animator::Speed],
    callbacks: &Value,
    events: &mut Vec<Value>,
) -> bopl_sim::animator::Result<()> {
    events.push(json!({"id":id,"state":state}));
    for command in callbacks[id.to_string()].as_array().unwrap() {
        animator_start(state, command, bodies, callbacks, events)?;
    }
    Ok(())
}
fn evaluate_spawn(q: &Value) -> Value {
    use bopl_sim::{
        animator,
        spawn::{self, Context},
        surface::Surface,
    };
    let mut state: spawn::State = read(q, "spawn");
    let mut animator: animator::State = read(q, "state");
    let mut bodies: Vec<animator::Speed> = read(q, "bodies");
    let mut players: Vec<spawn::Player> = read(q, "players");
    let surfaces: Vec<Surface> = read(q, "surfaces");
    let config: spawn::Config = read(q, "config");
    let offset: Vec2 = read(&q["config"], "detach_offset");
    let velocity: Vec2 = read(&q["config"], "detach_velocity");
    let mut on_detach = |state: &mut spawn::State, _angle| {
        state.motion.position += offset;
        state.motion.velocity += velocity;
        Ok(())
    };
    let holder: Option<i32> = read(q, "holder");
    let mut frames = Vec::new();
    for a in q["actions"].as_array().unwrap() {
        let mut events = Vec::new();
        let op = a["op"].as_str().unwrap();
        let result = if op == "animate" || op == "update" {
            // All animator speed reads precede the callback. The callback may
            // update the shared speed container and begin the fall animation.
            let speeds_before = bodies.clone();
            let protected = holder
                .and_then(|id| players.iter().find(|p| p.id == id))
                .map(|p| p.protected);
            let mut callback = |animator: &mut animator::State, id| {
                assert_eq!(id, 1);
                state.spawn(&mut Context {
                    animator,
                    speeds: &mut bodies,
                    players: &mut players,
                    surfaces: &surfaces,
                    config: &config,
                    events: &mut events,
                    on_detach: &mut on_detach,
                })
            };
            if op == "animate" {
                animator.animate(&speeds_before, &mut callback)
            } else {
                animator.update(read(a, "scale"), protected, &speeds_before, &mut callback)
            }
        } else {
            let mut c = Context {
                animator: &mut animator,
                speeds: &mut bodies,
                players: &mut players,
                surfaces: &surfaces,
                config: &config,
                events: &mut events,
                on_detach: &mut on_detach,
            };
            match op {
                "init" => state.init(&mut c),
                "before" => state.before(&mut c),
                "prepare" => state.prepare(read(a, "position"), &mut c),
                "respawn" => state.respawn(read(a, "position"), &mut c),
                "fast" => state.fast_respawn(read(a, "position"), &mut c),
                "spawn" => state.spawn(&mut c),
                "pop" => {
                    state.pop(read(a, "direction"), config.jump_strength);
                    Ok(())
                }
                _ => panic!("Unknown spawn action"),
            }
        };
        let frame = json!({"spawn":state,"animator":animator,"bodies":bodies,"players":players,"events":events});
        if let Err(error) = result {
            return json!({"error":error,"action":frames.len(),"completed":frames,"failed":frame});
        }
        frames.push(frame);
    }
    json!(frames)
}
fn evaluate(q: Value) -> Value {
    let contact = match q["operation"].as_str().unwrap() {
        "dynamics_spawn" => return evaluate_spawn(&q),
        "dynamics_animator" => {
            let mut state: bopl_sim::animator::State = read(&q, "state");
            let mut bodies: Vec<bopl_sim::animator::Speed> = read(&q, "bodies");
            let holder: Option<i32> = read(&q, "holder");
            let protected = holder
                .and_then(|id| {
                    q["players"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|p| p["id"].as_i64() == Some(i64::from(id)))
                })
                .map(|p| p["protected"].as_bool().unwrap());
            let mut frames = Vec::new();
            for a in q["actions"].as_array().unwrap() {
                let mut events = Vec::new();
                let mut callback = |state: &mut bopl_sim::animator::State, id| {
                    animator_callback(state, id, &bodies, &q["callbacks"], &mut events)
                };
                let result = match a["op"].as_str().unwrap() {
                    "speed_up" => {
                        let frames: Option<Vec<bopl_sim::animator::Frame>> = read(a, "frames");
                        frames
                            .ok_or("null_reference")
                            .and_then(|frames| {
                                bopl_sim::animator::speed_up(&frames, read(a, "factor"))
                            })
                            .map(|v| json!(v))
                    }
                    "animate" => state.animate(&bodies, &mut callback).map(|_| Value::Null),
                    "update" => state
                        .update(read(a, "scale"), protected, &bodies, &mut callback)
                        .map(|_| Value::Null),
                    "speed" => {
                        let index: usize = read(a, "index");
                        if let Some(body) = bodies.get_mut(index) {
                            body.velocity = read(a, "velocity");
                            body.rotation = read(a, "rotation");
                            Ok(Value::Null)
                        } else {
                            Err("index_out_of_range")
                        }
                    }
                    "current_sprite" => state.current_sprite().map(|s| json!(s)),
                    "passed" => Ok(json!(state.index.wrapping_add(1))),
                    "current_index" => Ok(json!(state.index)),
                    "current_animation" => Ok(json!(state.current)),
                    "left" => state.left_of_midpoint().map(|v| json!(v)),
                    _ => animator_start(&mut state, a, &bodies, &q["callbacks"], &mut events)
                        .map(|_| Value::Null),
                };
                let frame = json!({"state":state,"bodies":bodies,"events":events,"result":result.as_ref().cloned().unwrap_or(Value::Null)});
                if let Err(error) = result {
                    return json!({"error":error,"action":frames.len(),"completed":frames,"failed":frame});
                }
                frames.push(frame);
            }
            return json!(frames);
        }
        "dynamics_player_controller" => {
            use bopl_sim::player_controller::{self, Context, Event};
            let mut state: player_controller::State = read(&q, "state");
            let mut scene: physics::scene::Scene = read(&q, "scene");
            let bindings: Vec<bopl_sim::platform_motion::Binding> = read(&q, "bindings");
            let grounds: Vec<player_controller::Ground> = read(&q, "grounds");
            let config: player_controller::Config = read(&q, "config");
            let holder: Option<i32> = read(&q, "holder");
            let protected = holder
                .and_then(|id| {
                    q["players"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|p| p["id"].as_i64() == Some(i64::from(id)))
                })
                .map(|p| p["protected"].as_bool().unwrap());
            let mut hop_buffer: Vec<i32> = read(&q, "hop_buffer");
            let mut frames = Vec::new();
            for action in q["actions"].as_array().unwrap() {
                let mut context = Context {
                    scene: &mut scene,
                    bindings: &bindings,
                    grounds: &grounds,
                    config: &config,
                    holder,
                    protected,
                    time_scale: read(&q, "time_scale"),
                    space: read(&q, "space"),
                    ability: read(&q, "ability"),
                };
                let mut events = Vec::new();
                let mut notify = |s: &mut player_controller::State, event: Event| {
                    match &event {
                        Event::Attached(_) => {
                            s.physics.motion.position += read::<Vec2>(&q, "attach_offset")
                        }
                        Event::Detached(_) => {
                            s.physics.motion.position += read::<Vec2>(&q, "detach_offset");
                            s.physics.motion.velocity += read::<Vec2>(&q, "detach_velocity");
                        }
                        Event::Exit(_) => {
                            s.physics.motion.position += read::<Vec2>(&q, "exit_offset")
                        }
                        Event::JumpSound(_) => {}
                    }
                    events.push(event);
                };
                let result = match action["op"].as_str().unwrap() {
                    "update" => state
                        .update(read(action, "dt"), &mut context, &mut notify)
                        .map(|_| Value::Null),
                    "move" => state
                        .move_player(
                            read(action, "input"),
                            read(action, "dt"),
                            None,
                            read(action, "surface"),
                            &mut context,
                            &mut notify,
                        )
                        .map(|_| Value::Null),
                    "macho" => state
                        .move_player(
                            read(action, "input"),
                            read(action, "dt"),
                            Some(read(action, "multiplier")),
                            true,
                            &mut context,
                            &mut notify,
                        )
                        .map(|_| Value::Null),
                    "drop" => state
                        .drop_platform(&context, &mut notify)
                        .map(|_| Value::Null),
                    "align" => state.align(&context).map(|_| Value::Null),
                    "current" => state
                        .move_current(read(action, "dt"), &mut context)
                        .map(|_| Value::Null),
                    "hop" => state
                        .hop(
                            read(action, "radius"),
                            read(action, "speed"),
                            read(action, "dt"),
                            &mut hop_buffer,
                            &mut context,
                        )
                        .map(|_| Value::Null),
                    "alignment" => state
                        .input_alignment(read(action, "input"), &context)
                        .map(|v| json!(v)),
                    "increment" => Ok(json!(player_controller::increment(
                        read(action, "current"),
                        read(action, "target"),
                        read(action, "accel"),
                        read(action, "dt")
                    ))),
                    "accelerate" => Ok(json!(player_controller::accelerate(
                        read(action, "current"),
                        read(action, "target"),
                        read(action, "accel"),
                        context.dt(read(action, "dt"))
                    ))),
                    _ => panic!("Unknown player controller action"),
                };
                let orders: Vec<Vec<u32>> = [2, 0, 1]
                    .into_iter()
                    .map(|kind| {
                        scene
                            .objects
                            .iter()
                            .filter(|o| o.collider.kind() == kind)
                            .map(|o| o.id)
                            .collect()
                    })
                    .collect();
                let mut snapshot = scene.clone();
                snapshot.objects.sort_by_key(|o| o.id);
                let frame = json!({"state":state,"scene":snapshot,"orders":orders,"hop_buffer":hop_buffer,"events":events,"result":result.as_ref().cloned().unwrap_or(Value::Null)});
                if let Err(error) = result {
                    return json!({"error":error,"action":frames.len(),"completed":frames,"failed":frame});
                }
                frames.push(frame);
            }
            return json!(frames);
        }
        "dynamics_surface_motion" => {
            let mut scene: physics::scene::Scene = read(&q, "scene");
            let bindings: Vec<bopl_sim::platform_motion::Binding> = read(&q, "bindings");
            let mut frames = Vec::new();
            for action in q["actions"].as_array().unwrap() {
                let mut terrain = bopl_sim::platform_motion::Terrain {
                    scene: &mut scene,
                    bindings: &bindings,
                };
                let object = read(action, "object");
                let request: bopl_sim::platform_motion::Movement = read(action, "request");
                let result = match action["op"].as_str().unwrap() {
                    "move" => terrain.move_along(object, request, None).map(|v| json!(v)),
                    "multiple" => terrain
                        .move_along(object, request, Some(read(action, "second")))
                        .map(|v| json!(v)),
                    "hop" => terrain
                        .hop(object, request, &read::<Vec<i32>>(action, "disallowed"))
                        .map(|v| json!(v)),
                    "limit" => terrain
                        .max_movement(
                            object,
                            request.position,
                            request.radius,
                            read(action, "velocity"),
                            request.dt,
                        )
                        .map(|v| json!(v)),
                    _ => panic!("Unknown surface motion action"),
                };
                let orders: Vec<Vec<u32>> = [2, 0, 1]
                    .into_iter()
                    .map(|kind| {
                        scene
                            .objects
                            .iter()
                            .filter(|o| o.collider.kind() == kind)
                            .map(|o| o.id)
                            .collect()
                    })
                    .collect();
                let mut snapshot = scene.clone();
                snapshot.objects.sort_by_key(|o| o.id);
                let frame = json!({"scene": snapshot, "orders": orders, "result": result.as_ref().cloned().unwrap_or(Value::Null)});
                if let Err(error) = result {
                    return json!({"error":error,"action":frames.len(),"completed":frames,"failed":frame});
                }
                frames.push(frame);
            }
            return json!(frames);
        }
        "dynamics_player_rays" => {
            let mut state: bopl_sim::player_physics::State = read(&q, "state");
            let mut scene: bopl_sim::physics::scene::Scene = read(&q, "scene");
            let platforms: Vec<bopl_sim::player_physics::Platform> = read(&q, "platforms");
            let targets: Vec<bopl_sim::player_physics::GroundTarget> = read(&q, "targets");
            let holder: Option<i32> = read(&q, "holder");
            let protection = holder
                .and_then(|id| {
                    q["players"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|p| p["id"].as_i64() == Some(i64::from(id)))
                })
                .map(|p| p["protected"].as_bool().unwrap());
            let time_scale: bopl_sim::fixed::Fix = read(&q, "time_scale");
            let mut frames = Vec::new();
            for action in q["actions"].as_array().unwrap() {
                let mut events = Vec::new();
                let result = if action["op"] == "query" {
                    Ok(json!(scene.raycast_rounded(
                        read(action, "origin"),
                        read(action, "direction"),
                        read(action, "distance")
                    )))
                } else {
                    let velocity = if action["op"] == "velocity" {
                        state.motion.external_velocity + state.motion.velocity
                    } else {
                        read(action, "velocity")
                    };
                    let dt = bopl_sim::game_time::fixed_delta_time(
                        read(action, "dt"),
                        time_scale,
                        protection,
                    );
                    state
                        .collision_raycasts(
                            bopl_sim::player_physics::CollisionRaycast {
                                velocity,
                                effective_dt: dt,
                                attach: read(action, "attach"),
                            },
                            &mut scene,
                            &targets,
                            &platforms,
                            &mut |s, hit| {
                                events.push(hit);
                                s.motion.position += read::<Vec2>(&q, "event_offset");
                            },
                        )
                        .map(|v| json!(v))
                };
                let orders: Vec<Vec<u32>> = [2, 0, 1]
                    .into_iter()
                    .map(|kind| {
                        scene
                            .objects
                            .iter()
                            .filter(|o| o.collider.kind() == kind)
                            .map(|o| o.id)
                            .collect()
                    })
                    .collect();
                let mut snapshot = scene.clone();
                snapshot.objects.sort_by_key(|o| o.id);
                let value = result.as_ref().cloned().unwrap_or(Value::Null);
                let frame = json!({"state":state,"scene":snapshot,"orders":orders,"events":events,"result":value});
                if let Err(error) = result {
                    return json!({"error":error,"action":frames.len(),"completed":frames,"failed":frame});
                }
                frames.push(frame);
            }
            return json!(frames);
        }
        "dynamics_player_physics" => {
            let mut state: bopl_sim::player_physics::State = read(&q, "state");
            let config: bopl_sim::player_physics::Config = read(&q, "config");
            let platforms: Vec<bopl_sim::player_physics::Platform> = read(&q, "platforms");
            let mut rope: Option<bopl_sim::player_body::BodyRope> = read(&q, "rope");
            let event_offset: Vec2 = read(&q, "event_offset");
            let mut frames = Vec::new();
            for action in q["actions"].as_array().unwrap() {
                let mut events = Vec::new();
                let mut notify = |s: &mut bopl_sim::player_physics::State, angle| {
                    events.push(angle);
                    s.motion.position += event_offset;
                };
                let result = match action["op"].as_str().unwrap() {
                    "scale" => {
                        state.scale = read(action, "scale");
                        state.update_scale()
                    }
                    "gravity" => state.gravity(&config, read(&q, "space")),
                    "air" => state.air_move(read(action, "input"), &config),
                    "force" => {
                        state.add_force(read(action, "force"), &config);
                        Ok(())
                    }
                    "jump" => state
                        .jump(&platforms, &config)
                        .map(|angle| notify(&mut state, angle)),
                    "align" => state.align(&platforms),
                    "offset" => state.offset(read(action, "offset"), &platforms),
                    "sync" => state.sync(&read(action, "info"), read(action, "exit"), &platforms),
                    "late" => state.late_update(
                        &platforms,
                        rope.as_mut().map(|r| (r, read(&q, "top"))),
                        &config,
                        &mut notify,
                    ),
                    _ => panic!("unknown player physics action"),
                };
                let frame = json!({"state":state,"rope":rope,"events":events});
                if let Err(error) = result {
                    return json!({"error":error,"action":frames.len(),"completed":frames,"failed":frame});
                }
                frames.push(frame);
            }
            return json!(frames);
        }
        "dynamics_player_body" => {
            let mut state: bopl_sim::player_body::State = read(&q, "state");
            let context: bopl_sim::player_body::Context = read(&q, "context");
            let mut ropes: Vec<bopl_sim::player_body::BodyRope> = read(&q, "ropes");
            let mut frames = Vec::new();
            for action in q["actions"].as_array().unwrap() {
                let mut released = false;
                let result = match action["op"].as_str().unwrap() {
                    "init" => {
                        state.init();
                        Ok(())
                    }
                    "force" => {
                        state.add_force(read(action, "force"));
                        Ok(())
                    }
                    "attach" => {
                        state.attach(
                            read(action, "rope"),
                            read(action, "top"),
                            &mut ropes,
                            &context,
                        );
                        Ok(())
                    }
                    "clear" => {
                        state.clear_for_ability();
                        Ok(())
                    }
                    "update" => state
                        .update(read(action, "dt"), &mut ropes, &context)
                        .map(|value| released = value),
                    "late" => state.late_update(&mut ropes, &context),
                    "rope_step" => {
                        let mut result = Ok(());
                        for rope in &mut ropes {
                            result = rope
                                .physics
                                .try_step(
                                    bopl_sim::physics::rope::RopeConfig {
                                        dt: read(action, "dt"),
                                        gravity: context.rope_gravity,
                                        space: false,
                                        iterations: 3,
                                        stretchiness: bopl_sim::fixed::Fix::from_f64(0.1),
                                    },
                                    &[],
                                )
                                .map_err(|_| "division_by_zero");
                            if result.is_err() {
                                break;
                            }
                        }
                        result
                    }
                    _ => panic!("unknown player body action"),
                };
                let frame = json!({"state":state,"ropes":ropes,"released":released});
                if let Err(error) = result {
                    return json!({"error":error,"action":frames.len(),"completed":frames,"failed":frame});
                }
                frames.push(frame);
            }
            return json!(frames);
        }
        "dynamics_player_ground" => {
            let mut state: bopl_sim::player_ground::State = read(&q, "state");
            let surfaces: Vec<bopl_sim::surface::Surface> = read(&q, "surfaces");
            let mut frames = Vec::new();
            for action in q["actions"].as_array().unwrap() {
                let mut event = Value::Null;
                let result = match action["op"].as_str().unwrap() {
                    "attach" => {
                        let id: u32 = read(action, "surface");
                        state.attach(id, &surfaces[id as usize], read(action, "position"))
                    }
                    "unground" => {
                        event = json!(state.unground(
                            state.ground.map(|i| &surfaces[i as usize]),
                            read(action, "null_rotation")
                        ));
                        Ok(())
                    }
                    _ => panic!("unknown ground action"),
                };
                let frame = json!({"state":state,"event":event});
                if let Err(error) = result {
                    return json!({"error":format!("{error:?}"),"action":frames.len(),"completed":frames,"failed":frame});
                }
                frames.push(frame);
            }
            return json!(frames);
        }
        "dynamics_player_collision" => {
            let mut receiver: bopl_sim::player_collision::Receiver = read(&q, "receiver");
            let mut globals: bopl_sim::player_collision::Globals = read(&q, "globals");
            let mut hits: Vec<bopl_sim::player_collision::Hit> = read(&q, "hits");
            let mut frames = Vec::new();
            for action in q["actions"].as_array().unwrap() {
                let mut effects = Vec::new();
                let mut result = Value::Null;
                match action["op"].as_str().unwrap() {
                    "update" => receiver.update(),
                    "invulnerable" => {
                        receiver.set_invulnerable(read(action, "enabled"), read(action, "mask"))
                    }
                    "ignore_beam" => receiver.ignored_beam = read(action, "id"),
                    "copy_ignored" => receiver.ignored = read(action, "ignored"),
                    "collide" => {
                        let callback = read(action, "callback");
                        let outcome = receiver.collide(
                            &mut hits[read::<usize>(action, "hit")],
                            &mut globals,
                            callback,
                            &mut |effect| {
                                effects.push(effect);
                                read(action, "kill_result")
                            },
                        );
                        match outcome {
                            Ok(value) => {
                                if !callback {
                                    result = json!(value);
                                }
                            }
                            Err(error) => {
                                return json!({"error": error, "action": frames.len(), "completed": frames, "failed": {"receiver":receiver,"globals":globals,"hits":hits,"effects":effects,"result":Value::Null}});
                            }
                        }
                    }
                    _ => panic!("unknown collision policy action"),
                }
                frames.push(json!({"receiver":receiver,"globals":globals,"hits":hits,"effects":effects,"result":result}));
            }
            return json!(frames);
        }
        "dynamics_player_life" => {
            let mut state: bopl_sim::player_life::Book = read(&q, "state");
            let mut events: Vec<bopl_sim::player_life::Death> = read(&q["state"], "events");
            let mut frames = Vec::new();
            for a in q["actions"].as_array().unwrap() {
                if let Some(event) = state.kill(
                    read(a, "victim"),
                    read(a, "killer"),
                    read(a, "timestamp"),
                    read(a, "cause"),
                ) {
                    events.push(event);
                }
                frames.push(json!({"players": state.players, "registered_kills": state.registered_kills, "events": events}));
            }
            return json!(frames);
        }
        "dynamics_curve" => {
            let curve: bopl_sim::curve::Curve = read(&q, "curve");
            return json!(
                read::<Vec<bopl_sim::fixed::Fix>>(&q, "times")
                    .into_iter()
                    .map(|t| curve.evaluate(t))
                    .collect::<Vec<_>>()
            );
        }
        "dynamics_beam_hitbox" => {
            let config = read(&q, "config");
            let mut hitbox = bopl_sim::beam::hitbox::Hitbox::default();
            let mut states = Vec::new();
            for action in q["actions"].as_array().unwrap() {
                if action["operation"] == "initialize" {
                    hitbox.initialize(config);
                } else {
                    hitbox.set_dimensions(config, &read(action, "dimensions"));
                }
                states.push(hitbox.clone());
            }
            return json!(states);
        }
        "dynamics_beam_effects" => {
            let config = read(&q, "config");
            let mut targets = read::<Vec<bopl_sim::beam::effects::Target>>(&q, "targets");
            let mut states = Vec::new();
            for impacts in read::<Vec<Vec<bopl_sim::beam::effects::Impact>>>(&q, "frames") {
                bopl_sim::beam::effects::apply(&config, &impacts, &mut targets);
                states.push(targets.clone());
            }
            return json!(states);
        }
        "dynamics_beam_capture" => {
            let mut scene: physics::scene::Scene = read(&q, "scene");
            let mut beams: bopl_sim::beam::system::System = read(&q, "beams");
            let step = beams.simulate(&mut scene, read(&q, "dt"), &[]);
            scene.objects.sort_by_key(|o| o.collider.kind());
            return json!({"scene": scene, "beams": beams, "callbacks": step.physics.callbacks});
        }
        "dynamics_beam_scene" => {
            let mut scene: physics::scene::Scene = read(&q, "scene");
            let next_id = scene.objects.iter().map(|o| o.id).max().unwrap_or(0) + 1;
            let mut beams = bopl_sim::beam::system::System::new(
                read(&q, "config"),
                read(&q, "targets"),
                next_id,
            );
            let mut states = Vec::new();
            for frame in q["frames"].as_array().unwrap() {
                let input = read::<Vec<bopl_sim::beam::graph::Beam>>(frame, "beams");
                if let Some(slots) = frame["destroy_slots"].as_array() {
                    for slot in slots {
                        beams.destroy_slot(&mut scene, slot.as_u64().unwrap() as usize);
                    }
                }
                let step = beams.simulate(&mut scene, read(frame, "dt"), &input);
                scene.objects.sort_by_key(|o| o.collider.kind());
                states.push(json!({"scene": scene, "graph": step.graph, "pool": beams.pool, "targets": beams.targets,
                    "callbacks": step.physics.callbacks}));
            }
            return json!(states);
        }
        "dynamics_beam_graph" => {
            let mut graph = bopl_sim::beam::graph::Graph::new(read(&q, "config"));
            let mut scene: physics::scene::Scene = read(&q, "scene");
            let mut output = Vec::new();
            for frame in read::<Vec<Vec<bopl_sim::beam::graph::Beam>>>(&q, "frames") {
                output.push(graph.step(&frame, &mut |p, d, length, mask| {
                    scene.raycast(p, d, length, mask)
                }));
            }
            return json!(output);
        }
        "dynamics_beam_colors" => {
            let a: bopl_sim::beam::color::Colors = read(&q, "a");
            return json!(a.mix(read(&q, "b")));
        }
        "dynamics_beam_lines" => {
            return json!(bopl_sim::beam::intersection::lines(
                read(&q, "a"),
                read(&q, "b")
            ));
        }
        "dynamics_beam_intersection" => {
            return json!(bopl_sim::beam::intersection::beams(
                read(&q, "a"),
                read(&q, "b"),
                read(&q, "minimize_b")
            ));
        }
        "dynamics_rope" => {
            let mut rope: physics::rope::Rope = read(&q, "rope");
            let config = read(&q, "config");
            let rectangles: Vec<RoundedRect> = read(&q, "rectangles");
            let mut output = Vec::new();
            for a in q["actions"].as_array().unwrap() {
                let result = match a["op"].as_str().unwrap() {
                    "step" => {
                        rope.step(config, &rectangles);
                        Value::Null
                    }
                    "bounds" => json!(rope.bounds()),
                    "spawn" => json!(rope.spawn(read(a, "point"))),
                    "reel" => json!(rope.reel(read(a, "minimum"), false)),
                    "reverse_reel" => json!(rope.reel(read(a, "minimum"), true)),
                    "mass" => {
                        rope.set_mass(read(a, "value"), read(a, "top"), read(a, "scale"));
                        Value::Null
                    }
                    "detach" => {
                        rope.detach(read(a, "top"));
                        Value::Null
                    }
                    _ => panic!("unknown rope action"),
                };
                output.push(json!({"rope":rope,"result":result}));
            }
            return json!(output);
        }
        "dynamics_scene" => {
            let mut scene: physics::scene::Scene = read(&q, "scene");
            let mut steps = Vec::new();
            for (tick, dt) in read::<Vec<Fix>>(&q, "steps").into_iter().enumerate() {
                if let Some(commands) = q["commands"][tick].as_array() {
                    for command in commands {
                        let id: u32 = read(command, "id");
                        match command["op"].as_str().unwrap() {
                            "attach" => scene.attach(id, read(command, "root")),
                            "detach" => scene.detach(id),
                            "destroy" => scene.destroy(id),
                            "force" => {
                                scene
                                    .objects
                                    .iter_mut()
                                    .find(|o| o.id == id)
                                    .unwrap()
                                    .body
                                    .force += read::<Vec2>(command, "force")
                            }
                            _ => panic!("unknown scene command"),
                        }
                    }
                }
                let report = scene.step(dt);
                steps.push(json!({"scene":scene,"report":report}));
            }
            return json!(steps);
        }
        "dynamics_circle_circle" => contact::circle_circle(read(&q, "a"), read(&q, "b")),
        "dynamics_circle_box" => contact::circle_box(read(&q, "a"), read(&q, "b")),
        "dynamics_circle_rounded" => contact::circle_rounded(read(&q, "a"), read(&q, "b")),
        "dynamics_box_box" => contact::box_box(read(&q, "a"), read(&q, "b")),
        "dynamics_box_rounded" => contact::rounded_rounded(
            RoundedRect {
                inner: read(&q, "a"),
                radius: Fix::ZERO,
            },
            read(&q, "b"),
        ),
        "dynamics_rounded_rounded" => contact::rounded_rounded(read(&q, "a"), read(&q, "b")),
        "dynamics_integrate" => {
            let mut body: physics::Body = read(&q, "body");
            let translation = body.integrate(read(&q, "dt"), read(&q, "gravity"));
            return json!({"body":body,"translation":translation});
        }
        "dynamics_resolve" => {
            let mut a = read(&q, "a");
            let mut b = read(&q, "b");
            physics::resolve(
                read(&q, "contact"),
                &mut a,
                &mut b,
                read::<Vec2>(&q, "pos_a"),
                read::<Vec2>(&q, "pos_b"),
            );
            return json!({"a":a,"b":b});
        }
        "dynamics_correction" => {
            return json!(physics::positional_correction(
                read(&q, "contact"),
                read(&q, "inverse_a"),
                read(&q, "inverse_b"),
                read(&q, "full")
            ));
        }
        _ => panic!("unknown operation"),
    };
    json!(contact)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::panic::set_hook(Box::new(|_| {}));
    let mut out = BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let query = serde_json::from_str(&line?)?;
        let result = std::panic::catch_unwind(|| evaluate(query)).unwrap_or_else(|panic| {
            let message = panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            if message.contains("division by zero") {
                json!({"error":"division_by_zero"})
            } else if matches!(
                message,
                "negative squared length: Domain"
                    | "normalized vector outside cosine domain: Domain"
                    | "negative black hole mass magnitude: Domain"
            ) {
                json!({"error":"domain"})
            } else {
                json!({"error":"rust_panic", "message":message})
            }
        });
        writeln!(out, "{result}")?;
    }
    Ok(())
}
