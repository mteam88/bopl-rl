use super::read;
use bopl_sim::{
    bopl_body::{Context, State},
    fixed::Fix,
    physics::scene::Scene,
    scheduler,
};
use serde_json::{Value, json};
pub fn evaluate(q: &Value) -> Value {
    let mut state: State = read(q, "state");
    let mut scene: Scene = read(q, "scene");
    let mut context = Context {
        scene: &mut scene,
        object: read(q, "object"),
        initialized: read(q, "initialized"),
        transform_scale: read(q, "transform_scale"),
    };
    let mut frames = vec![];
    for action in q["actions"].as_array().unwrap() {
        let mut value = Value::Null;
        let result = (|| -> scheduler::Result<()> {
            match action["op"].as_str().unwrap() {
                "init" => {}
                "update" => state.update(&mut context)?,
                "hold" => state.hold(
                    read(action, "frames"),
                    read(action, "magnet"),
                    read(action, "change_sorting"),
                ),
                "drop" => state.drop_held(),
                "held" => value = json!(state.frames_until_drop >= 0),
                "new_body" => value = json!(state.config.new_body(read(action, "orientation"))?),
                "inverse_mass" => value = json!(state.inverse_mass(&mut context)?),
                "inverse_inertia" => value = json!(state.inverse_inertia(&mut context)?),
                "set_mass" => state.set_inverse_mass(read(action, "value"), &mut context)?,
                "set_inertia" => state.set_inverse_inertia(read(action, "value"), &mut context)?,
                "set_velocity" => state.set_velocity(read(action, "value"), &mut context)?,
                "set_angular" => state.set_angular_velocity(read(action, "value"), &mut context)?,
                "velocity" => value = json!(context.body()?.velocity),
                "angular" => value = json!(context.body()?.angular_velocity),
                "body" => value = json!(context.body()?),
                "force" => {
                    context.add_force(read(action, "force"), None, read(action, "impulse"))?
                }
                "force_at" => context.add_force(
                    read(action, "force"),
                    Some(read(action, "position")),
                    read(action, "impulse"),
                )?,
                "transform_scale" => {
                    let scale: Fix = read(action, "value");
                    let factor = scale
                        .checked_div(context.transform_scale)
                        .map_err(|_| "division_by_zero")?;
                    let object = context
                        .scene
                        .objects
                        .iter_mut()
                        .find(|o| o.id == context.object)
                        .unwrap();
                    object.transform.offset *= factor;
                    context.transform_scale = scale;
                }
                "bind_initialized" => context.initialized = read(action, "value"),
                _ => panic!("unknown BoplBody action"),
            }
            Ok(())
        })();
        let frame = json!({"state":state,"scene":context.scene,"initialized":context.initialized,"transform_scale":context.transform_scale,"result":value});
        if let Err(error) = result {
            return json!({"error":error,"action":frames.len(),"completed":frames,"failed":frame});
        }
        frames.push(frame);
    }
    json!(frames)
}
