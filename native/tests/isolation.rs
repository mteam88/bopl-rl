use bopl_sim::world::{Action, World};

fn run(actions: &[Action; 2], steps: usize) -> String {
    let mut world = World::level1();
    for _ in 0..steps {
        if world.terminal() {
            break;
        }
        world.step(actions).unwrap();
    }
    serde_json::to_string(&world).unwrap()
}

#[test]
fn worlds_are_independent_when_interleaved_and_threaded() {
    let a = [
        Action {
            angle: 1,
            buttons: 0,
        },
        Action {
            angle: 128,
            buttons: 0,
        },
    ];
    let b = [
        Action {
            angle: 64,
            buttons: 1,
        },
        Action {
            angle: 191,
            buttons: 3,
        },
    ];
    let mut wa = World::level1();
    let mut wb = World::level1();
    for _ in 0..100 {
        if !wa.terminal() {
            wa.step(&a).unwrap();
        }
        if !wb.terminal() {
            wb.step(&b).unwrap();
        }
    }
    assert_eq!(serde_json::to_string(&wa).unwrap(), run(&a, 100));
    assert_eq!(serde_json::to_string(&wb).unwrap(), run(&b, 100));
    let thread = std::thread::spawn(move || run(&a, 100));
    assert_eq!(thread.join().unwrap(), run(&a, 100));
}

#[test]
fn invalid_actions_leave_world_unchanged() {
    let mut world = World::level1();
    let before = serde_json::to_string(&world).unwrap();
    assert!(world.step(&[]).is_err());
    assert!(
        world
            .step(
                &[Action {
                    angle: 0,
                    buttons: 128
                }; 2]
            )
            .is_err()
    );
    assert_eq!(before, serde_json::to_string(&world).unwrap());
}
