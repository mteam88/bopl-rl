/* NullAbility is the catalog's intentionally unusable menu placeholder. The
 * public match boundary must reject id 0 on both backends; component-level
 * clone failure prefixes are covered by native/fixtures/null-ability-clones.
 *
 * cc -O2 -Wall -Wextra -Werror -I. -o /tmp/null_ability_ab \
 *   research/rust_ab/null_ability_ab.c bopl_rl.c -ldl -lm
 */
#include "bopl_rl.h"
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <BoplSimNative.so> <libbopl_sim.so>\n", argv[0]);
        return 2;
    }
    char error[2][512] = {{0}};
    BoplRl *world[2];
    for (int backend = 0; backend < 2; backend++) {
        world[backend] = bopl_rl_open(argv[backend + 1], "/tmp/gate_bootstrap.rep",
                                      "/tmp/null_ability_ab.log", 120000,
                                      error[backend], sizeof error[backend]);
        if (!world[backend]) {
            fprintf(stderr, "open failed: %s\n", error[backend]);
            return 1;
        }
        if (bopl_rl_ability_count(world[backend]) != 39) {
            fprintf(stderr, "backend %d catalog count is not 39\n", backend);
            return 1;
        }
    }
    BoplRlConfig config;
    bopl_rl_default_config(&config, 1);
    config.player[0].abilities[0] = 0;
    int reference = bopl_rl_reset(world[0], &config);
    int rust = bopl_rl_reset(world[1], &config);
    if (reference != BOPL_RL_ARGUMENT || rust != reference) {
        fprintf(stderr, "reset status reference=%d rust=%d\n", reference, rust);
        return 1;
    }
    for (int backend = 0; backend < 2; backend++) bopl_rl_close(world[backend]);
    puts("NULL_ABILITY_GOLDEN_OK status=argument catalog=39");
    return 0;
}
