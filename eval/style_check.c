/* The C style VM (train/bopl_style.h) over replay-dump states, for
 * eval/style_check.py: Python rebuilds each decision's BoplRlState rows
 * (Q32.32 restored exactly from the dumps' float32) and the C VM's per-term
 * payouts are compared bit for bit with eval/style_score.py's.
 * Built as the trainer builds bopl_style.h (clang -O3 -march=znver4
 * -ffp-contract=off): make style-check. */
#include "bopl_style.h"

/* hdr: per decision tick, episode_ticks, flags, entity_count, slime_count,
 * sudden_death. players/telemetry: decisions x player_count. Rows and events
 * concatenated with decisions+1 offsets. Out: per player, per decision, the
 * n_terms payouts, then the totals (player x decision). */
int style_check_run(const void *program, long length, int decisions, int player_count, const uint32_t *hdr,
                    const BoplRlPlayer *players, const BoplRlTelemetry *telemetry,
                    const BoplRlEntity *entities, const int64_t *entity_offsets,
                    const BoplRlSlime *slimes, const int64_t *slime_offsets,
                    const BoplRlEvent *events, const int64_t *event_offsets, int support,
                    float *terms_out, float *totals_out, char *error, int capacity) {
    BoplStyleProgram p;
    if (bopl_style_load(&p, program, (size_t)length, error, (size_t)capacity)) return -1;
    if (player_count<1 || player_count>6) { bopl_style_free(&p); return -2; }
    float *slots=calloc(p.hist_floats, sizeof(float));
    BoplRlState state;
    for (int player=0; player<player_count; player++) {
        memset(slots, 0, p.hist_floats*sizeof(float));
        for (int d=0; d<decisions; d++) {
            memset(&state, 0, sizeof(state));
            const uint32_t *h=hdr+6*(size_t)d;
            state.header.tick=h[0]; state.header.episode_ticks=h[1]; state.header.flags=h[2];
            state.header.entity_count=h[3]; state.header.slime_count=h[4]; state.header.sudden_death=h[5];
            state.header.player_count=(uint32_t)player_count;
            memcpy(state.players, players+(size_t)d*player_count, sizeof(BoplRlPlayer)*(size_t)player_count);
            memcpy(state.telemetry, telemetry+(size_t)d*player_count, sizeof(BoplRlTelemetry)*(size_t)player_count);
            state.entities=entities+entity_offsets[d];
            state.slimes=slimes+slime_offsets[d];
            if ((int64_t)h[3]!=entity_offsets[d+1]-entity_offsets[d] || (int64_t)h[4]!=slime_offsets[d+1]-slime_offsets[d]) {
                free(slots); bopl_style_free(&p); return -3;
            }
            const BoplRlEvent *ev=events?events+event_offsets[d]:NULL;
            uint32_t n=events?(uint32_t)(event_offsets[d+1]-event_offsets[d]):0;
            size_t at=(size_t)player*decisions+(size_t)d;
            totals_out[at]=bopl_style_step(&p, &state, n?ev:NULL, n, support, player, slots, terms_out+at*p.n_terms);
        }
    }
    free(slots);
    bopl_style_free(&p);
    return 0;
}

int style_check_sizes(int *out) {
    out[0]=(int)sizeof(BoplRlPlayer); out[1]=(int)sizeof(BoplRlTelemetry); out[2]=(int)sizeof(BoplRlEntity);
    out[3]=(int)sizeof(BoplRlSlime); out[4]=(int)sizeof(BoplRlEvent);
    return 5;
}
