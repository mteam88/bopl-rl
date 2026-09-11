#ifndef BOPL_BOTS_V2_H
#define BOPL_BOTS_V2_H
/* Scripted bot kinds after the frozen v1 set (SCRIPTED_BOTS.md "Versioning").
 * Included at the end of bopl_bots.h, so the v1 helpers (BotView, ballistic
 * solver, navigation) are available to build on; the v1 functions themselves
 * are never edited. Adding a kind: append its id to the enum and its name to
 * BOPL_BOT_NAMES in bopl_bots.h, mirror both in
 * evaluation_protocol.SCRIPTED_KINDS, and dispatch it below. */

_Static_assert(sizeof(BOPL_BOT_NAMES)/sizeof(BOPL_BOT_NAMES[0])==BOPL_BOT_KINDS+1, "one name per bot kind");

static void bopl_bot_decide_v2(BoplBot *bot, const BoplRlState *state, int player_index,
                               int hold_left, uint16_t held, const int hold_run[4], uint8_t out[3]) {
    (void)state; (void)player_index; (void)hold_run;
    bot->decisions++;
    /* Latched: keep the held buttons like every bot (bopl_latch ignores the
     * button head anyway; mirroring it keeps `last` honest). */
    if (hold_left>0) { out[0]=bot->last[0]; out[1]=(uint8_t)held; out[2]=0; return; }
    switch (bot->kind) {
    default: out[0]=BOT_AIM_UP; out[1]=0; out[2]=0; break;  /* unknown kind: stand */
    }
    memcpy(bot->last, out, 3);
}

#endif
