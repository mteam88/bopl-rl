# Reference re-score of overnight-3

Pool `native/artifacts/rl/sweeps/anchors/overnight-3/pool.json` (57d70df96257ed56), 240 games per checkpoint (first pass 6 per member, then the 15%-85% band), IQM over the last 3 evaluations, 95% bootstrap CI over seed pairs. Old score: the anchors scorer's final observation. Flags: no-offense (arrows_per_game < 0.05), platform-spam (platforms_per_game > 8), stalemate (timeout_rate > 0.5).

| trial | old score | old rank | final rating (95% CI) | IQM3 | new rank | move | flags |
|---|---|---|---|---|---|---|---|
| 012 | 0.536 | 1 | 413 [373, 453] | 371 | 1 | = |  |
| 009 | 0.500 | 2 | 344 [305, 382] | 359 | 2 | = |  |
| 002 | 0.422 | 4 | 369 [331, 408] | 351 | 3 | +1 |  |
| 006 | 0.375 | 9 | 383 [348, 422] | 350 | 4 | +5 | no-offense |
| 011 | 0.411 | 6 | 336 [298, 372] | 339 | 5 | +1 |  |
| 010 | 0.307 | 11 | 329 [291, 371] | 335 | 6 | +5 | no-offense |
| 005 | 0.302 | 12 | 329 [292, 366] | 324 | 7 | +5 | no-offense |
| 007 | 0.411 | 5 | 353 [313, 391] | 323 | 8 | -3 |  |
| 001 | 0.453 | 3 | 327 [289, 365] | 320 | 9 | -6 | no-offense |
| 013 | 0.297 | 13 | 300 [262, 339] | 314 | 10 | +3 |  |
| 003 | 0.396 | 8 | 292 [251, 331] | 312 | 11 | -3 | no-offense |
| 008 | 0.349 | 10 | 301 [261, 336] | 288 | 12 | -2 | no-offense |
| 000 | 0.406 | 7 | 310 [272, 349] | 288 | 13 | -6 | no-offense |
| 004 | 0.271 | 14 | 279 [234, 320] | 277 | 14 | = | no-offense |

Rank changes: 000, 001, 002, 003, 005, 006, 007, 008, 010, 011, 013.

## Last-k ratings per trial

- 000: policy-213352448.bin: 294 [250, 331], policy-240025600.bin: 260 [219, 302], policy-266698752.bin: 310 [272, 349]
- 001: policy-213352448.bin: 323 [286, 359], policy-240025600.bin: 310 [269, 349], policy-266698752.bin: 327 [289, 365]
- 002: policy-213352448.bin: 340 [294, 382], policy-240025600.bin: 343 [303, 380], policy-266698752.bin: 369 [331, 408]
- 003: policy-213352448.bin: 331 [295, 367], policy-240025600.bin: 313 [276, 351], policy-266698752.bin: 292 [251, 331]
- 004: policy-213352448.bin: 298 [258, 338], policy-240025600.bin: 253 [213, 292], policy-266698752.bin: 279 [234, 320]
- 005: policy-213352448.bin: 310 [271, 352], policy-240025600.bin: 333 [294, 376], policy-266698752.bin: 329 [292, 366]
- 006: policy-213352448.bin: 355 [321, 390], policy-240025600.bin: 310 [280, 341], policy-266698752.bin: 383 [348, 422]
- 007: policy-213352448.bin: 303 [261, 343], policy-240025600.bin: 313 [279, 346], policy-266698752.bin: 353 [313, 391]
- 008: policy-213352448.bin: 262 [221, 302], policy-240025600.bin: 302 [264, 342], policy-266698752.bin: 301 [261, 336]
- 009: policy-213352448.bin: 360 [316, 401], policy-240025600.bin: 372 [333, 412], policy-266698752.bin: 344 [305, 382]
- 010: policy-213352448.bin: 365 [326, 401], policy-240025600.bin: 311 [271, 351], policy-266698752.bin: 329 [291, 371]
- 011: policy-213352448.bin: 320 [279, 367], policy-240025600.bin: 362 [325, 402], policy-266698752.bin: 336 [298, 372]
- 012: policy-213352448.bin: 353 [311, 394], policy-240025600.bin: 348 [309, 384], policy-266698752.bin: 413 [373, 453]
- 013: policy-213352448.bin: 307 [272, 344], policy-240025600.bin: 334 [298, 374], policy-266698752.bin: 300 [262, 339]
