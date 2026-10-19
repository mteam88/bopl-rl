# Failure atlas: policy-1333395456.bin

5904 games from 8 files, 2952 seed pairs. A scores 0.613 [0.599, 0.627] overall. Intervals are 95%, clustered on seed pairs.

## Opponents

| opponent | games | score | 95% CI | A losses | draws (timeout / both-dead / neither) | median ticks |
|---|---|---|---|---|---|---|
| anneal20-26666m | 1076 | 0.547 | [0.513, 0.581] | 482 | 2 / 9 / 0 | 6878 |
| final5-18666m | 1076 | 0.532 | [0.498, 0.566] | 499 | 4 / 5 / 0 | 7143 |
| g111-20b-26027m | 1076 | 0.808 | [0.782, 0.834] | 199 | 2 / 13 / 0 | 4234 |
| l2-me32-trap-2000m | 400 | 0.596 | [0.540, 0.653] | 159 | 3 / 2 / 0 | 7520 |
| team5b-6250m | 1076 | 0.574 | [0.541, 0.607] | 450 | 5 / 12 / 0 | 7011 |
| x2-blinkplat-1500m | 400 | 0.649 | [0.596, 0.701] | 140 | 0 / 1 / 0 | 7372 |
| x2-drill-1500m | 400 | 0.588 | [0.533, 0.642] | 163 | 3 / 1 / 0 | 7151 |
| x2-missile-1500m | 400 | 0.598 | [0.543, 0.652] | 158 | 3 / 3 / 0 | 7123 |

## Losses by game phase and cause

Share of all games that A lost, by when the game ended and how A died.

| phase | games | A loss rate | fall | opponent | self | squished | beamed | eaten | uncredited |
|---|---|---|---|---|---|---|---|---|---|
| opening <15s | 1072 | 0.321 | 114 | 151 | 12 | 33 | 20 | 1 | 13 |
| early 15-60s | 1243 | 0.364 | 163 | 150 | 11 | 50 | 39 | 12 | 27 |
| late 60-116s | 559 | 0.367 | 68 | 70 | 12 | 23 | 10 | 6 | 16 |
| sudden death | 3030 | 0.412 | 999 | 131 | 32 | 14 | 54 | 6 | 13 |
| all | 5904 | 0.381 | 1344 | 502 | 67 | 120 | 123 | 25 | 69 |

## A vs opponent: decisive deaths by cause and phase

Per game: did the seat die this way and not win? A's rate and the opponent's in the same games. A is the stronger seat, so A/opp < 1 is normal; the cells nearest 1 are where A's edge is thinnest. q is BH over the table for ratio = 1.

| cause | phase | A | opp | A - opp | 95% CI | A/opp | 95% CI | q |
|---|---|---|---|---|---|---|---|---|
| beamed | sudden death | 0.0091 | 0.0107 | -0.0015 | [-0.0057, +0.0026] | 0.86 | [0.50, 1.22] | 0.44 |
| squished | early 15-60s | 0.0085 | 0.0108 | -0.0024 | [-0.0059, +0.0012] | 0.78 | [0.49, 1.07] | 0.16 |
| self | sudden death | 0.0063 | 0.0081 | -0.0019 | [-0.0052, +0.0015] | 0.77 | [0.40, 1.14] | 0.25 |
| self | late 60-116s | 0.0024 | 0.0032 | -0.0008 | [-0.0028, +0.0011] | 0.74 | [0.23, 1.25] | 0.33 |
| fall | sudden death | 0.1707 | 0.2332 | -0.0625 | [-0.0813, -0.0437] | 0.73 | [0.66, 0.80] | 2.1e-13 |
| uncredited | late 60-116s | 0.0027 | 0.0037 | -0.0010 | [-0.0031, +0.0010] | 0.73 | [0.26, 1.20] | 0.28 |
| beamed | late 60-116s | 0.0017 | 0.0024 | -0.0007 | [-0.0023, +0.0009] | 0.71 | [0.13, 1.29] | 0.34 |
| uncredited | early 15-60s | 0.0047 | 0.0069 | -0.0022 | [-0.0050, +0.0006] | 0.68 | [0.35, 1.02] | 0.076 |
| squished | all | 0.0207 | 0.0305 | -0.0098 | [-0.0158, -0.0038] | 0.68 | [0.52, 0.84] | 0.00017 |
| fall | all | 0.2293 | 0.3386 | -0.1092 | [-0.1312, -0.0873] | 0.68 | [0.62, 0.73] | 1.1e-30 |
| opponent | late 60-116s | 0.0120 | 0.0180 | -0.0059 | [-0.0106, -0.0013] | 0.67 | [0.46, 0.88] | 0.0034 |
| opponent | sudden death | 0.0225 | 0.0344 | -0.0119 | [-0.0185, -0.0052] | 0.66 | [0.50, 0.81] | 2.5e-05 |
| squished | opening <15s | 0.0058 | 0.0088 | -0.0030 | [-0.0063, +0.0002] | 0.65 | [0.35, 0.96] | 0.033 |
| squished | late 60-116s | 0.0041 | 0.0064 | -0.0024 | [-0.0049, +0.0002] | 0.63 | [0.31, 0.95] | 0.03 |
| opponent | early 15-60s | 0.0261 | 0.0427 | -0.0166 | [-0.0237, -0.0095] | 0.61 | [0.48, 0.74] | 1.5e-08 |
| fall | early 15-60s | 0.0278 | 0.0473 | -0.0195 | [-0.0268, -0.0122] | 0.59 | [0.47, 0.71] | 5.5e-11 |
| beamed | all | 0.0208 | 0.0356 | -0.0147 | [-0.0220, -0.0075] | 0.59 | [0.43, 0.74] | 2.9e-07 |
| self | all | 0.0137 | 0.0235 | -0.0098 | [-0.0152, -0.0044] | 0.58 | [0.41, 0.76] | 7e-06 |
| opponent | all | 0.0866 | 0.1518 | -0.0652 | [-0.0797, -0.0508] | 0.57 | [0.50, 0.64] | 1.1e-30 |
| self | opening <15s | 0.0025 | 0.0046 | -0.0020 | [-0.0042, +0.0001] | 0.56 | [0.21, 0.90] | 0.017 |
| uncredited | sudden death | 0.0036 | 0.0064 | -0.0029 | [-0.0051, -0.0007] | 0.55 | [0.30, 0.81] | 0.001 |
| fall | opening <15s | 0.0193 | 0.0354 | -0.0161 | [-0.0223, -0.0099] | 0.55 | [0.42, 0.67] | 2.1e-11 |
| squished | sudden death | 0.0024 | 0.0044 | -0.0020 | [-0.0041, +0.0001] | 0.54 | [0.19, 0.89] | 0.014 |
| uncredited | all | 0.0136 | 0.0257 | -0.0122 | [-0.0172, -0.0072] | 0.53 | [0.39, 0.67] | 1.6e-10 |
| fall | late 60-116s | 0.0115 | 0.0227 | -0.0112 | [-0.0160, -0.0063] | 0.51 | [0.36, 0.66] | 5.4e-10 |
| beamed | early 15-60s | 0.0066 | 0.0139 | -0.0073 | [-0.0112, -0.0033] | 0.48 | [0.28, 0.67] | 2.6e-07 |
| opponent | opening <15s | 0.0259 | 0.0567 | -0.0308 | [-0.0390, -0.0227] | 0.46 | [0.36, 0.55] | 7.2e-28 |
| eaten | all | 0.0042 | 0.0095 | -0.0053 | [-0.0085, -0.0020] | 0.45 | [0.22, 0.68] | 4.9e-06 |
| eaten | early 15-60s | 0.0020 | 0.0047 | -0.0027 | [-0.0049, -0.0005] | 0.43 | [0.13, 0.72] | 0.00026 |
| beamed | opening <15s | 0.0034 | 0.0086 | -0.0053 | [-0.0081, -0.0024] | 0.39 | [0.19, 0.59] | 1.1e-08 |
| self | early 15-60s | 0.0025 | 0.0076 | -0.0051 | [-0.0076, -0.0025] | 0.33 | [0.14, 0.53] | 8.2e-11 |
| uncredited | opening <15s | 0.0025 | 0.0086 | -0.0061 | [-0.0088, -0.0034] | 0.29 | [0.13, 0.46] | 5.6e-16 |
| eaten | late 60-116s | 0.0010 | 0.0036 | -0.0025 | [-0.0043, -0.0008] | 0.29 | [0.03, 0.54] | 1.7e-07 |

## Abilities A plays worse than its opponents (asymmetry)

own = residual when A holds the ability, opp = residual when the opponent holds it; asym = own + opp. Kit power alone gives asym ~0 (Gold mushroom: big own gain, equal loss against it). asym < 0: A gets less out of the ability than the opponents get out of it against A.

| ability | own (n) | opp (n) | asym | 95% CI | p | q (BH) | Holm |
|---|---|---|---|---|---|---|---|

## Paired seeds (same map and kits, sides swapped)

| opponent | pairs | A lost side 0 | side 1 | lost both | if independent | score corr |
|---|---|---|---|---|---|---|
| anneal20-26666m | 538 | 0.468 | 0.428 | 0.275 | 0.200 | +0.293 |
| final5-18666m | 538 | 0.476 | 0.452 | 0.290 | 0.215 | +0.307 |
| g111-20b-26027m | 538 | 0.201 | 0.169 | 0.076 | 0.034 | +0.278 |
| l2-me32-trap-2000m | 200 | 0.410 | 0.385 | 0.250 | 0.158 | +0.404 |
| team5b-6250m | 538 | 0.407 | 0.429 | 0.230 | 0.175 | +0.239 |
| x2-blinkplat-1500m | 200 | 0.325 | 0.375 | 0.180 | 0.122 | +0.252 |
| x2-drill-1500m | 200 | 0.415 | 0.400 | 0.235 | 0.166 | +0.291 |
| x2-missile-1500m | 200 | 0.410 | 0.380 | 0.230 | 0.156 | +0.315 |

648 of 2952 seed pairs were lost on both sides.

## Loss pockets (pre-game facts, opponent-adjusted)

199 pockets with >= 40 games tested. residual = pocket score minus A's score vs the same opponent; excess = points below expectation over the pocket. Rows with BH q < 0.05, most costly first, then the strongest pockets for contrast.

| pocket | family | games | score | residual | 95% CI | excess loss | p | q (BH) | Holm |
|---|---|---|---|---|---|---|---|---|---|
| opp holds Gold mushroom | opp ability | 482 | 0.370 | -0.249 | [-0.294, -0.203] | +119.8 | 4.4e-27 | 4.3e-25 | 8.7e-25 |
| opp holds Platform | opp ability | 480 | 0.452 | -0.163 | [-0.213, -0.114] | +78.3 | 1.2e-10 | 4.7e-09 | 2.3e-08 |
| opp holds Beam | opp ability | 422 | 0.475 | -0.145 | [-0.196, -0.094] | +61.2 | 2.7e-08 | 9e-07 | 5.3e-06 |
| A holds Throw | own ability | 530 | 0.530 | -0.094 | [-0.142, -0.045] | +49.6 | 0.00018 | 0.003 | 0.034 |
| A holds Tesla coil | own ability | 496 | 0.510 | -0.092 | [-0.141, -0.043] | +45.8 | 0.00021 | 0.0033 | 0.04 |
| opp holds Meteor | opp ability | 462 | 0.504 | -0.097 | [-0.145, -0.049] | +44.9 | 7.6e-05 | 0.0015 | 0.015 |
| A holds Web swing | own ability | 450 | 0.519 | -0.096 | [-0.148, -0.045] | +43.4 | 0.00025 | 0.0036 | 0.046 |
| A holds Mine | own ability | 484 | 0.530 | -0.087 | [-0.137, -0.037] | +42.1 | 0.00065 | 0.0068 | 0.12 |
| A holds Engine | own ability | 416 | 0.506 | -0.094 | [-0.146, -0.042] | +39.1 | 0.00041 | 0.0054 | 0.076 |
| A holds Black hole | own ability | 478 | 0.540 | -0.071 | [-0.119, -0.023] | +34.1 | 0.0035 | 0.029 | 0.62 |
| A holds Portal gun | own ability | 420 | 0.530 | -0.076 | [-0.130, -0.022] | +32.0 | 0.0059 | 0.039 | 1 |
| 14 Level15 heldout | map | 48 | 0.354 | -0.254 | [-0.399, -0.109] | +12.2 | 0.00059 | 0.0066 | 0.11 |
| A holds Gold mushroom | own ability | 456 | 0.890 | +0.268 | [+0.238, +0.297] | -122.0 | 2.6e-69 | 5.2e-67 | 5.2e-67 |
| A holds Platform | own ability | 516 | 0.756 | +0.143 | [+0.106, +0.180] | -73.8 | 5.9e-14 | 2.9e-12 | 1.1e-11 |
| A holds Beam | own ability | 460 | 0.767 | +0.157 | [+0.118, +0.197] | -72.3 | 4.9e-15 | 3.2e-13 | 9.6e-13 |
| opp holds Portal gun | opp ability | 496 | 0.716 | +0.109 | [+0.065, +0.153] | -53.9 | 1.3e-06 | 3.3e-05 | 0.00026 |
| A holds Meteor | own ability | 484 | 0.713 | +0.111 | [+0.066, +0.155] | -53.6 | 9.4e-07 | 2.7e-05 | 0.00018 |
| A holds Magnet gun | own ability | 500 | 0.694 | +0.080 | [+0.035, +0.125] | -40.0 | 0.00055 | 0.0064 | 0.1 |
| opp holds Push | opp ability | 510 | 0.699 | +0.072 | [+0.026, +0.118] | -36.7 | 0.0023 | 0.022 | 0.42 |
| A holds Rock | own ability | 460 | 0.686 | +0.079 | [+0.034, +0.124] | -36.5 | 0.00055 | 0.0064 | 0.1 |

Near misses (q < 0.2): A holds Teleport -0.062 (n 466, q 0.10); opp holds Rock -0.060 (n 460, q 0.06); opp holds Grenade -0.056 (n 492, q 0.11); opp holds Mjolnir -0.051 (n 522, q 0.13); opp holds Bow -0.053 (n 498, q 0.11); opp holds Magnet gun -0.050 (n 454, q 0.20); 121 espace 8 space -0.179 (n 46, q 0.11); 29 Level23 -0.161 (n 48, q 0.16).

## Out-of-sample confirmation

12912 games from 12 files on independent seeds; the weak pockets above re-tested, one-sided, Holm over the rows.

| pocket | games | residual | 95% CI | p (one-sided) | Holm |
|---|---|---|---|---|---|
| opp holds Gold mushroom | 1064 | -0.243 | [-0.273, -0.213] | 6.5e-57 | 7.9e-56 |
| opp holds Platform | 1094 | -0.193 | [-0.225, -0.160] | 1.9e-31 | 2.1e-30 |
| opp holds Beam | 1118 | -0.162 | [-0.193, -0.132] | 8.3e-26 | 8.3e-25 |
| A holds Throw | 1110 | -0.071 | [-0.102, -0.039] | 5.3e-06 | 2.6e-05 |
| A holds Tesla coil | 1042 | -0.088 | [-0.121, -0.055] | 6.6e-08 | 5.3e-07 |
| opp holds Meteor | 966 | -0.032 | [-0.064, +0.001] | 0.03 | 0.06 |
| A holds Web swing | 1050 | -0.061 | [-0.095, -0.028] | 0.00018 | 0.00071 |
| A holds Mine | 1048 | -0.047 | [-0.080, -0.015] | 0.0023 | 0.0068 |
| A holds Engine | 1056 | -0.090 | [-0.124, -0.057] | 8.1e-08 | 5.7e-07 |
| A holds Black hole | 1080 | -0.098 | [-0.129, -0.066] | 5.5e-10 | 4.9e-09 |
| A holds Portal gun | 1042 | -0.080 | [-0.115, -0.045] | 3.1e-06 | 1.9e-05 |
| 14 Level15 heldout | 100 | -0.070 | [-0.178, +0.037] | 0.099 | 0.099 |

Asymmetry, same independent rows:

| ability | discovery asym | confirm asym | 95% CI | p (one-sided) |
|---|---|---|---|---|

## What decides sudden-death games (descriptive)

3030 games reached sudden death (A 0.580); 999 of A's 1249 losses there are falls. Phase is an outcome, so these pockets describe which kits survive the endgame, not causes. Rows with BH q < 0.05:

| pocket | games | score | residual | 95% CI | q (BH) |
|---|---|---|---|---|---|
| opp holds Gold mushroom | 306 | 0.216 | -0.367 | [-0.415, -0.319] | 4e-49 |
| opp holds Platform | 279 | 0.290 | -0.292 | [-0.353, -0.232] | 6.3e-20 |
| opp holds Beam | 153 | 0.395 | -0.190 | [-0.271, -0.108] | 7.2e-05 |
| A holds Gust | 228 | 0.467 | -0.120 | [-0.196, -0.044] | 0.015 |
| A holds Mine | 264 | 0.473 | -0.109 | [-0.178, -0.041] | 0.015 |
| A holds Throw | 289 | 0.486 | -0.105 | [-0.173, -0.038] | 0.015 |
| A holds Web swing | 251 | 0.488 | -0.095 | [-0.166, -0.025] | 0.042 |
| A holds Rewind | 345 | 0.662 | +0.078 | [+0.021, +0.134] | 0.039 |
| opp holds Missile | 247 | 0.662 | +0.085 | [+0.021, +0.148] | 0.042 |
| opp holds Roll | 273 | 0.665 | +0.087 | [+0.022, +0.153] | 0.042 |
| opp holds Black hole | 239 | 0.674 | +0.098 | [+0.030, +0.166] | 0.028 |
| opp holds Drill | 259 | 0.691 | +0.101 | [+0.036, +0.166] | 0.015 |
| opp holds Portal gun | 246 | 0.683 | +0.110 | [+0.047, +0.172] | 0.0063 |
| A holds Beam | 138 | 0.732 | +0.171 | [+0.096, +0.247] | 0.0001 |
| A holds Platform | 326 | 0.819 | +0.236 | [+0.191, +0.280] | 8.8e-24 |
| A holds Gold mushroom | 274 | 0.954 | +0.370 | [+0.345, +0.395] | 2.4e-184 |

## Worst setups (lost on both sides)

| opponent | seed | map | A kit | opp kit | ticks (side 0, 1) | A death (side 0, 1) |
|---|---|---|---|---|---|---|
| final5-18666m | 311024 | 1 Level2 | Freeze ray/Duplicator/Teleport | Blink gun/Meteor/Dash | 218, 366 | opponent, opponent |
| x2-drill-1500m | 1911020 | 20 Level21 | Shrink ray/Random/Smoke | Drill/Roll/Beam | 238, 353 | opponent, opponent |
| anneal20-26666m | 711194 | 27 Level21 heldout | Rock/Grenade/Invisibility | Freeze ray/Rock/Beam | 371, 260 | fall, fall |
| team5b-6250m | 1111392 | 67 e 4 | Smoke/Teleport/Push | Missile/Rock/Growth ray | 498, 162 | opponent, opponent |
| anneal20-26666m | 711108 | 28 Level22 | Blink gun/Portal gun/Rock | Mine/Duplicator/Duplicator | 285, 376 | fall, fall |
| final5-18666m | 311170 | 87 e 24 | Spike/Meteor/Black hole | Web swing/Tack/Grappling hook | 459, 239 | fall, fall |
| anneal20-26666m | 711067 | 101 esnow 8 | Missile/Engine/Portal gun | Meteor/Gold mushroom/Growth ray | 369, 353 | squished, squished |
| anneal20-26666m | 711203 | 0 Level1 | Mine/Throw/Tack | Web swing/Beam/Shrink ray | 347, 381 | beamed, squished |
| final5-18666m | 311370 | 97 esnow 4 | Tack/Growth ray/Drill | Meteor/Random/Throw | 499, 235 | fall, opponent |
| x2-drill-1500m | 1911013 | 16 Level17 | Tack/Freeze ray/Mjolnir | Rock/Push/Bow | 439, 314 | opponent, opponent |
| team5b-6250m | 1111360 | 52 Level21 | Missile/Platform/Web swing | Meteor/Grappling hook/Duplicator | 306, 454 | opponent, opponent |
| x2-missile-1500m | 2311134 | 106 esnow 13 | Mjolnir/Invisibility/Random | Freeze ray/Grappling hook/Platform | 478, 315 | fall, squished |
| team5b-6250m | 1111313 | 64 e 1 | Meteor/Magnet gun/Random | Gust/Tesla coil/Black hole | 569, 228 | opponent, squished |
| anneal20-26666m | 711192 | 21 Level14 | Rewind/Grappling hook/Black hole | Gold mushroom/Drill/Teleport | 540, 294 | opponent, opponent |
| team5b-6250m | 1111385 | 15 Level16 | Blink gun/Growth ray/Grappling hook | Throw/Drill/Mjolnir | 429, 456 | opponent, opponent |
| x2-drill-1500m | 1911103 | 89 e 26 | Grappling hook/Rock/Blink gun | Drill/Poke ball/Black hole | 428, 481 | opponent, fall |
| x2-missile-1500m | 2311041 | 54 Level23 | Blink gun/Drill/Roll | Rewind/Gold mushroom/Rock | 254, 698 | opponent, opponent |
| team5b-6250m | 1111403 | 104 esnow 11 | Black hole/Mjolnir/Missile | Rock/Gust/Rewind | 322, 637 | opponent, opponent |
| x2-missile-1500m | 2311038 | 120 espace 7 space | Mine/Engine/Rock | Mine/Growth ray/Invisibility | 503, 499 | fall, fall |
| x2-missile-1500m | 2311195 | 43 TLevel5 | Teleport/Dash/Mine | Duplicator/Random/Meteor | 504, 511 | squished, squished |

