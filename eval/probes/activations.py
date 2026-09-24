"""The checkpoint's encoder intermediates for every dataset row.

Re-implements the CPU reference forward (train/bopl_set_encoder.h,
puffernet.h MinGRU, policy_eval.c decoder) in numpy/torch from the exported
weights, runs it over each (game, player) observation sequence in order and
writes <dataset>/acts-<tag>.npy with, per row:

    [0:128)   flat block after ReLU           (Linear(98 -> 128))
    [128:192) pooled player set (64)
    [192:256) pooled slot set (64)
    [256:320) pooled entity set (64)
    [320:384) pooled slime set (64)
    [384:512) fused 128 (ReLU(Linear(384 -> 128))), the MinGRU input
    [512:640) MinGRU state after this step (= the layer's memory)
    [640:768) MinGRU output (the decoder input)
    [768]     value head

The file is float16 (the f32 online value is labels.value_online).

`--check` also runs the native CPU policy on a few sequences and reports the
largest value / state difference (float16 storage of the observations bounds it).

    native/artifacts/rl/venv/bin/python eval/probes/activations.py --check
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import layout  # noqa: E402
import schema  # noqa: E402
from generate import OUT, FINAL  # noqa: E402

ACT_WIDTH = 769
SLICES = dict(flat=slice(0, 128), players=slice(128, 192), slots=slice(192, 256), entities=slice(256, 320),
              slimes=slice(320, 384), fused=slice(384, 512), gru_state=slice(512, 640), gru_out=slice(640, 768))


class Weights:
    """The flat checkpoint split the way puffernet's aligned reader does (8-float boundaries)."""

    def __init__(self, path):
        meta = json.loads(Path(str(path) + ".json").read_text())
        data = np.frombuffer(Path(path).read_bytes(), dtype="<f4")
        self.hidden = H = meta["hidden"]
        self.width = layout.decoder_logits(meta.get("aim_fourier", 0), meta.get("aim_ar", 0))
        self.idx = 0
        self.data = data

        def take(rows, cols):
            n = rows * cols
            block = self.data[self.idx:self.idx + n].reshape(rows, cols).astype(np.float32)
            self.idx = (self.idx + n + 7) & ~7
            return block
        self.flat = take(H, layout.SET_FLAT)
        self.player = take(layout.SET_DIM, layout.SET_PLAYER_PROJECTION)
        self.slot = take(layout.SET_DIM, layout.SET_SLOT_PROJECTION)
        self.entity = take(layout.SET_DIM, layout.SET_ENTITY_PROJECTION)
        self.slime = take(layout.SET_DIM, layout.SET_SLIME_PROJECTION)
        self.embedding = take(layout.EMBEDDING_IDS, layout.SET_DIM)
        self.fusion = take(H, H + 4 * layout.SET_DIM)
        self.decoder = take(self.width + 1, H)
        self.gru = [take(3 * H, H) for _ in range(meta["num_layers"])]
        assert self.idx >= data.size - 7, (self.idx, data.size)


def pool(rows, weight, embedding, id_column, device):
    """rows: (n, R, W) compact block. Bias-free projection of the first W-1 (or W) columns, embedding by id, max over valid rows."""
    valid = rows[:, :, 0] != 0
    width = weight.shape[1]
    x = rows[:, :, :width] @ weight.T  # (n, R, 64)
    if embedding is not None:
        ids = rows[:, :, id_column]
        exact = (ids == ids.round()) & (ids >= 0) & (ids < embedding.shape[0])
        ids = torch.where(exact, ids, torch.zeros_like(ids)).long()
        x = x + embedding[ids]
    x = torch.where(valid[:, :, None], x, torch.full_like(x, -float("inf")))
    pooled = torch.clamp(x.max(1).values, min=0.0)  # zero rows never win; empty set -> 0 (memset semantics)
    pooled = torch.where(valid.any(1)[:, None], pooled, torch.zeros_like(pooled))
    return pooled


def encode(compact_obs, w, device):
    """Compact observation rows -> (flat_relu, four pooled 64s, fused 128): (n, 384+128)."""
    obs = torch.as_tensor(np.asarray(compact_obs, dtype=np.float32), device=device)
    n = obs.shape[0]
    off = schema.COMPACT_OFFSETS
    flat_in = torch.cat([obs[:, off["prefix"]:off["prefix"] + layout.OBS_PREFIX],
                         obs[:, off["self"]:off["self"] + layout.SELF_FEATURES],
                         obs[:, off["previous"]:off["previous"] + layout.PREVIOUS_FEATURES]], 1)
    flat = torch.clamp(flat_in @ w["flat"].T, min=0)

    def block(name, width, kept):
        return obs[:, off[name]:off[name] + kept * width].reshape(n, kept, width)
    players = pool(block("players", layout.PLAYER_FEATURES, schema.PLAYER_CAP), w["player"], None, -1, device)
    slots = pool(block("slots", layout.SLOT_FEATURES, schema.SLOT_CAP), w["slot"], w["embedding"], layout.SLOT_ABILITY_ID, device)
    entities = pool(block("entities", layout.ENTITY_FEATURES, schema.ENTITY_CAP), w["entity"], w["embedding"], layout.ENTITY_ABILITY_ID, device)
    slimes = pool(block("slimes", layout.SLIME_FEATURES, schema.SLIME_CAP), w["slime"], None, -1, device)
    joined = torch.cat([flat, players, slots, entities, slimes], 1)
    fused = torch.clamp(joined @ w["fusion"].T, min=0)
    return torch.cat([joined, fused], 1)


def gru_step(x, state, proj):
    c = x @ proj.T
    H = x.shape[1]
    hidden, gate, hw = c[:, :H], c[:, H:2 * H], c[:, 2 * H:]
    gate_s = torch.sigmoid(gate)
    h_tilde = torch.where(hidden >= 0, hidden + 0.5, torch.sigmoid(hidden))
    new_state = state + gate_s * (h_tilde - state)
    hw_s = torch.sigmoid(hw)
    out = hw_s * new_state + (1 - hw_s) * x
    return out, new_state


class ZeroedPosition:
    """A read-only view of the obs memmap with the prefix's self x/64, y/64 (obs[8], obs[9]) zeroed."""

    def __init__(self, obs):
        self.obs = obs
        p = schema.COMPACT_OFFSETS["prefix"]
        self.cols = [p + 8, p + 9]

    def __getitem__(self, s):
        a = np.array(self.obs[s], dtype=np.float32)
        a[:, self.cols] = 0.0
        return a


def run(dataset, checkpoint, tag, device, obs_key="obs", batch=8192, zero_pos=False):
    index = json.loads((dataset / "index.json").read_text())
    obs = np.load(dataset / f"{obs_key}.npy", mmap_mode="r")
    if zero_pos:  # the absolute-position ablation: obs[8:10] (self x/64, y/64) -> 0 before the encoder
        obs = ZeroedPosition(obs)
    meta = np.load(dataset / "meta.npy", mmap_mode="r")
    rows = index["rows"]
    W = Weights(checkpoint)
    w = {k: torch.as_tensor(getattr(W, k), device=device) for k in ("flat", "player", "slot", "entity", "slime", "embedding", "fusion", "decoder")}
    gru = [torch.as_tensor(g, device=device) for g in W.gru]
    out = np.lib.format.open_memmap(dataset / f"acts-{tag}.npy", mode="w+", dtype=np.float16, shape=(rows, ACT_WIDTH))
    # Stage 1: the feedforward encoder in big batches. The fused block feeding the
    # recurrence is kept in float32 in RAM (the file is float16).
    fused32 = np.empty((rows, 128), dtype=np.float32)
    for start in range(0, rows, batch):
        enc = encode(obs[start:start + batch], w, device).cpu().numpy()
        out[start:start + batch, :512] = enc
        fused32[start:start + batch] = enc[:, 384:512]
    # Stage 2: the recurrent scan, per (game, player) sequence. Sequences are
    # contiguous (player-major inside a game) so each is one slice; scan many
    # sequences in lockstep to keep the matmuls wide.
    game_col, player_col = schema.META.index("game"), schema.META.index("player")
    keys = meta[:, game_col].astype(np.int64) * 2 + meta[:, player_col]
    bounds = np.flatnonzero(np.diff(keys)) + 1
    starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [rows]])
    lengths = ends - starts
    order = np.argsort(-lengths)
    group = 256
    for g0 in range(0, len(order), group):
        sel = order[g0:g0 + group]
        s, e = starts[sel], ends[sel]
        L = int(lengths[sel].max())
        H = W.hidden
        state = torch.zeros(len(sel), H, device=device)
        fused_all = [torch.as_tensor(fused32[s[i]:e[i]], device=device) for i in range(len(sel))]
        result = [np.zeros((e[i] - s[i], ACT_WIDTH - 512), dtype=np.float32) for i in range(len(sel))]
        for t in range(L):
            active = np.flatnonzero(lengths[sel] > t)
            x = torch.stack([fused_all[i][t] for i in active])
            st = state[active]
            for proj in gru:
                x, st = gru_step(x, st, proj)
            state[active] = st
            dec = x @ w["decoder"].T
            packed = torch.cat([st, x, dec[:, -1:]], 1).cpu().numpy()
            for j, i in enumerate(active):
                result[i][t] = packed[j]
        for i in range(len(sel)):
            out[s[i]:e[i], 512:] = result[i]
    out.flush()
    return out


def check(dataset, checkpoint, tag, sequences=6):
    """Compare against the native CPU policy on the first few sequences (float16 obs feed both)."""
    from native_policy import load
    policy, _ = load(checkpoint)
    acts = np.load(dataset / f"acts-{tag}.npy", mmap_mode="r")
    obs = np.load(dataset / "obs.npy", mmap_mode="r")
    labels = np.load(dataset / "labels.npy", mmap_mode="r")
    meta = np.load(dataset / "meta.npy", mmap_mode="r")
    value_col = schema.LABELS.index("value_online")
    game_col, player_col, ctrl_col = (schema.META.index(k) for k in ("game", "player", "controller"))
    keys = meta[:, game_col].astype(np.int64) * 2 + meta[:, player_col]
    bounds = np.flatnonzero(np.diff(keys)) + 1
    starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [len(keys)]])
    worst_state, worst_value, worst_online = 0.0, 0.0, 0.0
    for s, e in list(zip(starts, ends))[:sequences]:
        state = policy.initial(1, "cpu")
        for r in range(s, e):
            full = torch.from_numpy(schema.expand(obs[r:r + 1]))
            rows_, state = policy.step_raw(full, state)
            worst_state = max(worst_state, float(np.abs(state[0, 0].numpy() - acts[r, 512:640]).max()))
            worst_value = max(worst_value, abs(float(rows_[0, -1]) - float(acts[r, 768])))
            if meta[r, ctrl_col] == 1 and np.isfinite(labels[r, value_col]):
                worst_online = max(worst_online, abs(float(labels[r, value_col]) - float(acts[r, 768])))
    return dict(max_state_diff_vs_native=worst_state, max_value_diff_vs_native=worst_value,
                max_value_diff_vs_online_f32=worst_online)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=OUT)
    parser.add_argument("--checkpoint", type=Path, default=FINAL)
    parser.add_argument("--tag", default="024")
    parser.add_argument("--obs", default="obs", help="obs (delayed, what the policy saw) or obs_now")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--zero-pos", action="store_true", help="zero obs[8:10] (absolute self position) before the encoder")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if str(args.checkpoint) == "early":
        from generate import EARLY
        args.checkpoint = EARLY
    run(args.dataset, args.checkpoint, args.tag, args.device, args.obs, zero_pos=args.zero_pos)
    if args.check:
        print(json.dumps(check(args.dataset, args.checkpoint, args.tag)))


if __name__ == "__main__":
    main()
