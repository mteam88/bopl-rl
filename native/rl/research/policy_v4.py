"""Policy v4 reference forward pass in torch (POLICY_V4_PLAN.md section 3).

The oracle for the CUDA trainer (puffer/check_v4_encoder.py) and the CPU
evaluator (puffer/policy_eval.c, through native_policy.py): one flat float
buffer in layout.policy_v4_tensors order, one decision step at a time, the
decoder row [aim block(s), buttons, hold, value] exactly as v3 emits it.

    flat   = ReLU(W_flat  obs[:V4_FLAT])
    row_s  = ReLU(W2_s ReLU(W1_s x_s + E[id_s]))      per valid row (id sets only get E)
    pool_s = [max_rows row_s, mean_rows row_s]         zero when the set is empty
    u      = LN(W_fusion [flat, pool_player, pool_slot, pool_entity, pool_slime])
    h      = MinGRU(u)                                  upstream cell, highway gate
    y      = h + W_ffn_out ReLU(W_ffn_in LN(h))          when ffn_dim > 0
    logits = W_actor y
    z      = W_critic_in LN([y (detached when critic_detach), privileged])
    value  = W_critic_out LN(z + W_critic_b2 ReLU(W_critic_b1 LN(z)))

LayerNorm has no affine parameters and eps 1e-5. No biases anywhere.
"""
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layout  # noqa: E402

LN_EPS = 1e-5
SETS_WITH_EMBEDDING = ("slot", "entity")


def layer_norm(x):
    return F.layer_norm(x, x.shape[-1:], eps=LN_EPS)


def mingru_step(x, state, proj):
    """Upstream PufferNet MinGRU cell (puffernet.h mingru): returns (output, new_state)."""
    combined = x @ proj.t()
    hidden, gate, highway = combined.chunk(3, -1)
    h_tilde = torch.where(hidden >= 0, hidden + 0.5, torch.sigmoid(hidden))
    new_state = state + torch.sigmoid(gate) * (h_tilde - state)
    hw = torch.sigmoid(highway)
    return hw * new_state + (1 - hw) * x, new_state


def fourier_aim_matrix(harmonics, dtype=torch.float32):
    """(ACT_AIM, 1 + 2K): game aim logits = block @ matrix.T (bopl_aim.h bopl_aim_fourier_logit)."""
    m = torch.zeros(layout.ACT_AIM, 1 + 2 * harmonics, dtype=torch.float64)
    m[0, 0] = 1
    theta = 2 * np.pi * torch.arange(layout.ACT_AIM - 1, dtype=torch.float64) / 255.0
    for k in range(1, harmonics + 1):
        m[1:, 2 * k - 1] = torch.cos(k * theta)
        m[1:, 2 * k] = torch.sin(k * theta)
    m[1:] /= np.sqrt(harmonics)
    return m.to(dtype)


class PolicyV4:
    """Views of one flat weight buffer plus the forward pass. `weights` may be a
    numpy array or a torch tensor; pass a tensor with requires_grad to take
    gradients through the flat buffer."""

    def __init__(self, weights, hidden, num_layers=1, aim_fourier=4, aim_ar=0, **shape):
        self.hidden, self.num_layers = int(hidden), int(num_layers)
        self.aim_fourier, self.aim_ar = int(aim_fourier), int(aim_ar)
        self.shape = dict(layout.V4_DEFAULTS)
        self.shape.update({k: int(v) for k, v in shape.items() if k in self.shape})
        self.width = layout.decoder_logits(self.aim_fourier, self.aim_ar)
        self.tensors = layout.policy_v4_tensors(self.hidden, self.num_layers, self.aim_fourier, self.aim_ar, **self.shape)
        expected = layout.policy_v4_weights(self.hidden, self.num_layers, aim_fourier=self.aim_fourier,
                                            aim_ar=self.aim_ar, **self.shape)
        flat = weights if torch.is_tensor(weights) else torch.from_numpy(np.ascontiguousarray(weights, dtype=np.float32))
        if flat.numel() != expected:
            raise ValueError(f"policy v4 expects {expected} weights, got {flat.numel()}")
        self.flat = flat
        self.w, offset = {}, 0
        for name, (rows, cols) in self.tensors:
            self.w[name] = flat[offset:offset + rows * cols].view(rows, cols)
            offset += layout.aligned(rows * cols)
        self.obs_size = layout.V4_OBS_SIZE

    @property
    def state_shape(self):
        return (self.num_layers, self.hidden)

    def initial(self, agents):
        return torch.zeros(self.num_layers, agents, self.hidden, dtype=self.flat.dtype)

    def encode_set(self, obs, name, offset, count, width, id_column):
        rows = obs[:, offset:offset + count * width].reshape(obs.shape[0], count, width)
        valid = rows[:, :, 0] != 0
        if id_column is not None:
            ids = rows[:, :, id_column]
            ids_int = ids.long().clamp(0, layout.EMBEDDING_IDS - 1)
            ids_int = torch.where(ids == ids_int.to(ids.dtype), ids_int, torch.zeros_like(ids_int))
            x = rows[:, :, :id_column]
        else:
            x = rows
        h = x @ self.w[f"{name}_w1"].t()
        if name in SETS_WITH_EMBEDDING:
            h = h + self.w["embedding"][ids_int]
        h = torch.relu(h)
        h = torch.relu(h @ self.w[f"{name}_w2"].t())
        h = h * valid.unsqueeze(-1).to(h.dtype)
        pooled_max = h.max(1).values  # rows are >= 0 and invalid rows are 0, so an empty set pools to 0
        n = valid.sum(1, keepdim=True).to(h.dtype).clamp(min=1)
        pooled_mean = h.sum(1) / n
        if not self.shape["mean_pool"]:
            pooled_mean = torch.zeros_like(pooled_mean)
        return torch.cat([pooled_max, pooled_mean], -1)

    def trunk(self, obs, state):
        """(y, new_state): the actor trunk output and the recurrent state."""
        flat = torch.relu(obs[:, :layout.V4_FLAT] @ self.w["flat"].t())
        pools = [self.encode_set(obs, *spec) for spec in layout.V4_SETS]
        u = layer_norm(torch.cat([flat, *pools], -1) @ self.w["fusion"].t())
        x, new_state = u, []
        for layer in range(self.num_layers):
            x, s = mingru_step(x, state[layer], self.w[f"mingru{layer}"])
            new_state.append(s)
        y = x
        if self.shape["ffn_dim"]:
            y = y + torch.relu(layer_norm(y) @ self.w["ffn_in"].t()) @ self.w["ffn_out"].t()
        return y, torch.stack(new_state)

    def value(self, y, obs):
        if not self.shape["critic_dim"]:
            return torch.zeros(y.shape[0], 1, dtype=y.dtype)
        z_in = y.detach() if self.shape["critic_detach"] else y
        if self.shape["critic_privileged"]:
            z_in = torch.cat([z_in, obs[:, layout.V4_PRIVILEGED_OFFSET:layout.V4_PRIVILEGED_OFFSET + layout.V4_PRIVILEGED]], -1)
        z = layer_norm(z_in) @ self.w["critic_in"].t()
        z = z + torch.relu(layer_norm(z) @ self.w["critic_b1"].t()) @ self.w["critic_b2"].t()
        return layer_norm(z) @ self.w["critic_out"].t()

    def step_raw(self, obs, state):
        """(rows (B, width + 1), new_state): the raw decoder rows, logits then value."""
        if obs.shape[1] != self.obs_size:
            raise ValueError(f"policy v4 reads {self.obs_size} observation floats, got {obs.shape[1]}")
        y, new_state = self.trunk(obs, state)
        logits = y @ self.w["actor"].t()
        return torch.cat([logits, self.value(y, obs)], -1), new_state

    def game_logits(self, rows, buttons=None):
        """(logits (B, LOGITS), value (B,)) of raw rows for a button combo (bopl_aim.h)."""
        batch = rows.shape[0]
        block_width = layout.aim_block_width(self.aim_fourier)
        combos = torch.zeros(batch, dtype=torch.long) if buttons is None else torch.as_tensor(buttons).long()
        starts = (combos if self.aim_ar else torch.zeros_like(combos)) * block_width
        index = starts.unsqueeze(1) + torch.arange(block_width).unsqueeze(0)
        block = rows.gather(1, index)
        if self.aim_fourier:
            aim = block @ fourier_aim_matrix(self.aim_fourier, rows.dtype).t()
        else:
            aim = block
        rest_offset = layout.decoder_buttons_offset(self.aim_fourier, self.aim_ar)
        rest = rows[:, rest_offset:rest_offset + layout.ACT_BUTTONS + layout.ACT_HOLD]
        return torch.cat([aim, rest], -1), rows[:, self.width]

    def step(self, obs, state):
        rows, new_state = self.step_raw(obs, state)
        logits, value = self.game_logits(rows)
        return logits, value, new_state


def init_weights(hidden, num_layers=1, aim_fourier=4, aim_ar=0, seed=0, **shape):
    """Kaiming-uniform-ish flat buffer (fan-in scaled), for tests; the trainer has its own init."""
    rng = np.random.default_rng(seed)
    full = dict(layout.V4_DEFAULTS); full.update(shape)
    tensors = layout.policy_v4_tensors(hidden, num_layers, aim_fourier, aim_ar, **full)
    out = np.zeros(layout.policy_v4_weights(hidden, num_layers, aim_fourier=aim_fourier, aim_ar=aim_ar, **full), np.float32)
    offset = 0
    for name, (rows, cols) in tensors:
        bound = 1.0 / np.sqrt(cols) if name != "embedding" else 0.1
        out[offset:offset + rows * cols] = rng.uniform(-bound, bound, rows * cols)
        offset += layout.aligned(rows * cols)
    return out
