
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from ..services import midi_parser as Parser
from ..services import midi_tester as midi_tester
import os

from .base_model import BaseMusicModel, SEED_NOTES


MODEL_DIR = "./music/trained_models/transformer2"
MODEL_NUM = 1

os.makedirs(MODEL_DIR, exist_ok=True)


from .music_config import (
    PITCH_CLASS_VOCAB, OCTAVE_VOCAB, PITCH_VOCAB, VEL_VOCAB, DT_VOCAB, DUR_VOCAB,
    MAX_PITCH, MAX_VELOCITY, MAX_TIME, MAX_DURATION,
)

# REMI-style token layout — each note produces 3 tokens in sequence
PAD       = 0
BOS       = 1
PITCH_OFF = 2                          # tokens  2..129  (128 pitches)
VEL_OFF   = PITCH_OFF + PITCH_VOCAB    # tokens 130..138  (9 velocities)
DT_OFF    = VEL_OFF + VEL_VOCAB        # tokens 139..202  (64 dt values)
VOCAB_SIZE = DT_OFF + DT_VOCAB

# Token type IDs
T_PITCH, T_VEL, T_DT = 0, 1, 2
TYPE_CYCLE = [T_PITCH, T_VEL, T_DT]

# Valid token index range per type — used to mask sampling
VALID_RANGE = {
    T_PITCH: (PITCH_OFF, PITCH_OFF + PITCH_VOCAB),
    T_VEL:   (VEL_OFF,   VEL_OFF   + VEL_VOCAB),
    T_DT:    (DT_OFF,    DT_OFF    + DT_VOCAB),
}

MAX_SEQ_LEN = 1024  # tokens (~341 notes)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def encode_note(pitch: int, vel: int, dt: int):
    """One note → 3 REMI tokens."""
    return [PITCH_OFF + pitch, VEL_OFF + vel, DT_OFF + dt]


def decode_note(pitch_tok: int, vel_tok: int, dt_tok: int):
    return (pitch_tok - PITCH_OFF, vel_tok - VEL_OFF, dt_tok - DT_OFF)


class MusicDataset(Dataset):
    def __init__(self, songs, notes_per_chunk=64):
        self.notes_per_chunk = notes_per_chunk
        self.data = []

        for song in songs:
            step = notes_per_chunk
            for i in range(0, len(song) - notes_per_chunk - 1, step):
                chunk = song[i:i + notes_per_chunk + 1]
                if len(chunk) == notes_per_chunk + 1:
                    self.data.append(chunk)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = list(self.data[idx])

        # pitch transposition augmentation
        shift = random.randint(-6, 6)
        if shift != 0:
            chunk = [(max(0, min(127, p + shift)), v, d) for p, v, d in chunk]

        # tokenize: (notes_per_chunk+1) notes → (notes_per_chunk+1)*3 tokens
        toks = []
        for note in chunk:
            toks.extend(encode_note(*note))

        # language-model shift: x predicts y
        x = torch.tensor(toks[:-1], dtype=torch.long)
        y = torch.tensor(toks[1:], dtype=torch.long)

        # token type for every position in x (always aligned to note boundary)
        x_types = torch.tensor([TYPE_CYCLE[i % 3] for i in range(len(x))], dtype=torch.long)

        return (x, x_types), y


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps   = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.scale * x * rms


class SwiGLU(nn.Module):
    def __init__(self, d_model, ff_mult=4):
        super().__init__()
        # standard SwiGLU hidden size: d * 8/3, rounded to nearest multiple of 64
        hidden = ((int(d_model * ff_mult * 2 / 3) + 63) // 64) * 64
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_in   = nn.Linear(d_model, hidden, bias=False)
        self.w_out  = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.w_out(F.silu(self.w_gate(x)) * self.w_in(x))


def _precompute_rope(d_k: int, max_len: int, theta: float = 10000.0):
    half   = d_k // 2
    freqs  = 1.0 / (theta ** (torch.arange(0, half).float() / half))
    t      = torch.arange(max_len).float()
    freqs  = torch.outer(t, freqs)              # [max_len, half]
    cos    = torch.cat([freqs.cos(), freqs.cos()], dim=-1)  # [max_len, d_k]
    sin    = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos, sin


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _apply_rope(q, k, cos, sin):
    T   = q.shape[2]
    c   = cos[:T].unsqueeze(0).unsqueeze(0)  # [1, 1, T, d_k]
    s   = sin[:T].unsqueeze(0).unsqueeze(0)
    q   = q * c + _rotate_half(q) * s
    k   = k * c + _rotate_half(k) * s
    return q, k


class RoPEAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead     = nhead
        self.d_k       = d_model // nhead
        self.dropout_p = dropout

        self.q   = nn.Linear(d_model, d_model, bias=False)
        self.k   = nn.Linear(d_model, d_model, bias=False)
        self.v   = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        cos, sin = _precompute_rope(self.d_k, MAX_SEQ_LEN)
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)

    def forward(self, x):
        B, T, _ = x.shape
        H, D    = self.nhead, self.d_k

        Q = self.q(x).view(B, T, H, D).permute(0, 2, 1, 3)  # [B, H, T, D]
        K = self.k(x).view(B, T, H, D).permute(0, 2, 1, 3)
        V = self.v(x).view(B, T, H, D).permute(0, 2, 1, 3)

        Q, K = _apply_rope(Q, K, self.rope_cos, self.rope_sin)

        # Flash Attention when available (PyTorch 2.0+), falls back gracefully
        drop = self.dropout_p if self.training else 0.0
        out  = F.scaled_dot_product_attention(Q, K, V, is_causal=True, dropout_p=drop)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, H * D)
        return self.out(out)


class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attn  = RoPEAttention(d_model, nhead, dropout)
        self.ff    = SwiGLU(d_model)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class MusicTransformerT2(BaseMusicModel):
    def __init__(self, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()

        self.tok_emb  = nn.Embedding(VOCAB_SIZE, d_model)
        self.type_emb = nn.Embedding(3, d_model)

        self.layers = nn.ModuleList([
            TransformerLayer(d_model, nhead, dropout) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)

        self.out = nn.Linear(d_model, VOCAB_SIZE, bias=False)
        self.out.weight = self.tok_emb.weight

        self._init_weights(num_layers)

    def _init_weights(self, num_layers):
        std = 0.02
        for name, p in self.named_parameters():
            if p.dim() < 2 or 'norm' in name:
                continue
            # scale residual projections by 1/sqrt(2*L) (GPT-2 style)
            if 'out' in name and ('attn' in name or 'ff' in name):
                nn.init.normal_(p, std=std / math.sqrt(2 * num_layers))
            else:
                nn.init.normal_(p, std=std)

    def forward(self, tokens, types):
        x = self.tok_emb(tokens) + self.type_emb(types)
        for layer in self.layers:
            x = layer(x)
        return self.out(self.norm(x))   # [B, T, VOCAB_SIZE]

    def fineTune(self, song):
        fineTune(self, song)
        return self

    def generate(self, seedSong, length=200):
        return compose(self, seedSong, length=length)



def loss_fn(logits, targets):
    return F.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE),
        targets.reshape(-1),
        label_smoothing=0.1,
        ignore_index=PAD,
    )


def _make_optimizer(model, lr, weight_decay=0.1):
    """AdamW with weight decay only on matrices (not norms, biases, embeddings)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2 or 'norm' in name or 'bias' in name or 'emb' in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{'params': decay, 'weight_decay': weight_decay},
         {'params': no_decay, 'weight_decay': 0.0}],
        lr=lr, betas=(0.9, 0.95), eps=1e-8,
    )


def train(model, songs, epochs=5, batch_size=8, lr=3e-4, warmup_steps=500):
    use_amp  = torch.cuda.is_available()
    model    = model.to(DEVICE)
    dataset  = MusicDataset(songs)
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=use_amp)
    opt      = _make_optimizer(model, lr)
    scaler   = torch.amp.GradScaler('cuda', enabled=use_amp)

    total_steps = epochs * len(loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    for epoch in range(epochs):
        total = 0

        for (x_tok, x_types), y in loader:
            x_tok   = x_tok.to(DEVICE)
            x_types = x_types.to(DEVICE)
            y       = y.to(DEVICE)

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(x_tok, x_types)
                loss   = loss_fn(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            scheduler.step()

            total += loss.item()

        print(f"tr2 Epoch {epoch+1} | loss {total:.4f} | lr {scheduler.get_last_lr()[0]:.2e}")

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"pretrained{MODEL_NUM}.pt"))

    return model


def loadModel():
    model_path = os.path.join(MODEL_DIR, f"pretrained{MODEL_NUM}.pt")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"No trained TransformerT2 model found at {model_path}")

    model = MusicTransformerT2()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model


def fineTune(model, song, notes_per_chunk=64, epochs=2, batch_size=4, lr=1e-5):

    use_amp = torch.cuda.is_available()
    model = model.to(DEVICE)
    model.train()

    # dataset built from ONLY this song
    dataset = MusicDataset([song], notes_per_chunk=notes_per_chunk)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=use_amp)

    # smaller, constant LR than pretraining; no warmup/cosine schedule for a short run
    opt = _make_optimizer(model, lr)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    for epoch in range(epochs):
        total = 0

        for (x_tok, x_types), y in loader:
            x_tok = x_tok.to(DEVICE)
            x_types = x_types.to(DEVICE)
            y = y.to(DEVICE)

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(x_tok, x_types)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            total += loss.item()

        print(f"[fine-tune] epoch {epoch+1} | loss {total:.4f}")

    return model


def _nucleus_sample(logits, temperature, top_p):
    probs = F.softmax(logits / temperature, dim=-1)
    sorted_p, sorted_i = torch.sort(probs, descending=True)
    cs = sorted_p.cumsum(0)
    sorted_p[cs - sorted_p > top_p] = 0.0
    sorted_p /= sorted_p.sum()
    return sorted_i[torch.multinomial(sorted_p, 1)].item()


@torch.no_grad()
def compose(model, seedSong, length=200, temperature=1.0, top_p=0.9, rep_penalty=1.2):
    model.eval()

    seedSong = seedSong[:SEED_NOTES]

    tokens_per_note = len(TYPE_CYCLE)
    total_tokens = (len(seedSong) + length) * tokens_per_note
    if total_tokens > MAX_SEQ_LEN:
        raise ValueError(
            f"compose() would need {total_tokens} tokens "
            f"({len(seedSong)} seed notes + {length} generated notes) x {tokens_per_note} tokens/note, "
            f"but model max is {MAX_SEQ_LEN} tokens. Reduce length or SEED_NOTES."
        )

    # encode seed notes into token sequence
    seed_toks = []
    for note in seedSong:
        seed_toks.extend(encode_note(*note))

    tokens = torch.tensor(seed_toks, dtype=torch.long, device=DEVICE).unsqueeze(0)

    # track recent pitches for repetition penalty
    recent_pitches = [n[0] for n in seedSong[-32:]]

    generated_notes = []

    for _ in range(length):
        note_toks = []

        for tok_type in TYPE_CYCLE:   # generate PITCH → VEL → DT
            T = tokens.shape[1]
            types_t = torch.tensor(
                [TYPE_CYCLE[i % 3] for i in range(T)],
                dtype=torch.long, device=DEVICE
            ).unsqueeze(0)

            logits = model(tokens, types_t)[0, -1].clone()  # [VOCAB_SIZE]

            # mask out tokens that don't belong to this type
            lo, hi = VALID_RANGE[tok_type]
            type_mask = torch.full((VOCAB_SIZE,), float('-inf'), device=DEVICE)
            type_mask[lo:hi] = 0.0
            logits = logits + type_mask

            # repetition penalty on pitch tokens only
            if tok_type == T_PITCH:
                for p in set(recent_pitches):
                    logits[PITCH_OFF + p] /= rep_penalty

            tok = _nucleus_sample(logits, temperature, top_p)
            note_toks.append(tok)
            tokens = torch.cat([tokens, torch.tensor([[tok]], device=DEVICE)], dim=1)

        pitch, vel, dt = decode_note(*note_toks)
        generated_notes.append((pitch, vel, dt))
        recent_pitches = (recent_pitches + [pitch])[-32:]

    return list(seedSong) + generated_notes


def trainModel(songs):
    model = MusicTransformerT2()
    train(model, songs, epochs=5)

    return model


def composeMusic(seedSong):
    parser = Parser.MidiParser(MAX_VELOCITY, MAX_TIME, MAX_DURATION)

    model = loadModel()
    model = fineTune(model, seedSong, epochs=2, lr=1e-5)

    generated = compose(model, seedSong, length=200)
    generatedNotes = parser.convertedNotes(generated)
    midi_tester.testMidi(generatedNotes, "midiTransformerT2.mid")