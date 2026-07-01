
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

PITCH_CLASS_VOCAB = 12
OCTAVE_VOCAB = 11  # adjust if your dataset differs
PITCH_VOCAB = 128
VEL_VOCAB = 9
DT_VOCAB = 17
# DUR_VOCAB = 17
DUR_VOCAB = 2

# Max quantized values that transformer takes
MAX_PITCH = PITCH_VOCAB - 1
MAX_VELOCITY = VEL_VOCAB - 1
MAX_TIME = DT_VOCAB - 1
MAX_DURATION = DUR_VOCAB - 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



# ============================================================
# DATASET
# ============================================================

class MusicDataset(Dataset):
    def __init__(self, songs, seq_len=256):
        self.seq_len = seq_len
        self.data = []

        for song in songs:
            for i in range(0, len(song) - seq_len - 1, seq_len):
                chunk = song[i:i + seq_len + 1]
                if len(chunk) == seq_len + 1:
                    self.data.append(chunk)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        x = chunk[:-1]
        y = chunk[1:]

        def split(seq):
            p = torch.tensor([t[0] for t in seq], dtype=torch.long)
            v = torch.tensor([t[1] for t in seq], dtype=torch.long)
            d = torch.tensor([t[2] for t in seq], dtype=torch.long)

            pc = p % 12
            po = p // 12

            # ====================================================
            # TRUE RELATIVE PITCH (THIS IS THE FIX)
            # ====================================================
            rel = torch.zeros_like(p)
            rel[1:] = p[1:] - p[:-1]

            # clamp to embedding range
            rel = torch.clamp(rel + 64, 0, 127)

            return pc, po, v, d, rel

        return split(x), split(y)


# ============================================================
# MODEL
# ============================================================

class MusicTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()

        self.pitch_class_emb = nn.Embedding(PITCH_CLASS_VOCAB, d_model)
        self.pitch_oct_emb = nn.Embedding(OCTAVE_VOCAB, d_model)

        self.vel_emb = nn.Embedding(VEL_VOCAB, d_model)
        self.dt_emb = nn.Embedding(DT_VOCAB, d_model)

        # relative pitch embedding (now meaningful)
        self.rel_emb = nn.Embedding(128, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fuse = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        self.out_pc = nn.Linear(d_model, PITCH_CLASS_VOCAB)
        self.out_po = nn.Linear(d_model, OCTAVE_VOCAB)
        self.out_v = nn.Linear(d_model, VEL_VOCAB)
        self.out_d = nn.Linear(d_model, DT_VOCAB)

    def embed(self, pc, po, v, d, rel):
        pitch = self.pitch_class_emb(pc) + self.pitch_oct_emb(po)

        r = self.rel_emb(rel)

        x = torch.cat([
            pitch,
            self.vel_emb(v),
            self.dt_emb(d),
            r
        ], dim=-1)

        return self.fuse(x)

    def causal_mask(self, T, device):
        return torch.triu(
            torch.ones(T, T, device=device),
            diagonal=1
        ).bool()

    def forward(self, pc, po, v, d, rel):
        x = self.embed(pc, po, v, d, rel)
        B, T, _ = x.shape

        mask = self.causal_mask(T, x.device)
        x = self.transformer(x, mask=mask)

        return (
            self.out_pc(x),
            self.out_po(x),
            self.out_v(x),
            self.out_d(x),
        )


# ============================================================
# LOSS
# ============================================================

def loss_fn(logits, targets):
    lpc, lpo, lv, ld = logits
    tpc, tpo, tv, td = targets

    return (
        F.cross_entropy(lpc.reshape(-1, PITCH_CLASS_VOCAB), tpc.reshape(-1)) +
        F.cross_entropy(lpo.reshape(-1, OCTAVE_VOCAB), tpo.reshape(-1)) +
        F.cross_entropy(lv.reshape(-1, VEL_VOCAB), tv.reshape(-1)) +
        F.cross_entropy(ld.reshape(-1, DT_VOCAB), td.reshape(-1))
    )


# ============================================================
# TRAIN
# ============================================================

def train(model, songs, epochs=5, batch_size=8, lr=3e-4):
    model = model.to(DEVICE)

    dataset = MusicDataset(songs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total = 0

        for (x, y) in loader:
            xpc, xpo, xv, xd, xrel = [t.to(DEVICE) for t in x]
            ypc, ypo, yv, yd, _ = [t.to(DEVICE) for t in y]

            logits = model(xpc, xpo, xv, xd, xrel)
            loss = loss_fn(logits, (ypc, ypo, yv, yd))

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"Epoch {epoch+1} | loss {total:.4f}")

    return model


# ============================================================
# GENERATION
# ============================================================

@torch.no_grad()
def compose(model, seedSong, length=200, temperature=1.0):
    model.eval()

    def to_tensor(song):
        p = torch.tensor([n[0] for n in song], dtype=torch.long, device=DEVICE)
        v = torch.tensor([n[1] for n in song], dtype=torch.long, device=DEVICE)
        d = torch.tensor([n[2] for n in song], dtype=torch.long, device=DEVICE)

        pc = p % 12
        po = p // 12
        return pc, po, v, d, p

    pc, po, v, d, p = to_tensor(seedSong)

    pc = pc.unsqueeze(0)
    po = po.unsqueeze(0)
    v = v.unsqueeze(0)
    d = d.unsqueeze(0)
    p = p.unsqueeze(0)

    for _ in range(length):

        # TRUE relative pitch (this is the key fix)
        rel = torch.zeros_like(p)
        rel[:, 1:] = p[:, 1:] - p[:, :-1]
        rel = torch.clamp(rel + 64, 0, 127)

        lpc, lpo, lv, ld = model(pc, po, v, d, rel)

        def sample(logits):
            logits = logits[:, -1] / temperature
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1)

        new_pc = sample(lpc)
        new_po = sample(lpo)
        new_v = sample(lv)
        new_d = sample(ld)

        new_p = new_pc + new_po * 12

        pc = torch.cat([pc, new_pc], dim=1)
        po = torch.cat([po, new_po], dim=1)
        v = torch.cat([v, new_v], dim=1)
        d = torch.cat([d, new_d], dim=1)
        p = torch.cat([p, new_p], dim=1)

    p = p.squeeze(0).tolist()
    v = v.squeeze(0).tolist()
    d = d.squeeze(0).tolist()

    return list(zip(p, v, d))