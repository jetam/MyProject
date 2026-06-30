import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

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
            tokens = song  # already processed externally

            for i in range(0, len(tokens) - seq_len - 1, seq_len):
                chunk = tokens[i:i+seq_len+1]
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
            # u = torch.tensor([t[3] for t in seq], dtype=torch.long)
            return p, v, d

        return split(x), split(y)


# ============================================================
# POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ============================================================
# MODEL (FIXED ARCHITECTURE)
# ============================================================

class MusicTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()

        self.pitch_emb = nn.Embedding(PITCH_VOCAB, d_model)
        self.vel_emb   = nn.Embedding(VEL_VOCAB, d_model)
        self.dt_emb    = nn.Embedding(DT_VOCAB, d_model)
        # self.dur_emb   = nn.Embedding(DUR_VOCAB, d_model)

        self.pos = PositionalEncoding(d_model)

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

        self.out_pitch = nn.Linear(d_model, PITCH_VOCAB)
        self.out_vel   = nn.Linear(d_model, VEL_VOCAB)
        self.out_dt    = nn.Linear(d_model, DT_VOCAB)
        # self.out_dur   = nn.Linear(d_model, DUR_VOCAB)

    def embed(self, p, v, d):
        return self.pos(
            self.pitch_emb(p)
            + self.vel_emb(v)
            + self.dt_emb(d)
            # + self.dur_emb(u)
        )

    def causal_mask(self, T, device):
        # TRUE causal mask for TransformerEncoder
        mask = torch.triu(
            torch.ones(T, T, device=device),
            diagonal=1
        ).bool()
        return mask

    def forward(self, p, v, d):
        x = self.embed(p, v, d)
        B, T, _ = x.shape

        mask = self.causal_mask(T, x.device)

        x = self.transformer(x, mask=mask)

        return (
            self.out_pitch(x),
            self.out_vel(x),
            self.out_dt(x),
            # self.out_dur(x),
        )


# ============================================================
# LOSS
# ============================================================

def loss_fn(logits, targets):
    lp, lv, ld = logits
    tp, tv, td = targets

    return (
        F.cross_entropy(lp.reshape(-1, PITCH_VOCAB), tp.reshape(-1)) +
        F.cross_entropy(lv.reshape(-1, VEL_VOCAB), tv.reshape(-1)) +
        F.cross_entropy(ld.reshape(-1, DT_VOCAB), td.reshape(-1))
        # F.cross_entropy(lu.reshape(-1, DUR_VOCAB), tu.reshape(-1))
    )


# ============================================================
# TRAIN
# ============================================================

def train(model, songs, epochs=5, batch_size=8, lr=3e-4):
    print( "training..." )
    model = model.to(DEVICE)

    dataset = MusicDataset(songs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    epochs = 1 # todo: test delete !!!

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    print( "training..." )
    for epoch in range(epochs):
        # print("train1")
        total = 0

        for (x, y) in loader:
            # print( "train2" )
            xp, xv, xd = [t.to(DEVICE) for t in x]
            yp, yv, yd = [t.to(DEVICE) for t in y]

            logits = model(xp, xv, xd)
            loss = loss_fn(logits, (yp, yv, yd))

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
    print( "compose" )
    model.eval()

    def to_tensor(song):
        p = torch.tensor([n[0] for n in song], dtype=torch.long, device=DEVICE)
        v = torch.tensor([n[1] for n in song], dtype=torch.long, device=DEVICE)
        d = torch.tensor([n[2] for n in song], dtype=torch.long, device=DEVICE)
        # u = torch.tensor([n[3] for n in song], dtype=torch.long, device=DEVICE)
        return p, v, d

    p, v, d = to_tensor(seedSong)

    p, v, d = p.unsqueeze(0), v.unsqueeze(0), d.unsqueeze(0)

    for _ in range(length):
        lp, lv, ld = model(p, v, d)

        def sample(logits):
            logits = logits[:, -1] / temperature
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1)

        p = torch.cat([p, sample(lp)], dim=1)
        v = torch.cat([v, sample(lv)], dim=1)
        d = torch.cat([d, sample(ld)], dim=1)
        # u = torch.cat([u, sample(lu)], dim=1)

    p = p[0].tolist()
    v = v[0].tolist()
    d = d[0].tolist()
    # u = u[0].tolist()

    notes = list(zip(p, v, d))

    return notes

# todo: also try:
# todo: Music Transformer  // use googles model
# use tokens in stead of lists
