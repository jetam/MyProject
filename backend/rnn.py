
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

DT_VOCAB = 17
VEL_VOCAB = 9
SEQUENCE_LENGTH = 64


# class MusicDataset(Dataset):
#     def __init__(self, songs, seq_len):
#         self.songs = songs
#         self.seq_len = seq_len
#
#     def __len__(self):
#         return len(self.songs) * 50  # samples per song
#
#     def __getitem__(self, idx):
#         song = self.songs[idx % len(self.songs)]
#
#         start = torch.randint(0, len(song) - self.seq_len, (1,)).item()
#         seq = song[start:start + self.seq_len]
#
#         notes = torch.tensor([n[0] for n in seq], dtype=torch.long)
#         others = torch.tensor([[n[1], n[2]] for n in seq], dtype=torch.long)
#
#         return notes, others

class MusicDataset(Dataset):

    def __init__(self, songs, seq_len):
        self.songs = songs
        self.seq_len = seq_len

    def __len__(self):
        return len(self.songs) * 100

    def __getitem__(self, idx):
        song = self.songs[idx % len(self.songs)]

        start = torch.randint(0, len(song) - self.seq_len, (1,)).item()
        seq = song[start:start + self.seq_len]

        # return PURE PYTHON LISTS (fast)
        notes = [n[0] for n in seq]
        others = [(n[1], n[2]) for n in seq]

        return notes, others

def collate_fn(batch):
    notes, others = zip(*batch)

    notes = torch.tensor(notes, dtype=torch.long)

    others = torch.tensor(others, dtype=torch.long)

    return notes, others


class MusicRNN(nn.Module):

    def __init__(
        self,
        num_notes=128,
        embed_dim=32,
        vel_vocab=VEL_VOCAB,
        dt_vocab=DT_VOCAB,
        hidden_size=256,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.note_emb = nn.Embedding(num_notes, embed_dim)
        self.vel_emb = nn.Embedding(vel_vocab, 8)
        self.dt_emb = nn.Embedding(dt_vocab, 8)

        self.dropout = nn.Dropout(dropout)   # 👈 NEW

        self.event_proj = nn.Linear(embed_dim + 16, hidden_size)

        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.note_head = nn.Linear(hidden_size, num_notes)
        self.vel_head = nn.Linear(hidden_size, vel_vocab)
        self.dt_head = nn.Linear(hidden_size, dt_vocab)

    def forward(self, notes, others):

        vel = others[:, :, 0].long()
        dt  = others[:, :, 1].long()

        x = torch.cat([
            self.note_emb(notes),
            self.vel_emb(vel),
            self.dt_emb(dt)
        ], dim=-1)

        x = self.dropout(x)

        x = self.event_proj(x)

        x = self.dropout(x)

        out, _ = self.rnn(x)

        out = self.dropout(out)

        return (
            self.note_head(out),
            self.vel_head(out),
            self.dt_head(out)
        )

    def step(self, note, vel, dt, hidden=None):

        note = note.view(-1)
        vel  = vel.view(-1)
        dt   = dt.view(-1)

        note_x = self.note_emb(note)
        vel_x  = self.vel_emb(vel)
        dt_x   = self.dt_emb(dt)

        x = torch.cat([note_x, vel_x, dt_x], dim=-1)

        x = self.dropout(x)          # 👈 SAME IDEA AS TRAINING PATH

        x = self.event_proj(x)

        x = x.unsqueeze(1)

        out, hidden = self.rnn(x, hidden)

        h = out[:, -1, :]

        return (
            self.note_head(h),
            self.vel_head(h),
            self.dt_head(h),
            hidden
        )

# =========================
# TRAINING (autoregressive)
# =========================
def train(model, dataloader, epochs=3, lr=1e-3):

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    note_loss_fn = nn.CrossEntropyLoss()
    vel_loss_fn = nn.CrossEntropyLoss()
    dt_loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0

        print("dataloader size:", len(dataloader))

        for notes, others in dataloader:

            note_logits, vel_logits, dt_logits = model(notes, others)

            loss_note = note_loss_fn(
                note_logits[:, :-1].reshape(-1, note_logits.size(-1)),
                notes[:, 1:].reshape(-1)
            )

            loss_vel = vel_loss_fn(
                vel_logits[:, :-1].reshape(-1, vel_logits.size(-1)),
                others[:, 1:, 0].reshape(-1).long()
            )

            loss_dt = dt_loss_fn(
                dt_logits[:, :-1].reshape(-1, dt_logits.size(-1)),
                others[:, 1:, 1].reshape(-1).long()
            )

            loss = 2 * loss_note + loss_vel + loss_dt

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | loss {total_loss:.4f}")


@torch.no_grad()
def compose(model, seed_notes, seed_others, length=100):

    model.eval()

    notes = seed_notes.tolist()
    others = seed_others.tolist()

    hidden = None
    generated = []

    # warmup
    for i in range(len(notes)):

        note = torch.tensor([notes[i]])
        vel  = torch.tensor([others[i][0]])
        dt   = torch.tensor([others[i][1]])

        _, _, _, hidden = model.step(note, vel, dt, hidden)

    # generate
    for _ in range(length):

        note = torch.tensor([notes[-1]])
        vel  = torch.tensor([others[-1][0]])
        dt   = torch.tensor([others[-1][1]])

        note_logits, vel_logits, dt_logits, hidden = model.step(
            note, vel, dt, hidden
        )

        def sample(logits):
            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1).item()

        next_note = sample(note_logits)
        next_vel  = sample(vel_logits)
        next_dt   = sample(dt_logits)

        generated.append((next_note, next_vel, next_dt))

        notes.append(next_note)
        others.append([next_vel, next_dt])

    return generated


# =========================
# ENTRY POINT
# =========================
def composeMusic(songs):

    dataset = MusicDataset(songs, SEQUENCE_LENGTH)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_fn)

    model = MusicRNN()

    train(model, loader, epochs=3) # todo

    seed_notes = torch.tensor(
        [n[0] for n in songs[0][:SEQUENCE_LENGTH]],
        dtype=torch.long
    )

    seed_others = torch.tensor(
        [[n[1], n[2]] for n in songs[0][:SEQUENCE_LENGTH]],
        dtype=torch.long
    )

    music = compose(model, seed_notes, seed_others, length=200)

    print(music[:10])
    return music


# improvements:
# 1. Embed velocity and time too (big improvement) done
# 2. Predict autoregressively during training done
# 3. Longer sequences done
# 4. Don't restart LSTM state during generation // done got slow training here
# 5. Add dropout done
# 6. Weight losses differently done
# 7. Teacher forcing schedule skip
# 8. Better pitch representation (octaves): todo
#     Split note into:
#     pitch_class(0–11)
#     octave(0–10 - ish)
#     Then embed both separately and concatenate.

# 11. Gradient clipping maybe implement, probs not?
# 12. Layer normalization implement
# 16. Consider predicting NOTE EVENTS instead of separate outputs) (try in seperate file)

# train on specific piece:
# Step 1: “prime” the model
# Step 2: generate from last state
# also Use “teacher forcing warm-up”