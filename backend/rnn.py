
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

DT_VOCAB = 17
VEL_VOCAB = 9
SEQUENCE_LENGTH = 64


# =========================
# DATASET
# =========================
class MusicDataset(Dataset):

    def __init__(self, songs, seq_len):
        self.songs = songs
        self.seq_len = seq_len

    def __len__(self):
        return len(self.songs) * 200  # slightly more signal

    def __getitem__(self, idx):
        song = self.songs[idx % len(self.songs)]

        # mix random + deterministic sampling (IMPORTANT)
        if torch.rand(1).item() < 0.7:
            start = torch.randint(0, len(song) - self.seq_len, (1,)).item()
        else:
            start = (idx * self.seq_len) % (len(song) - self.seq_len)

        seq = song[start:start + self.seq_len]

        notes = [n[0] for n in seq]
        others = [(n[1], n[2]) for n in seq]

        return notes, others


def collate_fn(batch):
    notes, others = zip(*batch)
    return (
        torch.tensor(notes, dtype=torch.long),
        torch.tensor(others, dtype=torch.long)
    )


# =========================
# MODEL
# =========================
class MusicRNN(nn.Module):

    def __init__(
        self,
        pitch_class_vocab=12,
        octave_vocab=11,
        vel_vocab=VEL_VOCAB,
        dt_vocab=DT_VOCAB,
        hidden_size=256,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.pc_emb = nn.Embedding(pitch_class_vocab, 16)
        self.oct_emb = nn.Embedding(octave_vocab, 16)
        self.vel_emb = nn.Embedding(vel_vocab, 8)
        self.dt_emb = nn.Embedding(dt_vocab, 8)

        self.dropout = nn.Dropout(dropout)

        self.event_proj = nn.Linear(48, hidden_size)

        self.rnn = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.pc_head = nn.Linear(hidden_size, pitch_class_vocab)
        self.oct_head = nn.Linear(hidden_size, octave_vocab)
        self.vel_head = nn.Linear(hidden_size, vel_vocab)
        self.dt_head = nn.Linear(hidden_size, dt_vocab)

    def forward(self, notes, others):

        pc = notes % 12
        octv = notes // 12

        vel = others[:, :, 0].long()
        dt  = others[:, :, 1].long()

        x = torch.cat([
            self.pc_emb(pc),
            self.oct_emb(octv),
            self.vel_emb(vel),
            self.dt_emb(dt)
        ], dim=-1)

        x = self.dropout(x)
        x = self.event_proj(x)
        x = self.dropout(x)

        out, _ = self.rnn(x)
        out = self.dropout(out)

        return (
            self.pc_head(out),
            self.oct_head(out),
            self.vel_head(out),
            self.dt_head(out)
        )


# =========================
# TRAINING
# =========================
def train(model, dataloader, epochs=3, lr=1e-3):

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total = 0.0

        for notes, others in dataloader:

            pc_logits, oct_logits, vel_logits, dt_logits = model(notes, others)

            pc = notes % 12
            octv = notes // 12

            loss_pc = ce(pc_logits[:, :-1].reshape(-1, 12), pc[:, 1:].reshape(-1))
            loss_oct = ce(oct_logits[:, :-1].reshape(-1, 11), octv[:, 1:].reshape(-1))

            loss_vel = ce(
                vel_logits[:, :-1].reshape(-1, vel_logits.size(-1)),
                others[:, 1:, 0].reshape(-1)
            )

            loss_dt = ce(
                dt_logits[:, :-1].reshape(-1, dt_logits.size(-1)),
                others[:, 1:, 1].reshape(-1)
            )

            loss = 2 * loss_pc + loss_oct + loss_vel + loss_dt

            opt.zero_grad()
            loss.backward()

            # IMPORTANT FIX
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            opt.step()

            total += loss.item()

        print(f"epoch {epoch} | loss {total:.4f}")


# =========================
# COMPOSE
# =========================
@torch.no_grad()
def compose(model, seed_notes, seed_others, length=100):

    model.eval()

    seq_notes = seed_notes.tolist()
    seq_others = seed_others.tolist()

    generated = []

    def sample(logits, temp=0.9):
        probs = torch.softmax(logits / temp, dim=-1)
        return torch.multinomial(probs, 1).item()

    for _ in range(length):

        n = torch.tensor([seq_notes], dtype=torch.long)
        o = torch.tensor([seq_others], dtype=torch.long)

        pc_logits, oct_logits, vel_logits, dt_logits = model(n, o)

        pc = sample(pc_logits[:, -1])
        octv = sample(oct_logits[:, -1])
        vel = sample(vel_logits[:, -1])
        dt = sample(dt_logits[:, -1])

        next_note = octv * 12 + pc

        generated.append((next_note, vel, dt))

        seq_notes.append(next_note)
        seq_others.append([vel, dt])

        # keep context stable
        if len(seq_notes) > SEQUENCE_LENGTH:
            seq_notes = seq_notes[-SEQUENCE_LENGTH:]
            seq_others = seq_others[-SEQUENCE_LENGTH:]

    return generated


# =========================
# ENTRY
# =========================
def composeMusic(songs):

    dataset = MusicDataset(songs, SEQUENCE_LENGTH)

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True
    )

    model = MusicRNN()

    train(model, loader, epochs=3)

    seed_notes = torch.tensor([n[0] for n in songs[0][:SEQUENCE_LENGTH]], dtype=torch.long)
    seed_others = torch.tensor([[n[1], n[2]] for n in songs[0][:SEQUENCE_LENGTH]], dtype=torch.long)

    return compose(model, seed_notes, seed_others, length=200)



# from torch.utils.data import Dataset, DataLoader
# import torch
# import torch.nn as nn
#
# DT_VOCAB = 17
# VEL_VOCAB = 9
# SEQUENCE_LENGTH = 64
#
#
# class MusicDataset(Dataset):
#
#     def __init__(self, songs, seq_len):
#         self.songs = songs
#         self.seq_len = seq_len
#
#     def __len__(self):
#         return len(self.songs) * 100
#
#     def __getitem__(self, idx):
#         song = self.songs[idx % len(self.songs)]
#
#         start = torch.randint(0, len(song) - self.seq_len, (1,)).item()
#         seq = song[start:start + self.seq_len]
#
#         notes = [n[0] for n in seq]
#         others = [(n[1], n[2]) for n in seq]
#
#         return notes, others
#
#
# def collate_fn(batch):
#     notes, others = zip(*batch)
#
#     return (
#         torch.tensor(notes, dtype=torch.long),
#         torch.tensor(others, dtype=torch.long)
#     )
#
#
# class MusicRNN(nn.Module):
#
#     def __init__(
#         self,
#         pitch_class_vocab=12,
#         octave_vocab=11,
#         vel_vocab=VEL_VOCAB,
#         dt_vocab=DT_VOCAB,
#         embed_dim=32,
#         hidden_size=256,
#         num_layers=2,
#         dropout=0.1,
#     ):
#         super().__init__()
#
#         self.pc_emb = nn.Embedding(pitch_class_vocab, 16)
#         self.oct_emb = nn.Embedding(octave_vocab, 16)
#
#         self.vel_emb = nn.Embedding(vel_vocab, 8)
#         self.dt_emb = nn.Embedding(dt_vocab, 8)
#
#         self.dropout = nn.Dropout(dropout)
#
#         self.event_proj = nn.Linear(16 + 16 + 8 + 8, hidden_size)
#
#         self.rnn = nn.LSTM(
#             input_size=hidden_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout if num_layers > 1 else 0.0
#         )
#
#         self.pc_head = nn.Linear(hidden_size, pitch_class_vocab)
#         self.oct_head = nn.Linear(hidden_size, octave_vocab)
#
#         self.vel_head = nn.Linear(hidden_size, vel_vocab)
#         self.dt_head = nn.Linear(hidden_size, dt_vocab)
#
#     def forward(self, notes, others):
#
#         pc = notes % 12
#         octv = notes // 12
#
#         vel = others[:, :, 0].long()
#         dt  = others[:, :, 1].long()
#
#         x = torch.cat([
#             self.pc_emb(pc),
#             self.oct_emb(octv),
#             self.vel_emb(vel),
#             self.dt_emb(dt)
#         ], dim=-1)
#
#         x = self.dropout(x)
#         x = self.event_proj(x)
#         x = self.dropout(x)
#
#         out, _ = self.rnn(x)
#         out = self.dropout(out)
#
#         return (
#             self.pc_head(out),
#             self.oct_head(out),
#             self.vel_head(out),
#             self.dt_head(out)
#         )
#
#
# def train(model, dataloader, epochs=3, lr=1e-3):
#
#     opt = torch.optim.Adam(model.parameters(), lr=lr)
#     ce = nn.CrossEntropyLoss()
#
#     for epoch in range(epochs):
#         total = 0
#
#         for notes, others in dataloader:
#
#             pc_logits, oct_logits, vel_logits, dt_logits = model(notes, others)
#
#             pc = notes % 12
#             octv = notes // 12
#
#             loss_pc = ce(pc_logits[:, :-1].reshape(-1, 12), pc[:, 1:].reshape(-1))
#             loss_oct = ce(oct_logits[:, :-1].reshape(-1, 11), octv[:, 1:].reshape(-1))
#
#             loss_vel = ce(
#                 vel_logits[:, :-1].reshape(-1, vel_logits.size(-1)),
#                 others[:, 1:, 0].reshape(-1)
#             )
#
#             loss_dt = ce(
#                 dt_logits[:, :-1].reshape(-1, dt_logits.size(-1)),
#                 others[:, 1:, 1].reshape(-1)
#             )
#
#             loss = loss_pc + loss_oct + loss_vel + loss_dt
#
#             opt.zero_grad()
#             loss.backward()
#             opt.step()
#
#             total += loss.item()
#
#         print("epoch", epoch, "loss", total)
#
#
# def rebuild_note(pc, octv):
#     return octv * 12 + pc
#
#
# @torch.no_grad()
# def compose(model, seed_notes, seed_others, length=100):
#
#     model.eval()
#
#     notes = seed_notes.tolist()
#     others = seed_others.tolist()
#
#     generated = []
#
#     def sample(logits, temp=0.9):
#         probs = torch.softmax(logits / temp, dim=-1)
#         return torch.multinomial(probs, 1).item()
#
#     seq_notes = notes[:]
#     seq_others = others[:]
#
#     for _ in range(length):
#
#         n = torch.tensor([seq_notes], dtype=torch.long)
#         o = torch.tensor([seq_others], dtype=torch.long)
#
#         pc_logits, oct_logits, vel_logits, dt_logits = model(n, o)
#
#         pc = sample(pc_logits[:, -1])
#         octv = sample(oct_logits[:, -1])
#         vel = sample(vel_logits[:, -1])
#         dt = sample(dt_logits[:, -1])
#
#         next_note = octv * 12 + pc
#
#         generated.append((next_note, vel, dt))
#
#         seq_notes.append(next_note)
#         seq_others.append([vel, dt])
#
#         if len(seq_notes) > 64:
#             seq_notes = seq_notes[-64:]
#             seq_others = seq_others[-64:]
#
#     return generated
#
#
# def composeMusic(songs):
#
#     dataset = MusicDataset(songs, SEQUENCE_LENGTH)
#     loader = DataLoader(
#         dataset,
#         batch_size=16,
#         shuffle=True,
#         num_workers=2,
#         collate_fn=collate_fn
#     )
#
#     model = MusicRNN()
#
#     train(model, loader, epochs=3)
#
#     seed_notes = torch.tensor(
#         [n[0] for n in songs[0][:SEQUENCE_LENGTH]],
#         dtype=torch.long
#     )
#
#     seed_others = torch.tensor(
#         [[n[1], n[2]] for n in songs[0][:SEQUENCE_LENGTH]],
#         dtype=torch.long
#     )
#
#     music = compose(model, seed_notes, seed_others, length=200)
#
#     print(music[:10])
#     return music
#
#
# # improvements:
# # 1. Embed velocity and time too (big improvement) done
# # 2. Predict autoregressively during training done
# # 3. Longer sequences done
# # 4. Don't restart LSTM state during generation // done got slow training here
# # 5. Add dropout done
# # 6. Weight losses differently done
# # 7. Teacher forcing schedule skip
# # 8. Better pitch representation (octaves): todo
# #     Split note into:
# #     pitch_class(0–11)
# #     octave(0–10 - ish)
# #     Then embed both separately and concatenate.
#
# # 11. Gradient clipping maybe implement, probs not?
# # 12. Layer normalization implement
# # 16. Consider predicting NOTE EVENTS instead of separate outputs) (try in seperate file)
#
# # train on specific piece:
# # Step 1: “prime” the model
# # Step 2: generate from last state
# # also Use “teacher forcing warm-up”