
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import os
from .music_config import DT_VOCAB, VEL_VOCAB
from ..services import midi_parser as Parser
from ..services import midi_tester as midi_tester

from .base_model import BaseMusicModel, SEED_NOTES


SEQUENCE_LENGTH = 64
MODEL_DIR = "./music/trained_models/rnn"
MODEL_NUM = 1

from .music_config import (
    PITCH_CLASS_VOCAB, OCTAVE_VOCAB, PITCH_VOCAB, VEL_VOCAB, DT_VOCAB, DUR_VOCAB,
    MAX_PITCH, MAX_VELOCITY, MAX_TIME, MAX_DURATION,
)

os.makedirs(MODEL_DIR, exist_ok=True)

class MusicDataset(Dataset):

    def __init__(self, songs, seq_len):
        self.songs = songs
        self.seq_len = seq_len

    def __len__(self):
        return len(self.songs) * 50

    def __getitem__(self, idx):
        song = self.songs[idx % len(self.songs)]

        # mix random + deterministic sampling
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
class MusicRNN(BaseMusicModel):

    def __init__(
        self,
        pitch_class_vocab=12,
        octave_vocab=11,
        vel_vocab=VEL_VOCAB,
        dt_vocab=DT_VOCAB,
        hidden_size=256, # number of features (or dimensions) in the hidden state vector
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.pc_emb = nn.Embedding(pitch_class_vocab, 16) # todo: why 16
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
        x = self.event_proj(x) # Projection into model space
        x = self.dropout(x)

        out, _ = self.rnn(x) # shape: (16, 64, 256) (batch, seq len, hidden size)
        out = self.dropout(out)

        return (
            self.pc_head(out),
            self.oct_head(out),
            self.vel_head(out),
            self.dt_head(out)
        )

    def fineTune(self, song):
        fineTune(self, song)
        return self

    def generate(self, seedSong, length=200):
        return compose(self, seedSong, length=length)


def train(model, dataloader, epochs=3, lr=1e-3): # todo: what is lr

    opt = torch.optim.Adam(model.parameters(), lr=lr) # updates weights using gradients
    ce = nn.CrossEntropyLoss() # used because all outputs are classification problems

    for epoch in range(epochs):
        total = 0.0

        for notes, others in dataloader:

            pc_logits, oct_logits, vel_logits, dt_logits = model(notes, others)
            # shapes:
            # pc_logits(B, T, 12) # todo what is B, T
            # oct_logits(B, T, 11)
            # vel_logits(B, T, 9)
            # dt_logits(B, T, 17)

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

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient clipping

            opt.step()

            total += loss.item()

        print(f"rnn epoch {epoch} | loss {total:.4f}")

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"pretrained_{MODEL_NUM}.pt"))

def loadModel():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = os.path.join(MODEL_DIR, f"pretrained{MODEL_NUM}.pt")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"No trained RNN model found at {model_path}")

    model = MusicRNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model

def fineTune(model, song, seq_len=64, epochs=2, lr=1e-4):

    model.train()

    # 1) create dataset with ONLY this song
    dataset = MusicDataset([song], seq_len)

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,   # safer for single-song fine-tune
        collate_fn=collate_fn,
        pin_memory=True
    )

    # 2) optimizer (smaller LR than pretraining!)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0

        for notes, others in loader:

            pc_logits, oct_logits, vel_logits, dt_logits = model(notes, others)

            # split ground truth
            pc = notes % 12
            octv = notes // 12

            loss_pc = loss_fn(
                pc_logits[:, :-1].reshape(-1, 12),
                pc[:, 1:].reshape(-1)
            )

            loss_oct = loss_fn(
                oct_logits[:, :-1].reshape(-1, 11),
                octv[:, 1:].reshape(-1)
            )

            loss_vel = loss_fn(
                vel_logits[:, :-1].reshape(-1, vel_logits.size(-1)),
                others[:, 1:, 0].reshape(-1)
            )

            loss_dt = loss_fn(
                dt_logits[:, :-1].reshape(-1, dt_logits.size(-1)),
                others[:, 1:, 1].reshape(-1)
            )

            loss = 2 * loss_pc + loss_oct + loss_vel + loss_dt

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

        print(f"[fine-tune] epoch {epoch+1} | loss {total_loss:.4f}")

    return model

@torch.no_grad() # do not compute gradients
def compose(model, seedSong, length=100):

    model.eval()

    seedSong = seedSong[:SEED_NOTES]

    seq_notes = [n[0] for n in seedSong]
    seq_others = [[n[1], n[2]] for n in seedSong]

    generated = []

    def sample(logits, temp=0.9):
        probs = torch.softmax(logits / temp, dim=-1)
        return torch.multinomial(probs, 1).item() # Turns raw model scores into probabilities

    for _ in range(length):

        n = torch.tensor([seq_notes], dtype=torch.long)
        o = torch.tensor([seq_others], dtype=torch.long)

        pc_logits, oct_logits, vel_logits, dt_logits = model(n, o) # predicted distributions for each time step

        pc = sample(pc_logits[:, -1])
        octv = sample(oct_logits[:, -1])
        vel = sample(vel_logits[:, -1])
        dt = sample(dt_logits[:, -1])

        next_note = octv * 12 + pc

        generated.append((next_note, vel, dt))

        seq_notes.append(next_note) # update context
        seq_others.append([vel, dt])

        # keep context stable
        if len(seq_notes) > SEQUENCE_LENGTH:
            seq_notes = seq_notes[-SEQUENCE_LENGTH:]
            seq_others = seq_others[-SEQUENCE_LENGTH:]

    return list(seedSong) + generated

def trainModel(songs):
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

    train(model, loader, epochs=5)

    return model


def composeMusic(seedSong):
    parser = Parser.MidiParser(MAX_VELOCITY, MAX_TIME, MAX_DURATION)
    # model = trainModel(songs)

    model = loadModel()

    model = fineTune(model, seedSong, epochs=2, lr=1e-4)

    # seedSong = songs[0][:SEQUENCE_LENGTH]

    generated = compose(model, seedSong, length=200)
    generatedNotes = parser.convertedNotes(generated)  # todo: need this?
    midi_tester.testMidi(generatedNotes, "midiRNN1.mid")

    return generatedNotes
