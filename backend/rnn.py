
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

DT_VOCAB = 17
VEL_VOCAB = 9
SEQUENCE_LENGTH = 20

class MusicDataset(Dataset):
    def __init__(self, songs, seq_len):
        self.samples = []

        for song in songs:

            if len(song) <= seq_len:
                continue

            # Sliding windows
            for i in range(len(song) - seq_len):
                sequence = song[i:i + seq_len]
                target = song[i + seq_len]

                self.samples.append((sequence, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sequence, target = self.samples[idx]

        notes = torch.tensor(
            [note[0] for note in sequence],
            dtype=torch.long
        )

        others = torch.tensor(
            [[note[1], note[2]] for note in sequence],
            dtype=torch.long
        )

        target_note = torch.tensor(
            target[0],
            dtype=torch.long
        )

        target_other = torch.tensor(
            [target[1], target[2]],
            dtype=torch.long
        )

        return notes, others, target_note, target_other


class MusicRNN(nn.Module):

    def __init__(
            self,
            num_notes=128,
            embed_dim=32,
            hidden_size=256,
            num_layers=2,
    ):
        super().__init__()

        # MIDI pitch embedding
        self.embedding = nn.Embedding(num_notes, embed_dim)

        # velocity + delta_time
        other_features = 2

        self.rnn = nn.LSTM(
            input_size=embed_dim + other_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Predict next pitch
        self.note_head = nn.Linear(hidden_size, num_notes)

        # Predict next velocity + delta_time
        self.vel_head = nn.Linear(hidden_size, VEL_VOCAB)
        self.dt_head = nn.Linear(hidden_size, DT_VOCAB)


    def forward(self, notes, others):
        """
        notes  : (batch, seq_len)
        others : (batch, seq_len, 2)
        """
        note_embedding = self.embedding(notes)

        x = torch.cat(
            [note_embedding, others],
            dim=2,
        )

        out, _ = self.rnn(x)

        # Use only the last hidden state
        h = out[:, -1, :]

        note_logits = self.note_head(h)
        # other_output = self.other_head(h)
        vel_logits = self.vel_head(h)
        dt_logits = self.dt_head(h)

        print("note_embedding:", note_embedding.shape)
        print("others:", others.shape)
        print("concat result:", torch.cat([note_embedding, others], dim=2).shape)

        return note_logits, vel_logits, dt_logits
        # return note_logits, other_output

def train(model, dataloader, epochs=10, lr=1e-3):
    # model.to(device)

    epochs = 1 # todo: test delete

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    note_loss_fn = nn.CrossEntropyLoss()
    vel_loss_fn = nn.CrossEntropyLoss()
    dt_loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0

        print("dataloader size: ", len( dataloader ) )

        loopCount = 0
        for notes, others, y_notes, y_others in dataloader:

            y_vel = y_others[:, 0].long()
            y_dt = y_others[:, 1].long()

            # forward
            note_logits, vel_logits, dt_logits = model(notes, others)

            # losses
            loss_note = note_loss_fn(note_logits, y_notes)
            loss_vel = vel_loss_fn(vel_logits, y_vel)
            loss_dt = dt_loss_fn(dt_logits, y_dt)

            loss = loss_note + loss_vel + loss_dt

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

            loopCount += 1
            if loopCount > 100: # todo: this loops like 2000 times. its slow! check why
                break

        print(f"Epoch {epoch+1} | loss {total_loss:.4f}")

@torch.no_grad()
def compose(model, seed_notes, seed_others, length=100):
    print( "in compose" )
    model.eval()

    notes = seed_notes.unsqueeze(0)      # (1, T)
    others = seed_others.unsqueeze(0)    # (1, T, 2)

    generated = []

    def sample(logits, temperature=1.0):
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)  # (1,)

    for _ in range(length):
        print( "compose in for loop" )
        note_logits, vel_logits, dt_logits = model(notes, others)

        next_note = sample(note_logits)  # (1,)
        next_vel = sample(vel_logits)    # (1,)
        next_dt = sample(dt_logits)      # (1,)

        generated.append((
            next_note.item(),
            next_vel.item(),
            next_dt.item()
        ))

        # build next step input
        next_note_in = next_note.unsqueeze(1)  # (1,1)
        next_other_in = torch.stack([next_vel, next_dt], dim=-1).unsqueeze(1)  # (1,1,2)

        notes = torch.cat([notes, next_note_in], dim=1)
        others = torch.cat([others, next_other_in], dim=1)

    return generated

def composeMusic( songs ):
    dataset = MusicDataset(songs, SEQUENCE_LENGTH)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = MusicRNN()

    train(model, loader, epochs=10)

    seed_notes = torch.tensor(
        [note[0] for note in songs[0][:SEQUENCE_LENGTH]],
        dtype=torch.long
    )

    seed_others = torch.tensor(
        [note[1:] for note in songs[0][:SEQUENCE_LENGTH]],
        dtype=torch.float32
    )

    music = compose(model, seed_notes, seed_others, length=200)

    print(music[:10])
    return music