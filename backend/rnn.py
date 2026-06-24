
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# ------------------------
# Model
# ------------------------


# hidden_size = 64
# hidden_size = 128
# hidden_size = 256

class MusicRNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=256, num_layers=2, num_notes=128): # 128 possible notes
        # hidden_size = number of values in the LSTM’s internal state (memory vector), it’s what the LSTM uses to “understand” the sequence

        super().__init__() # initializes nn.Module

        print("Initializing MusicRNN")
        embed_dim = 32 # Each note gets a vector of this size
        out_features = 3
        # self.sequence_length = 20 # cut songs into pieces of this len # todo use this

        self.embedding = nn.Embedding( num_notes, embed_dim) # Each note (0–127) gets mapped to a vector of size embed_dim # todo: definethis globally?

        print( "blaa" )

        #torch.nn.LSTM(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False, proj_size=0, device=None, dtype=None)
        self.rnn = nn.LSTM(embed_dim + out_features, hidden_size, num_layers, batch_first=True) # 3: velocity, time, duration

        # separate heads (these are last layers)
        self.note_head = nn.Linear(hidden_size, num_notes) # classification: predicts which note is next
        self.other_head = nn.Linear(hidden_size, out_features) # predicts: velocity, time, duration


    def forward(self, x_notes, x_others ): # x: input sequence of music data. x = (batch of songs, time sequence, input features) // batch_size = 30 songs, sequence_length = 100 notes per song

        print("x_notes shape:", x_notes.shape)
        print("x_others shape:", x_others.shape)

        print( "first note: ", x_notes[ 0 ] )
        print( "max note:", x_notes.max())
        print( "min note:", x_notes.min())

        print(self.embedding)
        note_emb = self.embedding( x_notes )
        print( "after embedding" )

        x = torch.cat( [ note_emb, x_others ], dim=-1 )
        print( "after cat" )
        print( "note shape: ", note_emb.shape )
        print( "others shape: ", x_others.shape )

        out, _ = self.rnn( x ) # output at every time step. # out shape: (batch, time sequence, hidden state vector)
        print( "forward: after .rnn(x)" )

        # batch → how many sequences you fed in (you choose this)
        # sequence_length → how many time steps per sequence (from your data) todo: this has to be const
        # hidden_size → internal feature size of the LSTM (you choose this when defining the model)

        h = out[:, -1, :] # h shape = (batch, hidden state vector)
        # : → all batches
        # -1 → last timestep in the sequence
        # : → all hidden features

        note_logits = self.note_head(h) # tensor: note_logits shape = (batch, 128). // for each sequence in batch produces one score for every possible note (0–127 MIDI notes).
        other = self.other_head(h) # tensor: one prediction per sequence in the batch for velocity, time, ...

        return note_logits, other


def sequence_song( song, seq_len ): # song: list of tuples: ( noteNum, msg.note, velocity, start_time, duration )
    sequences = [] # list of lists of tuples
    chunkNum = len( song ) // seq_len
    print( "chunkNum:", chunkNum, " seq_len:", seq_len, " len song:", len(song) ) # todo: error here

    for i in range( chunkNum ):
        print( "sequence_song: i=" , i, " chunkNum=" , chunkNum )
        sequences.append( song[ i * seq_len : ( i + 1 ) * seq_len ] )

    return sequences

def prepare_data( sequences ): # the whole sequence (except last element) is used to predict one note (the last one)

    print( "prepare data begin" )
    print("sequencesss2 len: ", len( sequences ) )
    X, note_targets, other_targets = [], [], []

    count = 1
    for seq in sequences:
        print( "in forrrrrr" )
        print( "count: ", count, "seq: ", seq )
        print( "seqqq lennn: ", len( seq ) ) # todo error: this should not be 54
        count += 1
        X.append( seq[ : -1 ] ) # all elements in sequence except the last one

        print( "seq len: ", len( seq ) ) # todo: throw exception if 0  or not seqlen?
        print( "prepare data: len( seq[ 0 ] ): ", len( seq[ 0 ] )) # 5 -> feature vector

        print( "seqqqq [0]: , ", seq[0][0] )
        print( "seqqqq [0]: , ", seq[0][1] )

        note_targets.append( seq [ -1 ][ 0 ] )      # note in last element of sequence
        other_targets.append( seq[ -1 ][ 2: ] )     # rest of features in last element of sequence # todo: 1 is note number, dont need as feature

    print( "prepare data end" )

    print( "prepare data: X: ", np.shape( X ) ) # todo: this is of dimensions: (54, 19, 5) ???

    print( "prepare_data: tensor1" )
    # tensor1 = torch.tensor( X, dtype=torch.float32 )
    # tensor1 = torch.tensor( X )

    tensor1 = torch.tensor(X, dtype=torch.float32)

    x_notes = tensor1[:, :, 0].to(torch.long)
    x_other = tensor1[:, :, 2:].to(torch.float32)

    # embed_dim = 32  # todo! embedding should be in model!!!
    # embedding = nn.Embedding(128, embed_dim)
    # pitch_emb = embedding(x_pitch)

    # tensor1 = torch.cat([pitch_emb, x_other], dim=-1)

    print("prepare_data: tensor2")

    tensor2 = torch.tensor( note_targets, dtype=torch.long )
    print("prepare_data: tensor3")
    tensor3 = torch.tensor( other_targets, dtype=torch.float32 )

    # return tensor1, tensor2, tensor3
    return x_notes, x_other, tensor2, tensor3

    # return (
    #     torch.tensor( X, dtype=torch.float32 ),
    #     torch.tensor( note_targets, dtype=torch.long ),
    #     torch.tensor( other_targets, dtype=torch.float32 )
    # )


# ------------------------
# Train
# ------------------------

def train( model, x_notes, x_other, note_y, other_y, epochs=30 ): # todo: overlapping windows [0,20][1,21][2,22],... -> better learning

    print( "train begin" )
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    note_loss_fn = nn.CrossEntropyLoss()
    other_loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        print( "train in for loop" )
        model.train()
        print( "after training" )

        note_logits, other_pred = model( x_notes, x_other ) # each apoch - one full pass over training data

        print( "after model" )

        loss_note = note_loss_fn(note_logits, note_y) # loss functions
        loss_other = other_loss_fn(other_pred, other_y)

        loss = loss_note + loss_other

        print( "after loss" )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    print( "train end" )

# ------------------------
# Generate
# ------------------------

def generate(model, seed_notes, seed_others, length=20): # seed : initial sequence to start predicting on. todo: length == generated sequence len? this can be larger?
    model.eval() # switches a neural network from training to evaluation mode
    generated = []

    current_seq_notes = seed_notes.clone()
    current_seq_others = seed_others.clone()

    for _ in range(length):
        print( "generate: in for loop!" )
        with torch.no_grad():
            note_logits, other = model( current_seq_notes.unsqueeze( 0 ), current_seq_others.unsqueeze( 0 ) ) # this runs model  // unsqueeze(0) → adds batch dimension

        note = torch.argmax(note_logits, dim=1).item() # selects the highest-scoring note
        velocity, time, duration = other[ 0 ].tolist() # predicted continuous values for the next event

        new_event = torch.tensor( [ note, velocity, time, duration ] ) # one generated timestep

        generated.append( new_event.tolist() )

        new_note = torch.tensor([note], dtype=torch.long)

        new_other = torch.tensor( [[velocity, time, duration]], dtype=torch.float32 )

        # torch.cat: concatenates a sequence of tensors along a specified dimension
        # current_seq = torch.cat( [ current_seq[ 1: ], new_event.unsqueeze( 0 ) ], dim=0 ) # moves the sequence forward (dim=0-> time dimension/ next element in sequence (next note, velocity, time,...))

        current_seq_notes = torch.cat( [current_seq_notes[1:], new_note], dim=0 )
        current_seq_others = torch.cat( [current_seq_others[1:], new_other], dim=0 )

        # unsqueeze:
        # turns[note, velocity, time, duration]
        # into[[note, velocity, time, duration]]

    print( "after for loop" )
    return generated


def compose( songs, seq_len = 20 ):


    all_sequences = []
    for song in songs:
        song_sequences = sequence_song( song, seq_len )

        for song_sequence in song_sequences:
            print( "song_sequence lennn: ", len( song_sequence ) )

        all_sequences.extend( song_sequences ) # todo: error here?

    print( "sequences length: ", len( all_sequences ) )

    notes, others, note_targets, other_targets = prepare_data( all_sequences )

    print( "after prepare data" )

    model = MusicRNN()

    print( "after music rnn" )

    # train(model, X, note_y, other_y)
    train( model, notes, others, note_targets, other_targets ) # todo: rewrite train() so that it takes notes, others

    # last sequence = songs[-1][-seq_len:] # todo: use this as seed?

    seed_notes = notes[0]
    seed_others = others[0]
    music = generate(model, seed_notes, seed_others )

    print("\nGenerated:")
    for note in music:
        print(note)

    return music

