import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage


# -----------------------------
# INPUT FORMAT
# (note, velocity, delta_time, duration)
# delta_time = time AFTER previous note ends
# duration = how long the note is held
# -----------------------------
notes = [
    (60, 100, 0, 0.5),   # C4
    (62, 100, 0, 0.5),   # D4 (immediately after previous)
    (64, 100, 0.2, 0.8), # E4 (small gap before)
    (67, 100, 0, 1.0),   # G4
]


# -----------------------------
# SETTINGS
# -----------------------------
TICKS_PER_BEAT = 480
BPM = 120

mid = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
track = MidiTrack()
mid.tracks.append(track)

# set tempo
tempo = mido.bpm2tempo(BPM)
track.append(MetaMessage('set_tempo', tempo=tempo, time=0))


# -----------------------------
# TIME CONVERSION
# seconds -> ticks
# -----------------------------
def sec_to_ticks(seconds, bpm=BPM, tpq=TICKS_PER_BEAT):
    beat_sec = 60 / bpm
    return int((seconds / beat_sec) * tpq)


# -----------------------------
# BUILD MIDI EVENTS
# -----------------------------
current_tick_time = 0

# def testMidi( notes ):
#     print( "midi testttttttttttttttttttttttt" )
#     for note, velocity, delta_time, duration in notes:
#         # gap before note
#         delta_ticks = sec_to_ticks(delta_time)
#         start_time = delta_ticks
#
#         # note on
#         track.append(Message(
#             'note_on',
#             note=note,
#             velocity=velocity,
#             time=start_time
#         ))
#
#         # note off (after duration)
#         duration_ticks = sec_to_ticks(duration)
#         track.append(Message(
#             'note_off',
#             note=note,
#             velocity=0,
#             time=duration_ticks
#         ))


def sec_to_ticks(seconds):
    beat_sec = 60 / BPM
    return int((seconds / beat_sec) * TICKS_PER_BEAT)


def testMidi(notes):
    print("midi testttttttttttttttttttttttt")

    mid = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = MidiTrack()
    mid.tracks.append(track)

    # set tempo
    tempo = mido.bpm2tempo(BPM)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))

    # ----------------------------
    # build absolute event timeline
    # ----------------------------
    events = []
    current_time = 0

    for note, velocity, delta_time, duration in notes:

        current_time += sec_to_ticks(delta_time)
        # current_time += delta_time


        start = current_time
        end = start + sec_to_ticks(duration)
        # end = start + duration


        events.append((start, 'note_on', note, velocity))
        events.append((end, 'note_off', note, 0))

    # ----------------------------
    # sort events globally
    # ----------------------------
    events.sort(key=lambda x: x[0])

    # ----------------------------
    # convert to MIDI delta time stream
    # ----------------------------
    last_time = 0

    for time, msg_type, note, velocity in events:

        delta = time - last_time
        last_time = time

        track.append(Message(
            msg_type,
            note=note,
            velocity=velocity,
            time=delta
        ))

    # ----------------------------
    # save + playback
    # ----------------------------
    filename = "outputTest.mid"
    mid.save(filename)
    print("saved:", filename)



