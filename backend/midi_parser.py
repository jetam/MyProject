
import mido
import numpy as np

# This is only intended for piano/ single instrument music!

MAX_MIDI_PITCH = 127
MAX_MIDI_VELOCITY = 127

# INFO: Tempo is put into times and duration of the notes
#       Every MIDI event has time. the time difference between events is time * current relativeTempo



class MidiParser:
    def __init__(self, MAX_VELOCITY, MAX_TIME, MAX_DURATION, MAX_PITCH = 127 ):
        self.MAX_VELOCITY = MAX_VELOCITY
        self.MAX_TIME = MAX_TIME
        self.MAX_DURATION = MAX_DURATION
        self.MAX_PITCH = MAX_PITCH

    # convert MIDI into feature vectors: pitch, velocity, time. Time = time since last event (note or meta message)
    def read_midi(self, midi_file_path):

        mid = mido.MidiFile( midi_file_path )

        current_time = 0
        startTime = False # Start measuring time at first note event
        self.midi_data = [] # put feature vectors here [pitch, velocity, time]
        self.meta_data = [] # meta data. todo: need this?
        active_notes = {} # notes that havent been turned off yet. This is used to get note durations
        noteNumber = 0
        midiTemporaryData = []

        maxTime = 0.0 # time and delay need to be quantized.
        maxDuration = 0.0


        def handleActiveNote(msg):
            noteNum, start_time, velocity, relativeTempo = active_notes[msg.note]
            duration = current_time - start_time

            midiTemporaryData.append(
                ( msg.note, noteNum, velocity, start_time, duration, relativeTempo )
            )
            del active_notes[msg.note]


        startTempo = -1
        currentTempo = 1

        for msg in mid:
            # print( "        Midi msg: " + str(msg) )
            # todo: handle different channels? - have option to set only main channel

            noteType = msg.type

            if( not startTime and noteType == "note_on" ):
                startTime = True

            if startTime:
                current_time += msg.time

            if noteType == 'set_tempo':
                # print("tempo change!!!!: ", msg.tempo)
                if startTempo == -1:
                    startTempo = msg.tempo
                currentTempo = msg.tempo


            if noteType != 'note_on' and noteType != 'note_off':

                self.meta_data.append( [noteNumber, msg] )
                # todo: include control change 64. (sustain). also check if note numbering is ok. include sustain in midi_data
                # todo: also include cc 7 (volume)
                # todo: set tempo:  Midi msg: MetaMessage('set_tempo', tempo=983606, time=0.03225803333333333)
                # todo: tempo change can be put in times. read start tempo in beginning. then change times: time = time * startTempo/currentTempo
                continue



            if msg.velocity == 0 and noteType == 'note_on':
                noteType = 'note_off' # some MIDI files have onNotes with velocity 0 instead of offNotes

            if noteType == "note_on":
                noteNumber += 1

                # todo: whatif active notes already contains the note? - end last note and startnew one?
                if active_notes.__contains__( msg.note ):
                    continue

                active_notes[msg.note] = (noteNumber, current_time, msg.velocity, startTempo/currentTempo)

            elif noteType == "note_off":
                if msg.note not in active_notes:
                    continue
                handleActiveNote( msg )
            else:
                print( "unknown msg type {}".format( noteType ) )

        # order by noteNum
        sortedData = sorted(midiTemporaryData, key=lambda x: x[1])

        previousTime = 0
        relativeTempo = 1

        for msg in sortedData:

            deltaTime = ( msg[3] - previousTime ) * relativeTempo
            duration = msg[4] * relativeTempo
            previousTime = msg[3]

            velocity = ( msg[2] // ( MAX_MIDI_VELOCITY // self.MAX_VELOCITY ) )  # max is 8

            if( maxTime < deltaTime ):
                maxTime = deltaTime
            if( maxDuration < duration ):
                maxDuration = duration

            self.midi_data.append([msg[0], velocity, deltaTime, duration])  # ( note, velocity, delta time, duration)

            relativeTempo = msg[5]

        for data in self.midi_data:

            data[2] = ( self.MAX_TIME * data[2] ) // maxTime

            # data[3] =  ( self.MAX_DURATION * data[3] ) // maxDuration # todo: doesnt work well.
            data[3] = self.MAX_DURATION # todo: keep note duration constant for now, maybe change later

        self.midi_data = [song[:-1] for song in self.midi_data]


    def convertedNotes(self, generatedNotes): # todo: this is used after transformer. make so everything is universal
        print( "in convertedNotes" )
        converted = [(p, v * (MAX_MIDI_VELOCITY // self.MAX_VELOCITY), dt / self.MAX_TIME) for (p, v, dt) in generatedNotes] # leave out duration for now: u / self.MAX_DURATION
        return converted

