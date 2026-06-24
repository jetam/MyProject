
import mido
import numpy as np

class MidiParser:
    def __init__(self): #time_delta, pitch, velocity
        self.midi_data = []

    def read_midi(self, midi_file_path):
        print("midi_file_path:")
        # msg.time - time to wait since next event
        # todo: feature vector: note, velocity, start_time, duration

        mid = mido.MidiFile( midi_file_path )

        current_time = 0
        self.midi_data = []
        self.meta_data = []
        active_notes = {}
        noteNumber = 0
        midiTemporaryData = []

        def handleActiveNote(msg):
            noteNum, start_time, velocity = active_notes[msg.note]
            duration = current_time - start_time

            midiTemporaryData.append(
                ( msg.note, noteNum, velocity, start_time, duration)
            )
            del active_notes[msg.note]


        startTempo = -1

        for msg in mid:
            print( "        Midi msg: " + str(msg) )
            # todo: handle different channels? - have option to set only main channel

            noteType = msg.type

            if noteType == 'set_tempo' and startTempo == -1:
                print( "tempo change!!!!: ", msg.tempo )
                startTempo = msg.tempo


            if noteType != 'note_on' and noteType != 'note_off':
                # print( msg )
                # print( msg.time )
                # print( noteType )
                current_time += msg.time
                self.meta_data.append( [noteNumber, msg] )
                # todo: include control change 64. (sustain). also check if note numbering is ok. include sustain in midi_data
                # todo: also include cc 7 (volume)
                # this is only intended for piano/ single instrument music!
                # todo: set tempo:  Midi msg: MetaMessage('set_tempo', tempo=983606, time=0.03225803333333333)
                # todo: tempo change can be put in times. read start tempo in beginning. then change times: time = time * startTempo/currentTempo
                continue

            current_time += msg.time

            if msg.velocity == 0 and noteType == 'note_on':
                noteType = 'note_off' # some MIDI files have onNotes with velocity 0 instead of offNotes

            if noteType == "note_on":
                noteNumber += 1

                # todo: whatif active notes already contains the note? - end last note and startnew one?
                if active_notes.__contains__( msg.note ):
                    continue

                active_notes[msg.note] = (noteNumber, current_time, msg.velocity)

            elif noteType == "note_off":
                if msg.note not in active_notes:
                    continue
                handleActiveNote( msg )
            else:
                print( "unknown msg type {}".format( noteType ) )


        # order by noteNum
        sortedData = sorted(midiTemporaryData, key=lambda x: x[1])

        #timeOffset = sortedData[0][3] # offset. first note should start at time 0
        #deltaTime = 0
        previousTime = 0

        for msg in sortedData:
            deltaTime = msg[3] - previousTime
            previousTime = msg[3]
            self.midi_data.append( [ msg[0], msg[2], deltaTime, msg[4] ] ) # ( note, velocity, delta time, duration)


        # ( msg.note, noteNum, velocity, start_time, duration) NOTENUMBER SHOULD NOT BE USED BY ai

        # print( "midi data: " )
        # for item in self.midi_data:
        #     print( item )