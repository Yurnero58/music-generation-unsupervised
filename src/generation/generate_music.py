import pretty_midi
import numpy as np

def matrix_to_midi(matrix, output_path, fs=4):
    """
    TASK 1: Converts an 88-dimension matrix into a single-track piano MIDI.
    """
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    step_duration = 1.0 / fs
    
    for pitch_idx in range(matrix.shape[1]):
        pitch = pitch_idx + 21
        note_events = matrix[:, pitch_idx] == 1.0
        
        start_time = None
        for t, is_playing in enumerate(note_events):
            current_time = t * step_duration
            if is_playing and start_time is None:
                start_time = current_time
            elif not is_playing and start_time is not None:
                note = pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=current_time)
                instrument.notes.append(note)
                start_time = None
                
        if start_time is not None:
            end_time = matrix.shape[0] * step_duration
            note = pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=end_time)
            instrument.notes.append(note)
            
    pm.instruments.append(instrument)
    pm.write(output_path)


def multi_matrix_to_midi(matrix, output_path, fs=10):
    """
    TASKS 2-4: Converts a 352-dimension matrix back into 4 separate MIDI tracks.
    (Drums, Piano, Guitar, Synth Strings)
    """
    pm = pretty_midi.PrettyMIDI()
    
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    piano = pretty_midi.Instrument(program=0, is_drum=False, name="Piano")
    guitar = pretty_midi.Instrument(program=27, is_drum=False, name="Guitar")
    synth = pretty_midi.Instrument(program=50, is_drum=False, name="Synth Strings")
    
    instruments = [drums, piano, guitar, synth]
    
    for t in range(matrix.shape[0]):
        for i in range(4):
            track_slice = matrix[t, (i*88):((i+1)*88)]
            
            for pitch in range(88):
                if track_slice[pitch] > 0.5:
                    note = pretty_midi.Note(
                        velocity=100, pitch=pitch + 21,
                        start=t / fs, end=(t + 1) / fs
                    )
                    instruments[i].notes.append(note)
                    
    for inst in instruments:
        if len(inst.notes) > 0:
            pm.instruments.append(inst)
            
    pm.write(output_path)