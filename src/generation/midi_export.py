# =============================================================================
# src/generation/midi_export.py
# Convert piano-roll numpy arrays → pretty_midi objects → .mid files
# =============================================================================

import numpy as np
import pretty_midi
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import PITCH_RANGE, STEPS_PER_BAR


def pianoroll_to_midi(roll: np.ndarray,
                      bpm: float = 120.0,
                      steps_per_bar: int = STEPS_PER_BAR,
                      pitch_offset: int = PITCH_RANGE[0],
                      instrument_program: int = 0,
                      velocity: int = 80) -> pretty_midi.PrettyMIDI:
    """
    Convert binary piano-roll (T, n_pitches) → PrettyMIDI object.

    Args:
        roll:       (T, n_pitches) float array, values in [0,1]
        bpm:        tempo in beats per minute
        steps_per_bar: time steps per bar (default 16 = 16th-note resolution)
        pitch_offset: MIDI pitch of roll column 0
        instrument_program: GM program number (0=Acoustic Grand Piano)
        velocity:   MIDI velocity for all notes
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    instr = pretty_midi.Instrument(program=instrument_program,
                                   is_drum=False, name="Generated")

    # Time per step (seconds)
    beats_per_step = 4.0 / steps_per_bar    # 1 bar = 4 beats
    sec_per_step   = beats_per_step * (60.0 / bpm)

    T, n_p = roll.shape
    roll_bin = (roll > 0.5).astype(np.uint8)

    for p in range(n_p):
        midi_pitch = p + pitch_offset
        in_note    = False
        note_start = 0.0

        for t in range(T):
            active = bool(roll_bin[t, p])
            if active and not in_note:
                note_start = t * sec_per_step
                in_note    = True
            elif not active and in_note:
                note_end = t * sec_per_step
                if note_end > note_start:
                    instr.notes.append(pretty_midi.Note(
                        velocity=velocity,
                        pitch=midi_pitch,
                        start=note_start,
                        end=note_end))
                in_note = False

        # Close any open note at end of roll
        if in_note:
            note_end = T * sec_per_step
            instr.notes.append(pretty_midi.Note(
                velocity=velocity,
                pitch=midi_pitch,
                start=note_start,
                end=note_end))

    pm.instruments.append(instr)
    return pm


def save_midi(pm: pretty_midi.PrettyMIDI, path: str):
    """Write PrettyMIDI to disk, creating parent dirs as needed."""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pm.write(path)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile, os
    roll = (np.random.rand(64, 48) > 0.90).astype(np.float32)
    pm   = pianoroll_to_midi(roll)
    print(f"Notes: {len(pm.instruments[0].notes)}")
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        tmp = f.name
    save_midi(pm, tmp)
    print(f"Saved to: {tmp}  ({os.path.getsize(tmp)} bytes)")
    os.remove(tmp)