# Field evaluation — NOT MEASURED

No labelled field dataset was found at `data/field`.

Everything the project reports past the detector (tracking, association, OCR,
fines, speed, calibration) is measured on synthetic inputs, and the detector
itself only on still images (`docs/FIELD_EVALUATION.md`). This command measures
all of it on real footage once footage exists:

1. Record sequences from fixed cameras; label them in the schema of
   `modules/field_data.py` (`dataset.json`, `vehicles.jsonl`, `frames.jsonl`),
   exhaustively on labelled frames, with annotator and reviewer per label.
2. `python3 field_dataset.py validate data/field --check-files`
3. `python3 field_dataset.py assign-splits data/field --external-camera <a camera never
   developed on>`
4. `python3 evaluate_field.py --dataset data/field --split held_out --model <weights>`

Until then there is no field accuracy to report, and none is claimed.
