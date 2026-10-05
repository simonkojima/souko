# souko

Export EEG trials once, including expensive Euclidean alignment (EA), and
load them repeatedly with explicit experiment splits.

```python
from souko.xy import export_data, get_data

# dataset is a MOABB-compatible dataset instance.
export_data(dataset, resample=128, tmin=0.5, tmax=4,
            event_id={"left_hand": 0, "right_hand": 1})

data = get_data("Lee2019_MI", subject=1, session=1,
                train_test_split=0.8, valid=True,
                train_valid_split=0.8, ea=True)
# data["train"], data["valid"], data["test"] contain eeg and label arrays.
```

The default split is **stratified chronological**: take the leading fraction
of each class for training, then restore original trial order within each
partition. This supports arbitrary class labels. Fractions are rounded down
at cumulative boundaries; small classes may produce empty partitions.
Class proportions are approximately preserved, not signal distributions.
Class-specific temporal boundaries may overlap. `strategy="chronological"`
uses one temporal boundary across all classes.

Validation is taken exclusively from training. With the defaults, the overall
fractions are approximately train 64%, valid 16%, test 20%.

For explicit run splits:

```python
data = get_data("Dreyer2023", subject=1,
                runs={"train": [1, 2], "test": [3, 4, 5, 6]}, valid=True)
# Equivalent dataset preset:
data = get_data("Dreyer2023", subject=1, strategy="preset")
```

## Cache and metadata

Exports live under `~/datasets/<dataset>/sub-<id>/ses-<id>/<preprocessing>/`.
`export_data(save_base=...)` sets the dataset root; loader `base=...` sets
its parent (the directory containing `datasets`). Specify the same processing
parameters on export and load. Defaults are 128 Hz, 7–30 Hz, tmin=0.5,
tmax=4 for Lee2019_MI and 5 otherwise; export uses dataset intervals when
bounds are omitted.

Run files contain X, y, and original event sample positions. Session metadata
records original session/run names, trial counts, label codes, preprocessing,
and EA scopes. Numeric session/run IDs follow source iteration order.
Sessions are concatenated in numeric order by default; explicit session lists
in `get_data_cross` define concatenation order.

EA and online EA are computed during export and cached at session scope.
Loaders select the corresponding trial ranges without recomputation.
Session EA uses the full session, including subsequently selected test trials;
it therefore represents a full-session adaptation protocol. Online behavior
is provided by `transfer_bci.euclidean.euclidean_alignment`. Use `ea=False`
on export to skip both, or `online=False` to skip online EA.

`subject_list` and `sessions_list` discover exported data instead of assuming
dataset-specific subject counts. Any dataset using the common export format
can be loaded. `get_data_cross(..., return_info=True)` returns metadata as
`{"sessions": {"1": ..., "2": ...}}` so no session metadata is lost.
Representation-independent `souko.split.split_indices` can also be applied
to future covariance arrays with the same trial ordering.

## Migration

- Dreyer's default is now a class-wise chronological split; pass
  `strategy="preset"` to retain the previous run selection.
- Lee loads the common exporter layout. Old `lee2019/x_y` caches must be
  re-exported; existing common-format run caches remain readable.
- Label mapping matches exact slash-delimited event tokens and rejects
  ambiguous or missing mappings.
- Dataset aliases `dreyer_2023` and `lee_2019` remain accepted.

## Dependencies and checks

Loading requires NumPy only. Install `souko[export]` for MNE; EA export also
requires the separately installed `transfer_bci` library. Install MOABB if
using its dataset adapters.

```sh
python -m unittest discover -s tests
```
