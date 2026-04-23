# split_csvs

This folder stores per-timepoint CSV files of cell-coordinate observations used by the data preprocessing pipeline.

## What each CSV contains

- Plain CSV with no header row.
- Each row is one detected cell/point.
- The pipeline reads the first two columns as:
  - column 1: `x` coordinate
  - column 2: `y` coordinate
- If extra columns exist, they are ignored by the loader.

Code reference: `Training/data/python_v2/modules/data_class.py` (`read_points_csv`) reads files with `pd.read_csv(..., header=None)` and uses `df.iloc[:, :2]`.

## Filename format

Expected format (used by the pipeline):

```text
color_{species}_dose_{x2}_ic_{x3}_rep1_{x4}_rep2_{x5}_hours_{x6}_plate_{x7}.csv
```

Where:

- `species`: channel label, typically `green` or `red`
- `dose_{x2}`: dose condition key (e.g., `0p0`)
- `ic_{x3}`: initial-condition key (e.g., `0p0`)
- `rep1_{x4}`: replicate group/index 1
- `rep2_{x5}`: replicate group/index 2
- `hours_{x6}`: timepoint in hours (e.g., `8`, `24`, `64`)
- `plate_{x7}`: plate identifier (e.g., `1`)

## Example

```text
color_green_dose_0p0_ic_0p0_rep1_2_rep2_3_hours_24_plate_1.csv
```

This means:

- species/channel: `green`
- dose: `0p0`
- initial condition: `0p0`
- replicate pair: `rep1=2`, `rep2=3`
- timepoint: `24` hours
- plate: `1`

## How the pipeline uses these names

The suffix is generated in:

- `Training/data/python_v2/modules/data_class.py`

using:

```text
_dose_{x2}_ic_{x3}_rep1_{x4}_rep2_{x5}_hours_{x6}_plate_{x7}.csv
```

and prefixed with `color_green` / `color_red` to load channel-specific files.
