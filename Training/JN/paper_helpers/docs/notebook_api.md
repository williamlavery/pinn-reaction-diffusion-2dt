# Notebook API Surface

`experim.ipynb` currently imports the following helpers directly:

- `helpers.file_finder`: `paths_to_df`, `find_data_obj_files`, `condense_df`
- `helpers.utils`: save/path/prediction/histogram utilities
- `helpers.paper_plots.*`: plotting classes
- `helpers.fwd_sim_wrapper`: forward-simulation batch execution
- `helpers.SR_wrapper`, `helpers.SR_post_analysis`: SR workflow and reporting

Planned migration target:

- `paper_helpers.io.*`
- `paper_helpers.simulation.*`
- `paper_helpers.sr.*`
- `paper_helpers.plotting.*`

Keep behavior equivalent while progressively replacing import paths.
