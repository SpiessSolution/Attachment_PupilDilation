# Contains

This folder contains:

**Data processing pipeline**

The following python scripts read, (pre)process, segment, aggregate, combine, and export the eyetracker and behavioral data so that they eventually can be read by the statistical analysis script.
Below a short description of each script:

1. 001_preprocess_single_subject_data.py: Imports the raw single-subject eye tracker and stimulus file data, preprocess (clean, segment, standardize) the data and export each dataset.
   inputs: `~data/external/adults/eyetracker/*.tsv`
   output:` ~data/interim/adults/*cleaned.parquet`
2. 002_aggregate_to_group_data.py: Loads and aggregate the single-subject data to create a single dataset that contains the trial-level aggregated and base-line corrected pupil dilation data of all participants
   inputs: `~data/interim/adults/*cleaned.parquet `
   output: `~data/processed/adults/group_data.xlsx`
3. 003_compute_stimulus_luminance.py: Loads all stimuli (from the Radboud face database) and compute the average luminance and the standard deviation of the luminance of each stimulus. The result is exported as an Excel sheet.
   inputs: `~data/stimuli/*jpg`
   output: `~data/processed/stimulus_luminance_values.xlsx`
4. 004_join_datasets.py: Combines all three datasets:

   1. `~data/processed/adults/group_data.xlsx`
   2. `~data/processed/stimulus_luminance_values.xlsx`
   3. `~data/external/adults/Pupilproject_behdata.csv`

   into a single new dataset that can be used for statistical analysis. The final dataset will be exported to:

   `~data/final/adults_final_data.xlsx`


**Statistical Data analysis**

Statistical data analysis is performed in R. The script is embedded in a jupyter notebook

`005_statistical_analysis_R.ipynb`

Input: `~data/final/adults_final_data.xlsx`
Output: `~reports/figures/adults/`

Note that only some visualizations are actually exported.
