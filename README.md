# Overview

The repository contains the source code for the processing and analysis of the data communicated in the research paper:

**Attachment is in the Pupil of the Beholder: A Pupillometry Study on Emotion Processing**

**Stefania V. Vacaru**, Theodore E.A. Waters, Sabine Hunnius

1. New York University Abu Dhabi, UAE
2. Vrije Universiteit Amsterdam, The Netherlands
3. Department of Applied Psychology, New York University, New York, USA
4. Donders Institute for Brain, Cognition and Behavior, Radboud University Nijmegen, The Netherlands

**Corresponding author:** Stefania Victorita Vacaru - [vsv9970@nyu.edu](mailto:vsv9970@nyu.edu)

[ORCID](https://orcid.org/0000-0001-6897-2963)

# Introduction

The repository contains the source code for

1. Processing the raw eye tracker data, stimulus trigger data, and stimuli
2. Analysing the group-level data

More information about each script does can be found inside the script.

## Processing raw eye tracker data

Processing the eye tracking data follows the methodology outlined [in this paper](https://link.springer.com/article/10.3758/s13428-018-1075-y).
The eye tracker data is subsequently segmented with the help of trigger files that specify at which moment a time a particular stimulus was shown. After processing the eye tracker and trigger information from each individual participant, the datasets are combined together with behavioral data containing demographic information as well as responses to various questionnaires. Moreover, from the stimuli (i.e., the facial emotional expressions from the [Radboud face database](https://rafd.socsci.ru.nl/?p=main) ), the luminance values are calculated and added to the dataset.

## Statistical analyses

Statistical analyses primarily involve creating multiple different mixed effect models.

# Files and directories


```markdown
# Attachment_PupilDilation

* [data/](./Attachment_PupilDilation/data)
  * [external/](./Attachment_PupilDilation/data/external)
    * [adults/](./Attachment_PupilDilation/data/external/adults)
      * [eyetracker/](./Attachment_PupilDilation/data/external/adults/eyetracker)
        * [eyetracker_schema.json](./Attachment_PupilDilation/data/external/adults/eyetracker/eyetracker_schema.json)
        * [readme.md](./Attachment_PupilDilation/data/external/adults/eyetracker/readme.md)
      * [stimfiles/](./Attachment_PupilDilation/data/external/adults/stimfiles)
        * [readme.md](./Attachment_PupilDilation/data/external/adults/stimfiles/readme.md)
        * [stimfile_schema.json](./Attachment_PupilDilation/data/external/adults/stimfiles/stimfile_schema.json)
      * [readme.md](./Attachment_PupilDilation/data/external/adults/readme.md)
    * [Pupilproject_behdata.csv](./Attachment_PupilDilation/data/external/Pupilproject_behdata.csv)
    * [Pupilproject_behdata_schema.json](./Attachment_PupilDilation/data/external/Pupilproject_behdata_schema.json)
  * [final/](./Attachment_PupilDilation/data/final)
    * [adults_final_data.xlsx](./Attachment_PupilDilation/data/final/adults_final_data.xlsx)
    * [readme.md](./Attachment_PupilDilation/data/final/readme.md)
  * [processed/](./Attachment_PupilDilation/data/processed)
    * [readme.md](./Attachment_PupilDilation/data/processed/readme.md)
    * [stimulus_luminance_values.xlsx](./Attachment_PupilDilation/data/processed/stimulus_luminance_values.xlsx)
  * [stimuli/](./Attachment_PupilDilation/data/stimuli)
    * [readme.md](./Attachment_PupilDilation/data/stimuli/readme.md)
* [notebooks/](./Attachment_PupilDilation/notebooks)
  * [006_visualizations.ipynb](./Attachment_PupilDilation/notebooks/006_visualizations.ipynb)
  * [readme.md](./Attachment_PupilDilation/notebooks/readme.md)
* [reports/](./Attachment_PupilDilation/reports)
  * [figures/](./Attachment_PupilDilation/reports/figures)
    * [adults/](./Attachment_PupilDilation/reports/figures/adults)
      * [Dismissing_int.png](./Attachment_PupilDilation/reports/figures/adults/Dismissing_int.png)
      * [Preoccupied_int.png](./Attachment_PupilDilation/reports/figures/adults/Preoccupied_int.png)
      * [Secure_int_separateModel.png](./Attachment_PupilDilation/reports/figures/adults/Secure_int_separateModel.png)
      * [bc_pd_ts_condition.png](./Attachment_PupilDilation/reports/figures/adults/bc_pd_ts_condition.png)
      * [pd_ts_condition.png](./Attachment_PupilDilation/reports/figures/adults/pd_ts_condition.png)
    * [.gitkeep](./Attachment_PupilDilation/reports/figures/.gitkeep)
  * [readme.md](./Attachment_PupilDilation/reports/readme.md)
* [src/](./Attachment_PupilDilation/src)
  * [modules/](./Attachment_PupilDilation/src/modules)
    * [__init__.py](./Attachment_PupilDilation/src/modules/__init__.py)
    * [readme.md](./Attachment_PupilDilation/src/modules/readme.md)
    * [utils.py](./Attachment_PupilDilation/src/modules/utils.py)
  * [001_preprocess_single_subject_data.py](./Attachment_PupilDilation/src/001_preprocess_single_subject_data.py)
  * [002_aggregate_to_group_data.py](./Attachment_PupilDilation/src/002_aggregate_to_group_data.py)
  * [003_compute_stimulus_luminance.py](./Attachment_PupilDilation/src/003_compute_stimulus_luminance.py)
  * [004_join_datasets.py](./Attachment_PupilDilation/src/004_join_datasets.py)
  * [005_statistical_analysis_R.ipynb](./Attachment_PupilDilation/src/005_statistical_analysis_R.ipynb)
  * [__init__.py](./Attachment_PupilDilation/src/__init__.py)
  * [readme.md](./Attachment_PupilDilation/src/readme.md)
* [README.md](./Attachment_PupilDilation/README.md)
* [R_environment.txt](./Attachment_PupilDilation/R_environment.txt)
* [environment.yaml](./Attachment_PupilDilation/environment.yaml)
```



# Software dependencies

The source code for processing the raw eye tracker data relies on Python while the statistical analyses were performed in R exclusively.

- The details of the python environment can be found in the root directory of the repository in a file called `environment.yml`.
- The details of the R environment can be found in the root directory of the repository in a file called `R_environment.txt`.

# How to use

## Installation

It is recommended to run the source code using an advanced IDE such as [Visual Studio Code](https://code.visualstudio.com/docs/languages/r) which is freely available and provides support for jupyter notebooks with Python and R kernels (which is required for running the code).

### Python

To re-create the exact compute environment it is recommended to make use of a virtual environment management software such as [Anaconda](https://www.anaconda.com) which is available for free.
When Anaconda is installed, a new virtual python environment can be created using:

```
conda env create -f environment.yml
```

### R

The compute environment for R can be replicated by installing the libraries specified in `R_environment`.

## Application

The source code is located in the `src` directory of the repository. The python files should be executed in the following order manually:

1. 001_process_single_subject_data.py
2. 002_aggregate_to_group_data.py
3. 003_compute_stimulus_luminance.py
4. 004_join datasets.py

After running the code, a final dataset ready for the statistical analysis is available in `~data/final`.

The jupyter notebook `src/005_statistical_analysis_R.ipynb` can be run to reproduce the analysis results and visualizations included in the paper.

The folder `~notebooks/` contains a python jupyter notebook that can be used to generate single-subject quality control figures and group level visualizations of the pupil dilation time series data including the analysis window, timelocked to the stimulus onset.

## Notes

Since the raw data is not included in the repository, the running the python data processing pipeline will fail because it cannot find the raw data.

Therefore, the final dataset that is used for the statistical analysis can be found in `~data/final/`. The notebook `src/005_statistical_analysis_R.ipynb` can still be used as it only relies on that dataset.

For access to the raw data, please get in contact with the corresponding author of the paper.
