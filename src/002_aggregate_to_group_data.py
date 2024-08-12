# # Aggregate Single-Subject Data
#
# Purpose of this script is to load and aggregate the single-subject data to create a single dataset that contains the trial-level aggregated baeline corrected pupil dilation data of all participants.
# Inputs: ~data/interim/adults/*cleaned.parquet
# Outputs: ~data/processed/adults/group_data.xlsx



# fmt: off
from pathlib import Path
import sys
sys.path.append(str(Path().cwd().parent/'src'))
import modules.utils as utils
import pandas as pd 
# fmt:on

if __name__ == '__main__':
    #################
    ###### I/O ######
    #################
    print("Setting up I/O")

    CURRENT_FILEPATH = Path(__file__)
    DATA_DIR = CURRENT_FILEPATH.parent.parent / "data"
    INTERIM_DATA_DIR = DATA_DIR / "interim" / "adults"
    INTERIM_REPORTS = Path.cwd() / "interim_reports"

    DATA_OUTPUT_DIR = DATA_DIR / "processed" / "adults"
    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    COL_SELECT = [
        "ParticipantName",
        "trial_nr",
        "Emotion",
        "Picture",
        "%samples_fixation",
        "%samples_stimulus",
    ]  # which columns to include in the aggregated data

    SPLIT_TIME_MS = 1500  # Only use data in stimulus period post SPLIT_TIME_MS

    ##################
    ###### main ######
    ##################
    # Note that we aggregate PD values 800ms post-stimulus onset
    filepath_generator = INTERIM_DATA_DIR.glob("*cleaned.parquet")
    dataset_loader = map(pd.read_parquet, filepath_generator)
    dataset_cleaner = map(lambda x: utils.prepare_dataset(
        x, split_time_ms=SPLIT_TIME_MS), dataset_loader)
    dataset_aggregator = map(
        lambda df: utils.aggregate_single_subject_data(
            df, COL_SELECT), dataset_cleaner
    )

    group_df = pd.concat(dataset_aggregator)  # execute
    group_df = group_df.rename(
        columns={"pupilsize_baseline_corrected": f"avg_pd_bc_post{SPLIT_TIME_MS}ms"}
    )

    # Save the aggregated data
    group_df.to_excel(DATA_OUTPUT_DIR / "group_data.xlsx", index=False)
    print("Done")
