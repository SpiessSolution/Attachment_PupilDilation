# # Join Datasets
#
# Purpose of this script is to combine all three datasets into a single dataset.
# In particular, we will combine:
# - `~data/processed/adults/group_data.xlsx`
# - `~data/processed/stimulus_luminance_values.xlsx`
# - `~data/external/adults/Pupilproject_behdata.csv`
#
# The output dataset will be saved in `~data/final/adults_final_data.xlsx`


# fmt: off
from pathlib import Path
import pandas as pd
# fmt: on


if __name__ == '__main__':
    #################
    ###### I/O ######
    #################
    print("Setting up I/O")

    CURRENT_FILEPATH = Path(__file__)
    DATA_DIR = CURRENT_FILEPATH.parent.parent / "data"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    EXTERNAL_ADULT_DATA_DIR = DATA_DIR / "external" / "adults"
    OUTPUT_DATA_DIR = DATA_DIR / "final"
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ## Support Functions

    def extract_digits(s):
        return ''.join(filter(str.isdigit, s))

    ##################
    ###### main ######
    ##################

    # ## Load data
    eyetracker_df = pd.read_excel(
        PROCESSED_DATA_DIR / "adults" / "group_data.xlsx")
    luminance_df = pd.read_excel(
        PROCESSED_DATA_DIR / "stimulus_luminance_values.xlsx")
    behavioral_df = pd.read_csv(
        EXTERNAL_ADULT_DATA_DIR / "Pupilproject_behdata.csv", delimiter=";")

    # ## Combine datasets
    eyetracker_df.loc[:, "ParticipantName"] = eyetracker_df["ParticipantName"].apply(
        extract_digits).astype(int)
    eyetracker_df

    eyetracker_lum_beh_df = (
        eyetracker_df
        .merge(luminance_df, left_on="Picture", right_on="filename", how="left")
        .merge(behavioral_df, left_on="ParticipantName", right_on="Participant", how="left")
    ).drop(columns=["filename", "Participant"])
    eyetracker_lum_beh_df.head()

    # save final dataset
    eyetracker_lum_beh_df.sort_values(by=["ParticipantName", "trial_nr"]).to_excel(
        OUTPUT_DATA_DIR/"adults_final_data.xlsx", index=False)
    print("Done")
