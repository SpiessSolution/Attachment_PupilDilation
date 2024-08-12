# # ADULT DATA: SingleSubject level preprocessing and segmentation
#
#
# ### Purpose
# Purpose of the script is to load a single-subject files, preprocess the data and export each as a new dataset that can later be combined.
# Inputs: ~data/external/adults/eyetracker/*tsv, ~data/external/adults/stimfiles/*txt
# Outputs: ~data/interim/adults/*cleaned.parquet


# fmt: off
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import modules.utils as utils
# fmt: on


if __name__ == '__main__':
    #################
    ###### I/O ######
    #################
    print("Setting up I/O")

    CURRENT_FILEPATH = Path(__file__)
    EXTERNAL_DATA_DIR = CURRENT_FILEPATH.parent.parent / "data" / "external" / "adults"
    STIMULUS_FILES_FOLDER = EXTERNAL_DATA_DIR / "stimfiles"
    EYETRACKER_FILES_FOLDER = EXTERNAL_DATA_DIR / "eyetracker"
    OUTPUT_DIR = CURRENT_FILEPATH.parent.parent / "data" / "interim" / "adults"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PICTURE_OUTPUT_DIR = (
        Path.cwd().parent / "reports" / "figures" / "SingleSubject_QA" / "adults"
    )
    SAMPLE_DATA_DIR = CURRENT_FILEPATH.parent / "data" / "sample_data"
    # Get all filepaths from the corresponding folders
    eyetracker_filepaths_list = list(EYETRACKER_FILES_FOLDER.glob("*.tsv"))
    stimfiles_filepath_list = list(STIMULUS_FILES_FOLDER.glob("*.txt"))

    print(f"{len(eyetracker_filepaths_list)} eyetracker files found")
    print(f"{EYETRACKER_FILES_FOLDER =}")
    ##################
    ###### main ######
    ##################
    # Process each subject individually
    for counter, eyetracker_fp in enumerate(eyetracker_filepaths_list):
        # Load data
        subject_id = utils.extract_subjectid_from_filename(
            eyetracker_fp.parts[-1])
        print(f"Working on subjectid {subject_id}, index = {counter}")
        matched_stim_file = [
            str(x) for x in stimfiles_filepath_list if str(subject_id) in str(x)
        ][0]
        eye_df = utils.load_eye_tracker_file_adults(eyetracker_fp)
        stimfile_df = utils.load_stim_file_adults(matched_stim_file)
        # check whether the eyetracker and stimfile match
        input_files_match = utils.check_match_eyetracker_stimfile(
            eye_df, stimfile_df)
        if not input_files_match:
            raise ValueError(
                f"eyetracker and stimfile for subjectid {subject_id} do not match"
            )
        # Preprocess
        segmented_eye_clean_df = utils.preprocess_single_subject_data(
            eyetracker_df=eye_df.copy(), remove_outliers=True
        )
        # Enrich with information from stimulus file
        segmented_eye_clean_df = segmented_eye_clean_df.merge(
            stimfile_df[["Trial_Nr", "ItemID",
                         "Emotion", "Picture", "Marker"]],
            left_on="trial_nr",
            right_on="Trial_Nr",
            how="left",
        )
        # Enrich with other information
        segmented_eye_clean_df = utils.add_percentage_datapoints_per_trial(
            segmented_eye_clean_df
        )
        # perform baseline correction and add relative trial time
        segmented_eye_clean_df = utils.perform_baseline_correction(
            segmented_eye_clean_df)
        segmented_eye_clean_df = utils.add_relative_trial_time(
            segmented_eye_clean_df)
        # Drop columns that are not needed
        segmented_eye_clean_df = segmented_eye_clean_df.drop(
            columns=[
                "RecordingTimestamp",
                "EyeTrackerTimestamp",
                "ExternalEventIndex",
                "ExternalEvent",
                "ExternalEventValue",
                "PupilLeft_dist",
                "PupilRight_dist",
                "difference_avg_left_min_right_pupil",
                "PupilLeft_change",
                "PupilRight_change",
                "sample_count_fixation",
                "sample_count_stimulus",
                "ValidityLeft",
                "ValidityRight",
                "PupilMean_uncorrected",
                "Trial_Nr",
            ]
        )
        # General cleanup
        segmented_eye_clean_df = segmented_eye_clean_df.infer_objects()
        segmented_eye_clean_df = segmented_eye_clean_df.astype(
            {"trial_nr": "int32", "RelativeTime_ms": "int32"})

        # export dataset
        fln = OUTPUT_DIR / f"A{subject_id}_cleaned.parquet"
        segmented_eye_clean_df.to_parquet(fln)
