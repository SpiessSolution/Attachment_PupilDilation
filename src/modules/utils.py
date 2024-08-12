# fmt: off
from typing import Union, Tuple, List
from datetime import timedelta
from math import inf
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from pandas_schema import Column, Schema
from pandas_schema.validation import CanConvertValidation, InListValidation, InRangeValidation, MatchesPatternValidation, _BaseValidation, IsDtypeValidation, CustomElementValidation
from pandas import DataFrame, Timestamp, Timedelta
from statsmodels.nonparametric.smoothers_lowess import lowess
from PIL import Image

# fmt:on
##################################################
############ input data validation ###############
##################################################


EYETRACKER_ADULTS_COLS = [
    "StudioProjectName", "ParticipantName", "RecordingDate", "RecordingTimestamp",
    "LocalTimeStamp", "EyeTrackerTimestamp", "ExternalEventIndex", "ExternalEvent",
    "ExternalEventValue", "PupilLeft", "PupilRight", "ValidityLeft", "ValidityRight",
]


stim_file_schema_adults = Schema([
    Column('Participant_Nr', [IsDtypeValidation(
        int), InRangeValidation(1, 120), ]),
    Column('Trial_Nr', [IsDtypeValidation(int), InRangeValidation(1, 229)]),
    Column('ItemID', [IsDtypeValidation(int)]),
    Column('Emotion', [MatchesPatternValidation(r'^(happy|neutral|sad)$')]),
    Column('Picture', [MatchesPatternValidation(
        r'^\d+-(neutral|happy|sad)\.jpg$')]),
    Column('Marker', [IsDtypeValidation(int), InRangeValidation(1, 300)]),
    Column('Beep_Start_Time', [IsDtypeValidation(
        int), InRangeValidation(1, inf)]),
    Column('Fixation_Start_Time', [
           IsDtypeValidation(int), InRangeValidation(1, inf)]),
    Column('Picture_Start_Time', [
           IsDtypeValidation(int), InRangeValidation(1, inf)]),
    Column('ITI_Start_Time', [IsDtypeValidation(
        int), InRangeValidation(1, inf)]),
    Column('iBlockNumber', [IsDtypeValidation(int), InRangeValidation(1, 4)]),
])


def is_valid_date_format(date_string):
    try:
        pd.to_datetime(date_string, format='%m/%d/%Y')
        return True
    except ValueError:
        return False


def is_valid_timestamp_format(timestamp_string):
    try:
        pd.to_datetime(timestamp_string, format='%H:%M:%S.%f')
        return True
    except ValueError:
        return False


valid_date_validator = CustomElementValidation(
    lambda d: is_valid_date_format(d),
    "is not a valid date in the format MM/DD/YYYY"
)

valid_timestamp_validator = CustomElementValidation(
    lambda t: is_valid_timestamp_format(t),
    "is not a valid timestamp in the format HH:MM:SS.sss"
)
object_type_validation = IsDtypeValidation(object)
eyetracker_file_schema_adults = Schema([
    Column('StudioProjectName', [
           MatchesPatternValidation('Children_S_AQSEMG_Stefania')]),
    Column('ParticipantName', [MatchesPatternValidation(r'^A\d+$')]),
    Column('RecordingDate', [valid_date_validator]),
    Column('RecordingTimestamp', [IsDtypeValidation(
        int), InRangeValidation(1, np.inf)]),
    Column('LocalTimeStamp',  [valid_timestamp_validator]),
    Column('EyeTrackerTimestamp', [IsDtypeValidation(
        float)]),
    Column('ExternalEventIndex', [IsDtypeValidation(float)]),
    Column('ExternalEvent', [IsDtypeValidation(object)]),
    Column('ExternalEventValue', [IsDtypeValidation(float)]),
    Column('PupilLeft', [IsDtypeValidation(float)]),
    Column('PupilRight', [IsDtypeValidation(float)]),
    Column('ValidityLeft', [IsDtypeValidation(float),
           InListValidation([0, 1, 2, 3, 4, np.nan])]),
    Column('ValidityRight', [IsDtypeValidation(float),
                             InListValidation([0, 1, 2, 3, 4, np.nan])]),
])


def load_eye_tracker_file_adults(file_path, select_cols: list = EYETRACKER_ADULTS_COLS, skip_validation: bool = False):
    """
    Load eye tracker data from a file for adults.

    Args:
        file_path (str): The path to the eye tracker file.
        select_cols (list, optional): The columns to select from the file. Defaults to EYETRACKER_ADULTS_COLS.
        skip_validation (bool, optional): Whether to skip data validation. Defaults to False.

    Returns:
        pandas.DataFrame: The loaded eye tracker data.
    """
    df = pd.read_csv(file_path, sep='\t')[select_cols]
    if not skip_validation:
        errors = eyetracker_file_schema_adults.validate(df)
        if errors:
            for error in errors:
                print(error)
            raise ValueError(f'Invalid file {file_path}')
    return df


def load_stim_file_adults(file_path, skip_validation: bool = False):
    """
    Load a stimulus file for adults.

    Args:
        file_path (str): The path to the stimulus file.
        skip_validation (bool, optional): Whether to skip validation of the file. Defaults to False.

    Returns:
        pandas.DataFrame: The loaded stimulus data.

    Raises:
        ValueError: If the file is invalid and skip_validation is False.
    """
    df = pd.read_csv(file_path, sep='\t')
    if not skip_validation:
        errors = stim_file_schema_adults.validate(df)
        if errors:
            for error in errors:
                print(error)
            raise ValueError(f'Invalid file {file_path}')
    return df


##################################################
########### Preprocessing input data #############
##################################################
def basic_preprocess_eyetracking_adult(df) -> pd.DataFrame:
    """
    Preprocesses eyetracking data specifically tailored for studies involving adult participants. This process involves
    several steps to clean and prepare the data for further analysis:

    1. Converts the 'LocalTimeStamp' column to datetime format for accurate time representation.
    2. Filters out rows with missing pupil data unless there is a trigger event, ensuring data completeness.
    3. Marks pupil data as NaN where validity codes indicate unreliable measurements, to prevent skewed analysis.
    4. Removes pupil size values that fall outside of predefined acceptable ranges, to exclude physiologically implausible values.

    The function aims to ensure the dataset is clean, focusing on reliable and relevant eyetracking measures from adults,
    making it suitable for detailed analysis of gaze patterns, attention, and other cognitive metrics derived from eyetracking data.

    Args:
        df (pd.DataFrame): The input dataframe containing raw eyetracking data from adult participants.

    Returns:
        pd.DataFrame: The preprocessed dataframe, ready for analysis, with irrelevant, invalid, or erroneous data points removed or corrected.
    """

    df["LocalTimeStamp"] = pd.to_datetime(
        df["LocalTimeStamp"], format='%H:%M:%S.%f')  # .dt.time
    # remove rows with missing pupil data and anything before the first trial
    condition_keep_row = (
        # either one of the pupils has valid data
        ((df["ValidityLeft"] == 0) | (df["ValidityRight"] == 0))
        # or there is a trigger event (in which case both pupils are invalid)
        | ~pd.isnull(df["ExternalEvent"])
    )
    df = df[condition_keep_row]
    # set pupil values for invalid samples to np.nan
    df.loc[df["ValidityLeft"] != 0, "PupilLeft"] = np.nan
    df.loc[df["ValidityRight"] != 0, "PupilRight"] = np.nan
    # remove pupil values outside of a reasonable range
    df = remove_pupil_dil_vales_outside_range(df)
    return df


##################################################
########## Pupil dilation QA functions ###########
##################################################

def preprocess_single_subject_data(
    eyetracker_df: pd.DataFrame, remove_outliers: bool
) -> pd.DataFrame:
    """
    Preprocesses the eyetracker data for a single subject.

    Args:
        eyetracker_df (pd.DataFrame): The eyetracker data for a single subject.
        remove_outliers (bool): Flag indicating whether to remove outliers.

    Returns:
        pd.DataFrame: The preprocessed and segmented eyetracker data.
    """
    # Preprocess and QA
    eye_processed_df = basic_preprocess_eyetracking_adult(eyetracker_df)
    eye_processed_df = remove_pupil_dil_vales_outside_range(
        eye_processed_df)

    # remove outliers in dilation speed
    eye_processed_df = mark_outliers_in_pupil_dilation_speed_based_on_MAD(
        eye_processed_df, k=3
    )
    if remove_outliers:
        eye_processed_df = remove_dilation_speed_outliers(
            eye_processed_df)

    # # remove samples near gaps
    eye_processed_df = mark_samples_near_gaps(eye_processed_df)
    if remove_outliers:
        eye_processed_df = remove_pupil_dilation_near_gaps(
            eye_processed_df)

    # Segment
    segmented_eye_df = segment_dataset(
        eye_processed_df, fixation_duration_ms=1000, stimulus_duration_ms=2000
    )
    segmented_eye_df = segmented_eye_df.drop_duplicates()

    # remove samples too far away from trendline after segmenting
    segmented_eye_df = mark_pupil_trendline_outliers(
        segmented_eye_df, k=8, loess_fraction=1 / 4
    )
    if remove_outliers:
        segmented_eye_df = remove_trend_outliers(segmented_eye_df)

    # remove records in the df that do not contain any valid pupil data anymore (after the transformations)
    if remove_outliers:
        segmented_eye_df = segmented_eye_df.dropna(
            subset=["PupilLeft", "PupilRight"], how="all"
        )

    # Compute corrected mean pupil dilation
    segmented_eye_df = add_corrected_mean_pupil_size(segmented_eye_df)

    return segmented_eye_df


def remove_pupil_dil_vales_outside_range(df: pd.DataFrame, min_pupil_dilation_mm: float = 1.5, max_pupil_dilation_mm: float = 9) -> pd.DataFrame:
    """
    Removes pupil dilation values outside the specified range from the given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing pupil dilation data.
        min_pupil_dilation_mm (float, optional): The minimum allowed pupil dilation value in millimeters. Defaults to 1.5.
        max_pupil_dilation_mm (float, optional): The maximum allowed pupil dilation value in millimeters. Defaults to 9.

    Returns:
        pd.DataFrame: The DataFrame with pupil dilation values outside the range replaced with NaN.

    See also:
        Kret et al., 2014, as cited in Kre and & Sjak-Shie, 2018, p. 3.; doi: https://doi.org/10.3758/s13428-018-1075-y
    """
    df.loc[df["PupilLeft"] < min_pupil_dilation_mm, "PupilLeft"] = np.nan
    df.loc[df["PupilLeft"] > max_pupil_dilation_mm, "PupilLeft"] = np.nan
    df.loc[df["PupilRight"] < min_pupil_dilation_mm, "PupilRight"] = np.nan
    df.loc[df["PupilRight"] > max_pupil_dilation_mm, "PupilRight"] = np.nan

    return df


def mark_outliers_in_pupil_dilation_speed_based_on_MAD(df: pd.DataFrame, k: Union[float, int] = 3) -> pd.DataFrame:
    """
    Mark outliers in pupil dilation speed based on Median Absolute Deviation (MAD).

    Args:
        df (pd.DataFrame): The input DataFrame containing pupil dilation data.
        k (Union[float, int], optional): The scaling factor to use for thresholding. Defaults to 3.

    Returns:
        pd.DataFrame: The input DataFrame with additional columns indicating outliers in pupil dilation speed.

    See also:
    https://doi.org/10.3758/s13428-018-1075-y for details
    """
    # add two columns to the dataframe containing the time-normalized pupil change/speed
    df = add_pupil_change_to_df(df)
    # compute the threshold for pupil change based on MAD
    threshold = compute_threshold_for_pupil_change_based_on_MAD(df, k)
    # mark outliers in pupil dilation speed
    df = (df
          .assign(PupilLeft_dilation_speed_outlier=np.where(df["PupilLeft_change"] > threshold["PupilLeft_change"], 1, 0))
          .assign(PupilRight_dilation_speed_outlier=np.where(df["PupilRight_change"] > threshold["PupilRight_change"], 1, 0))
          )
    return df


def mark_samples_near_gaps(df: pd.DataFrame, gapsize_ms: int = 75, ms_to_remove_before_after_gap: int = 50) -> pd.DataFrame:
    """
    Marks pupil dilation records from a pandas DataFrame that are within a specified timeframe before and after gaps in the data,
    where a gap is defined as a period with missing data longer than a given threshold defined by gapsize_ms. 

    The implementation is informed by the guidelines proposed by Kret et al., 2014, as cited in Kre and & Sjak-Shie, 2018, p. 4.
    DOI: https://doi.org/10.3758/s13428-018-1075-y. These guidelines recommend excluding data points close to gaps as they often reveal a slope change.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the eye-tracking data with a 'LocalTimeStamp' column.
    - gapsize_ms (int, optional): The minimum duration (in milliseconds) of a break in data collection to be considered a gap. Default is 75 ms.
    - ms_to_remove_before_after_gap (int, optional): The duration (in milliseconds) before and after each gap during which data points will be excluded. Default is 50 ms.

    Returns:
    - pd.DataFrame: A DataFrame with the same structure as the input but with an additional 'sample_near_gap' column that marks rows for exclusion.

    Note:
    - The 'LocalTimeStamp' column must be in pandas datetime format for the function to correctly calculate time differences.
    - The function assumes that the DataFrame is sorted by the 'LocalTimeStamp' column in ascending order.
    - The function excludes samples on the record level, not at the left/right eye level
    - The function should be applied on the segmented data but is not applied to segments individually
    """

    # Calculate the time difference between consecutive samples in milliseconds
    df['LocalTimeStampDiff'] = df['LocalTimeStamp'].diff(
    ).shift(-1).dt.total_seconds() * 1000
    # Initialize 'Exclude' column to False
    df['sample_near_gap'] = False

    # Iterate through the DataFrame to find gaps and mark rows within 50ms before and after the gap
    # Exclude the last row to prevent index out of bounds
    for index in range(len(df) - 1):
        row = df.iloc[index]
        if row['LocalTimeStampDiff'] > gapsize_ms:  # If the current row precedes a gap
            # Mark rows within 50ms before the gap. We need to check backward from the current row
            prev_index = index - 1
            while prev_index >= 0:
                prev_row = df.iloc[prev_index]
                if (row['LocalTimeStamp'] - prev_row['LocalTimeStamp']).total_seconds() * 1000 <= ms_to_remove_before_after_gap:
                    df.at[df.index[prev_index], 'sample_near_gap'] = True
                    prev_index -= 1
                else:
                    break
            # Mark the current row as it's immediately before a gap and rows after the gap
            # Current row before the gap
            df.at[df.index[index], 'sample_near_gap'] = True
            next_index = index + 1
            while next_index < len(df):
                next_row = df.iloc[next_index]
                if (next_row['LocalTimeStamp'] - row['LocalTimeStamp']).total_seconds() * 1000 <= ms_to_remove_before_after_gap + row['LocalTimeStampDiff']:
                    df.at[df.index[next_index], 'sample_near_gap'] = True
                    next_index += 1
                else:
                    break
    df = df.drop(columns=["LocalTimeStampDiff"])

    return df


def mark_pupil_trendline_outliers(df: pd.DataFrame,  k: Union[float, int] = 8, loess_fraction: float = 1/4) -> pd.DataFrame:
    """
    Marks outliers in the pupil data based on the distance between each pupil measurement and the trendline.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the pupil data.
        k (Union[float, int]): The number of standard deviations to use for determining outliers. Default is 3.
        loess_fraction (float): The fraction of data to use for the LOESS smoother. Default is 1/200.

    Returns:
        pd.DataFrame: The input DataFrame with additional columns indicating whether each pupil measurement is an outlier.
    """
    df = add_loess_smoother_per_trial(df, fraction=loess_fraction)
    df = add_pupil_distance_to_trendline(df)
    threshold = compute_threshold_for_pupil_distance_to_trendline_based_on_MAD(
        df, k=k)
    df = df.assign(
        PupilLeft_trend_outlier=df["PupilLeft_dist"] > threshold["PupilLeft_dist"],
        PupilRight_trend_outlier=df["PupilRight_dist"] > threshold["PupilRight_dist"]
    )
    return df
##################################################
############ Segmentation functions ##############
##################################################


def extract_data_in_time_window(df: DataFrame, start_time: Timestamp, end_time: Timestamp, inclusive: str = "right") -> DataFrame:
    """
    Extracts data from a pandas dataframe within a specified time window.

    Parameters:
    df (pandas.DataFrame): The dataframe containing eye tracking data.
    start_time (pandas._libs.tslibs.timestamps.Timestamp): The start time of the time window (exclusive).
    end_time (pandas._libs.tslibs.timestamps.Timestamp): The end time of the time window (inclusive).
    inclusive (str, optional): The side of the interval to make inclusive. Defaults to "right". Can be "both", "neither","left","right"

    Returns:
    pandas.DataFrame: The subset of the dataframe within the specified time window.
    """
    return df[(df.LocalTimeStamp.between(start_time, end_time, inclusive=inclusive))].copy()


def add_milliseconds_to_timestamp(milliseconds: int, start_time: Timestamp, ) -> Timestamp:
    """
    Adds a specified number of milliseconds to a given timestamp.

    Parameters:
    milliseconds (int): The number of milliseconds to add.
    start_time (pandas._libs.tslibs.timestamps.Timestamp): The start time.

    Returns:
    pandas._libs.tslibs.timestamps.Timestamp: The new time.

    Note:
    Add negative numbers to go back in time
    """
    return start_time + pd.Timedelta(milliseconds, unit="ms")


def extract_fixation_segment(df: pd.DataFrame, fixation_onset_time: Timestamp, fix_duration_ms: int) -> pd.DataFrame:
    """
    Extracts a segment of data from a DataFrame corresponding to a fixation event.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        fixation_onset_time (Timestamp): The timestamp indicating the onset time of the fixation.
        fix_duration_ms (int): The duration of the fixation in milliseconds.

    Returns:
        pd.DataFrame: The extracted segment of data corresponding to the fixation event.
    """
    fixation_end_time = add_milliseconds_to_timestamp(
        fix_duration_ms, fixation_onset_time)
    fixation_segment_df = extract_data_in_time_window(
        df, start_time=fixation_onset_time, end_time=fixation_end_time, inclusive="left")
    fixation_segment_df = fixation_segment_df.assign(segment_type="fixation")
    return fixation_segment_df


def extract_stimulus_segment(df, stimulus_onset_time: Timestamp, stimulus_duration_ms: int) -> pd.DataFrame:
    """
    Extracts a segment of data from a DataFrame based on the stimulus onset time and duration.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        stimulus_onset_time (Timestamp): The timestamp indicating the onset time of the stimulus.
        stimulus_duration_ms (int): The duration of the stimulus in milliseconds.

    Returns:
        pd.DataFrame: The extracted segment of data including a column indicating for each row whether it is in the fixation or stimulus period

    """
    stimulus_end_time = add_milliseconds_to_timestamp(
        stimulus_duration_ms, stimulus_onset_time)
    stimulus_segment_df = extract_data_in_time_window(
        df, start_time=stimulus_onset_time, end_time=stimulus_end_time, inclusive="right")
    stimulus_segment_df = stimulus_segment_df.assign(segment_type="stimulus")
    return stimulus_segment_df


def segment_individual_trial(df: pd.DataFrame, assign_trial_nr: int, trigger_onset_time: Timestamp, fixation_duration_ms: int, stimulus_duration_ms: int) -> pd.DataFrame:
    """
    Function to extract from a trial the baseline period dataframe and the stimulus period data (i.e., the rows from df)

    Args:
        df (pd.DataFrame): The input dataframe containing the trial data.
        assign_trial_nr (int): The trial number to assign to the trial.
        trigger_onset_time (Timestamp): The timestamp marking the effective beginning of the trial.
        fixation_duration_ms (int): The duration of the fixation period in milliseconds.
        stimulus_duration_ms (int): The duration of the stimulus period in milliseconds.

    Returns:
        pd.DataFrame: The output dataframe including only the segments of the trial.
    """
    fixation_onset_time = add_milliseconds_to_timestamp(
        -1*fixation_duration_ms, trigger_onset_time)

    fixation_segment_df = extract_fixation_segment(
        df, fixation_onset_time, fixation_duration_ms)

    stimulus_segment_df = extract_stimulus_segment(
        df, trigger_onset_time, stimulus_duration_ms)

    trial_segment_df = pd.concat([
        fixation_segment_df, stimulus_segment_df
    ])

    trial_segment_df = trial_segment_df.assign(trial_nr=assign_trial_nr)
    return trial_segment_df


def segment_dataset(df: pd.DataFrame, fixation_duration_ms: int = 1000, stimulus_duration_ms: int = 2000, ExternalEvent_indicator_string: str = "TriggerData") -> pd.DataFrame:
    """
    Segments the dataset based on trigger onset times.

    Args:
        df (pd.DataFrame): The input dataframe.
        fixation_duration_ms (int, optional): The duration of fixation in milliseconds. Defaults to 500. Time-locked to the trigger event.
        stimulus_duration_ms (int, optional): The duration of stimulus in milliseconds. Defaults to 2000. 
        ExternalEvent_indicator_string (str, optional): The string indicating the external event. Defaults to "TriggerData". Fixation and Stimulus period will be locked to this event in terms of LocalTimeStamp

    Returns:
        pd.DataFrame: The segmented dataset.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The input should be a pandas DataFrame")
    if not isinstance(fixation_duration_ms, int):
        raise ValueError("The fixation duration should be an integer")
    if not isinstance(stimulus_duration_ms, int):
        raise ValueError("The stimulus duration should be an integer")
    if not isinstance(ExternalEvent_indicator_string, str):
        raise ValueError(
            "The ExternalEvent_indicator_string should be a string")

    # Create a df that only contains the trigger onset times
    trigger_onset_times_df = df[df["ExternalEvent"]
                                == ExternalEvent_indicator_string]

    # Segment all the trials in the dataset
    segmented_data = list()
    for index, trigger_onset_row in trigger_onset_times_df.iterrows():
        trial_onset_time = trigger_onset_row["LocalTimeStamp"]
        trial_nr = trigger_onset_row["ExternalEventIndex"]
        trial_segment_df = (
            segment_individual_trial(
                df=df, trigger_onset_time=trial_onset_time, assign_trial_nr=trial_nr, fixation_duration_ms=fixation_duration_ms, stimulus_duration_ms=stimulus_duration_ms)
        )
        segmented_data.append(trial_segment_df)
    # Concatenate all the segmented trials
    segmented_data_df = pd.concat(segmented_data, ignore_index=True)
    segmented_data_df = segmented_data_df.sort_values(by="LocalTimeStamp")
    return segmented_data_df


##################################################
############### Calc.  functions #################
##################################################

def add_relative_trial_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds relative trial time to the DataFrame. Time will be calculated relative to the onset of the stimulus 
    and fixation periods and returned as a new column RelativeTime_ms. The time is in milliseconds.   

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.

    Returns:
        pd.DataFrame: The DataFrame with the added relative trial time column.

    Note: 
        RelativeTime_ms will be calculated for rows where segment_type is 'stimulus' or 'fixation'.
        It will be NaN for other segment_type values.
    """
    df_copy = df.copy()

    # Ensure LocalTimeStamp is in datetime format if not already
    df_copy['LocalTimeStamp'] = pd.to_datetime(df_copy['LocalTimeStamp'])

    # Identify rows where segment_type is 'stimulus' or 'fixation'
    segment_condition = df_copy['segment_type'].isin(['stimulus', 'fixation'])

    # Create a column to store the relative time in milliseconds initialized with NaN
    df_copy['RelativeTime_ms'] = pd.NA

    # Calculate the time difference within each trial for the specified segment types
    for segment in ['stimulus', 'fixation']:
        condition = df_copy['segment_type'] == segment
        time_diff = df_copy.loc[condition].groupby(
            'trial_nr')['LocalTimeStamp'].diff().fillna(pd.Timedelta(seconds=0))
        time_diff_seconds = time_diff.dt.total_seconds() * 1000  # Convert to milliseconds

        # Assign the results back using loc
        df_copy.loc[condition, 'RelativeTime_ms'] = time_diff_seconds.groupby(
            df_copy[condition]['trial_nr']).cumsum().fillna(0)

    return df_copy


def perform_baseline_correction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform baseline correction on the given DataFrame. The average pupil size furing fixation is calculated based on the corrected 
    mean (left+right) pupil size. The average pupil size during fixation is then subtracted from the corrected mean pupil size during stimulus period.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        pd.DataFrame: The DataFrame with the baseline corrected data. A new column
        'pupilsize_baseline_corrected' is added to the DataFrame.

    """
    fixation_pupil_avg_df = (
        df[df.segment_type == "fixation"]
        .groupby("trial_nr")
        .aggregate({"PupilMean_corrected": np.mean})
        .rename(columns={"PupilMean_corrected": "FixAvg_PupilMean_corr"})
    )  # actually, i'd rather prefer the median
    df = df.merge(fixation_pupil_avg_df, on="trial_nr", how="left")
    # Perform correction
    df.loc[df.segment_type == "stimulus", ["pupilsize_baseline_corrected"]] = (
        df.PupilMean_corrected - df.FixAvg_PupilMean_corr
    )

    return df


def add_avg_pupil_dilation_per_trial_and_eye(df: DataFrame) -> DataFrame:
    """
    Adds average pupil dilation per trial and eye to the input DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame containing pupil dilation data.

    Returns:
    DataFrame: The modified DataFrame with average pupil dilation per trial and eye added.
    """
    df = (
        df
        .merge(
            df.groupby("trial_nr")[["PupilLeft", "PupilRight"]].mean().rename(columns={"PupilLeft": "PupilLeft_trial_avg", "PupilRight": "PupilRight_trial_avg"}), on="trial_nr", how="left"
        )
    )
    return df


def add_corrected_mean_pupil_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds corrected mean pupil size to the DataFrame based on left and right eye pupil dilation.

    This function processes a DataFrame containing pupil size measurements from both eyes across
    different records and trials. It computes the uncorrected mean pupil size for each record, 
    adjusts these values based on the per-trial average pupil dilation differences between the eyes,
    and then applies corrections where data from one eye is missing.

    The corrections are based on the assumption that the mean pupil size should be adjusted by half 
    the average difference in pupil size between the left and right eyes, to account for missing data 
    from one eye.

    Parameters:
    - df (pd.DataFrame): A DataFrame containing at least the columns 'PupilLeft' and 'PupilRight' 
      with raw pupil sizes for the left and right eyes, respectively.

    Returns:
    - pd.DataFrame: The original DataFrame with added columns:
      - 'PupilMean_uncorrected': The mean of 'PupilLeft' and 'PupilRight' for each record.
      - 'PupilMean_corrected': The corrected mean pupil size for each record, adjusted for 
        per-trial differences and missing data.
      - 'difference_avg_left_min_right_pupil': The per-trial difference between the average 
        left-eye and right-eye pupil dilation.

    Note:
    The function requires an external function `add_avg_pupil_dilation_per_trial_and_eye` to compute 
    average pupil size per trial for each eye, which should be defined elsewhere and capable of 
    handling the input DataFrame appropriately.

    Example:
    ```
    df = pd.DataFrame({
        'PupilLeft': [3, np.nan, 5, 4],
        'PupilRight': [4, 5, np.nan, 4]
    })
    corrected_df = add_corrected_mean_pupil_size(df)
    print(corrected_df.columns)
    ```
    """
    # Raw uncorrected mean (left/right eye) pupil dilation per record
    df = df.assign(PupilMean_uncorrected=df[[
                   "PupilLeft", "PupilRight"]].mean(axis=1))
    # Average pupil size per trial
    df = add_avg_pupil_dilation_per_trial_and_eye(df)
    # per-trial difference between average left-eye and right-eye pupil dilation
    df = df.assign(
        difference_avg_left_min_right_pupil=df["PupilLeft_trial_avg"] - df["PupilRight_trial_avg"])
    # initialize with uncorrected values
    # initialize with uncorrected values
    df = df.assign(PupilMean_corrected=df["PupilMean_uncorrected"])
    # perform correction where data is missing from one eye
    df.loc[pd.isnull(df["PupilLeft"]), "PupilMean_corrected"] = df["PupilMean_uncorrected"] + \
        (0.5 * df["difference_avg_left_min_right_pupil"])
    df.loc[pd.isnull(df["PupilRight"]), "PupilMean_corrected"] = df["PupilMean_uncorrected"] - \
        (0.5 * df["difference_avg_left_min_right_pupil"])

    return df


def add_percentage_datapoints_per_trial(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds percentage of collected datapoints per trial for specific segments to the DataFrame.

    This function calculates the percentage of available datapoints for two predefined segments 
    ('fixation' and 'stimulus') in each trial based on a fixed maximum number of expected samples. 
    The percentages are calculated by dividing the actual count of 'PupilMean_corrected' datapoints 
    for each segment type by the maximum possible samples for that segment, then multiplying by 100 
    to convert to a percentage. These percentages are added to the original DataFrame as new columns 
    for each segment type.

    Parameters:
    - df (pd.DataFrame): A DataFrame containing eye-tracking data with columns 'trial_nr' to identify 
      each trial, 'segment_type' to classify segments within a trial, and 'PupilMean_corrected' for 
      the datapoints of interest.

    Returns:
    - pd.DataFrame: The original DataFrame augmented with two new columns:
        - '%samples_fixation': The percentage of 'PupilMean_corrected' datapoints collected during 
          the fixation segment out of the maximum expected samples.
        - '%samples_stimulus': The percentage of 'PupilMean_corrected' datapoints collected during 
          the stimulus segment out of the maximum expected samples.

    Note:
    The maximum number of samples for the 'fixation' segment is based on a duration of 1000 ms with 
    a sampling rate of 16 ms, and for the 'stimulus' segment on a duration of 2000 ms with the same 
    sampling rate. These values are hardcoded within the function and may need adjustment to match 
    specific experimental setups or sampling rates.

    Example:
    ```
    # Assuming df is your DataFrame with the necessary columns
    enhanced_df = add_percentage_datapoints_per_trial(df)
    print(enhanced_df.columns)  # To verify the addition of new percentage columns
    ```
    """
    # init
    max_samples_fixation_period = 1000//16
    max_samples_stimulus_period = 2000//16
    # count number of datapoints in PupilMean_corrected per trial
    sample_count = df.groupby(["trial_nr", "segment_type"])["PupilMean_corrected"].count(
    ).to_frame().reset_index().pivot(columns="segment_type", index="trial_nr").reset_index()
    sample_count.columns = ["trial_nr",
                            "sample_count_fixation", "sample_count_stimulus"]
    # calculate percentages
    sample_count.loc[:, "%samples_fixation"] = (
        sample_count["sample_count_fixation"] / max_samples_fixation_period)*100
    sample_count.loc[:, "%samples_stimulus"] = (
        sample_count["sample_count_stimulus"] / max_samples_stimulus_period)*100
    # add to original df
    df = (
        df.merge(sample_count, on="trial_nr", how="left")
    )
    return df


##################################################
############### Helper  functions ################
##################################################
def check_match_eyetracker_stimfile(eyetracker_df: DataFrame, stimfile_df: DataFrame) -> bool:
    """
    Check if the eyetracker and stimfile match.

    Parameters:
    eyetracker_df (DataFrame): DataFrame containing eyetracker data.
    stimfile_df (DataFrame): DataFrame containing stimfile data.

    Returns:
    bool: True if the eyetracker and stimfile match, False otherwise.
    """
    # subset the eyetracker data and the stimfile to the trigger events and sort the events by rownumber
    eye_subset_series = eyetracker_df[eyetracker_df["ExternalEvent"]
                                      == "TriggerData"]["ExternalEventValue"].sort_index().astype(int)
    stimfile_marker_series = stimfile_df["Marker"].iloc[:len(
        eye_subset_series)].astype(int)
    # subtract the marker values from the eyetracker EventValues. If the difference is 0, then the eyetracker and stimfile match
    return (eye_subset_series - stimfile_marker_series).sum() == 0


def extract_subjectid_from_filename(filepath: str) -> int:
    """
    Extracts the subject ID from the given file path.

    Args:
        filepath (str): The file path containing the subject ID.

    Returns:
        int: The extracted subject ID.

    Raises:
        AssertionError: If more than one number is found in the filename.
    """
    digits = re.findall(r"\d+", filepath)
    assert len(digits) == 1, "Error: more than one number found in filename"
    return int(digits[0])


def timestamp_to_relative_time_in_ms(timestamp_series: pd.Series) -> pd.Series:
    """
    Converts a series of timestamps into a series of cumulative relative times in milliseconds, facilitating the analysis 
    of time-series data by representing each timestamp relative to the first timestamp in the series. This function calculates 
    the difference between each timestamp and the preceding one, converts these differences into milliseconds, and then 
    accumulates these differences to express each timestamp as the amount of time passed since the start of the series.

    The first timestamp in the series is considered the start point (0 milliseconds), and all subsequent timestamps are 
    represented as the cumulative time in milliseconds from this start point. This approach is particularly useful in 
    contexts where the absolute timing of events is less relevant than the duration between them or their sequence, such as 
    analyzing the dynamics of events in a trial or the timing of responses in a dataset.

    Parameters:
        timestamp_series (pd.Series): A pandas Series object containing datetime or timestamp entries. It is assumed that 
        this series is sorted in ascending order, representing a sequence of events or measurements over time.

    Returns:
        pd.Series: A series of the same length as the input, where each value represents the cumulative time in milliseconds 
        from the first timestamp in the series. The first value will always be 0, indicating the start point, with subsequent 
        values indicating the time elapsed in milliseconds.

    Note:
        - This function assumes that the input series is sorted in chronological order.
        - The function treats the first timestamp as the reference point (0 milliseconds) and calculates the relative time for 
        each subsequent timestamp accordingly.
        - It is designed to handle time-series data where precise measurement of time intervals and their accumulation is 
        necessary for analysis.
    """
    def timedelta_to_timediff_in_ms(t: Timestamp) -> float:
        if isinstance(t, Timedelta):
            return t/timedelta(milliseconds=1)
        else:  # assume start of the trial
            return 0

    return timestamp_series.diff(1).apply(timedelta_to_timediff_in_ms).cumsum()


def create_lagged_df(df) -> pd.DataFrame:
    """
    Creates a lagged dataframe by shifting the columns of the input dataframe.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        pd.DataFrame: The lagged dataframe with shifted columns.
    """
    zero_shift_df = df[["PupilLeft", "PupilRight", "LocalTimeStamp"]]
    pre_shift_df = zero_shift_df.shift(1)
    post_shift_df = zero_shift_df.shift(-1)

    pre_shift_df.columns = [f"pre_{col}" for col in pre_shift_df.columns]
    post_shift_df.columns = [f"post_{col}" for col in post_shift_df.columns]
    lagged_df = pd.concat([pre_shift_df, zero_shift_df, post_shift_df], axis=1)

    return lagged_df


def milliseconds_diff(t1: pd.Series, t2: pd.Series):
    """
    Calculates the difference in milliseconds between two pandas Series objects including Timestamp data.

    Parameters:
    t1 (pd.Series): The first pandas Series object.
    t2 (pd.Series): The second pandas Series object.

    Returns:
    pd.Series: The difference between t1 and t2 in milliseconds.
    """
    return (t1 - t2)/timedelta(milliseconds=1)


def compute_time_normalized_pupil_change(lagged_df: DataFrame, pupil_col_name: str):
    """
    Computes the time-normalized pupil change for a given DataFrame and pupil column name.

    Parameters:
    lagged_df (DataFrame): The DataFrame containing the lagged pupil data.
    pupil_col_name (str): The name of the pupil column to compute the change for. Must be either "PupilLeft" or "PupilRight".

    Returns:
    DataFrame: A DataFrame containing the time-normalized pupil change values and corresponding timestamps.

    See also:
    https://doi.org/10.3758/s13428-018-1075-y for details on the computation of time-normalized pupil change.

    """
    assert pupil_col_name in ["PupilLeft", "PupilRight"]

    # Compute the dilation change for each time point
    pre_dilation_change = abs((lagged_df[pupil_col_name] - lagged_df[f"pre_{pupil_col_name}"]) / milliseconds_diff(
        lagged_df["LocalTimeStamp"], lagged_df["pre_LocalTimeStamp"]))
    post_dilation_change = abs((lagged_df[f"post_{pupil_col_name}"] - lagged_df[pupil_col_name]) /
                               milliseconds_diff(lagged_df["post_LocalTimeStamp"], lagged_df["LocalTimeStamp"]))
    max_dilation_change = np.maximum(pre_dilation_change, post_dilation_change)
    pupil_change_df = pd.DataFrame({f'{pupil_col_name}_change': max_dilation_change.values,
                                   'LocalTimeStamp': lagged_df["LocalTimeStamp"].values}, index=range(len(lagged_df)))

    return pupil_change_df


def add_pupil_change_to_df(df: DataFrame, ):
    """
    Adds the time-normalized pupil change to the lagged_df DataFrame.
    """
    lagged_df = create_lagged_df(df)
    pupil_change_left = compute_time_normalized_pupil_change(
        lagged_df, "PupilLeft")
    pupil_change_right = compute_time_normalized_pupil_change(
        lagged_df, "PupilRight")
    df = (
        df
        .merge(pupil_change_left, on="LocalTimeStamp", how="left")
        .merge(pupil_change_right, on="LocalTimeStamp", how="left")
    )
    return df


def compute_threshold_for_pupil_change_based_on_MAD(df: pd.DataFrame, k: Union[float, int]) -> pd.Series:
    """
    Compute the threshold for pupil change based on Median Absolute Deviation (MAD).

    Parameters:
    - df (pd.DataFrame): The input dataframe containing pupil dilation change columns.
    - k (Union[float, int]): The scaling factor for MAD.

    Returns:
    - pd.Series: The threshold for pupil change.

    Raises:
    - AssertionError: If the pupil dilation change columns are not found in the dataframe.

    See also:
    https://doi.org/10.3758/s13428-018-1075-y for details
    """

    pupil_change_cols = ["PupilLeft_change", "PupilRight_change"]
    assert all(
        col in df.columns for col in pupil_change_cols), "Pupil dilation change columns not found in dataframe"

    median_pupil_change = df[pupil_change_cols].median()
    # Median absolute deviation
    mad = abs(
        df[pupil_change_cols] - median_pupil_change
    ).median()
    # Only need to compute upper threshold since Pupil dilation change is compute using absolute values (i.e., reflecting pupil speed)
    threshold = median_pupil_change + (k * mad)
    return threshold


def add_loess_smoother_per_trial(segmented_df: pd.DataFrame, fraction: float = 1/5) -> pd.DataFrame:
    """
    Applies a loess smoother separately to the 'PupilLeft' and 'PupilRight' columns of the given DataFrame
    for each unique trial number specified in the 'trial_nr' column.

    Args:
        segmented_df (pd.DataFrame): The DataFrame containing the 'LocalTimeStamp', 'PupilLeft', 'PupilRight',
                                     and 'trial_nr' columns.
        fraction (float, optional): The fraction of data points to use for smoothing within each trial.
                                     Defaults to 1/200.

    Returns:
        pd.DataFrame: The DataFrame with additional columns 'PupilLeft_smoothed' and 'PupilRight_smoothed'
                      containing the smoothed values for each trial.
    """

    # Initialize columns for smoothed values
    segmented_df['PupilLeft_smoothed'] = pd.Series(dtype='float')
    segmented_df['PupilRight_smoothed'] = pd.Series(dtype='float')

    # Convert LocalTimeStamp to numeric once to avoid doing it inside the loop
    segmented_df['LocalTimeStampNumeric'] = pd.to_numeric(
        segmented_df['LocalTimeStamp'])

    # Apply LOWESS smoothing separately for each trial_nr
    for trial_nr in segmented_df['trial_nr'].unique():
        trial_data = segmented_df[segmented_df['trial_nr'] == trial_nr]

        LocalTimeStampNumeric = trial_data['LocalTimeStampNumeric']

        PupilLeft_smoothed = lowess(
            trial_data['PupilLeft'], LocalTimeStampNumeric, frac=fraction, return_sorted=False)
        PupilRight_smoothed = lowess(
            trial_data['PupilRight'], LocalTimeStampNumeric, frac=fraction, return_sorted=False)

        segmented_df.loc[segmented_df['trial_nr'] == trial_nr,
                         'PupilLeft_smoothed'] = PupilLeft_smoothed
        segmented_df.loc[segmented_df['trial_nr'] == trial_nr,
                         'PupilRight_smoothed'] = PupilRight_smoothed

    segmented_df.drop(columns=['LocalTimeStampNumeric'], inplace=True)

    return segmented_df


def _add_loess_smoother(df: pd.DataFrame, fraction: float = 1/200) -> pd.DataFrame:
    """
    Applies a loess smoother to the 'PupilLeft' and 'PupilRight' columns of the given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the 'PupilLeft' and 'PupilRight' columns.
        fraction (float, optional): The fraction of data points to use for smoothing. Defaults to 1/200.

    Returns:
        pd.DataFrame: The DataFrame with additional columns 'PupilLeft_smoothed' and 'PupilRight_smoothed'
                      containing the smoothed values.
    """

    LocalTimeStampNumeric = pd.to_numeric(df['LocalTimeStamp'])
    PupilLeft_smoothed = lowess(
        df['PupilLeft'], LocalTimeStampNumeric, frac=fraction, xvals=LocalTimeStampNumeric)
    PupilRight_smoothed = lowess(
        df['PupilRight'], LocalTimeStampNumeric, frac=fraction, xvals=LocalTimeStampNumeric)

    df = (
        df
        .assign(PupilLeft_smoothed=PupilLeft_smoothed)
        .assign(PupilRight_smoothed=PupilRight_smoothed)
    )

    return df


def add_pupil_distance_to_trendline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns to the given DataFrame containing the distance of the 'PupilLeft' and 'PupilRight' columns
    to the smoothed trendlines.

    Args:
        df (pd.DataFrame): The DataFrame containing the 'PupilLeft', 'PupilRight', 'PupilLeft_smoothed' and 'PupilRight_smoothed' columns.

    Returns:
        pd.DataFrame: The DataFrame with additional columns 'PupilLeft_dist' and 'PupilRight_dist' containing the distances.
    """
    df = (
        df
        .assign(PupilLeft_dist=abs(df["PupilLeft"] - df["PupilLeft_smoothed"]))
        .assign(PupilRight_dist=abs(df["PupilRight"] - df["PupilRight_smoothed"]))
    )

    return df


def compute_threshold_for_pupil_distance_to_trendline_based_on_MAD(df: pd.DataFrame, k: Union[float, int] = 3) -> pd.Series:
    """
    Computes a threshold for the distance of the 'PupilLeft' and 'PupilRight' columns to the smoothed trendlines
    based on the median absolute deviation (MAD) of the distances.

    Returns:
        pd.Series: The threshold for the 'PupilLeft_dist' and 'PupilRight_dist' columns.
    """
    median_distance = df[["PupilLeft_dist", "PupilRight_dist"]].median()
    mad_median_distance = (
        df[["PupilLeft_dist", "PupilRight_dist"]] - median_distance).abs().median()
    threshold = median_distance + (k * mad_median_distance)
    return threshold


def remove_dilation_speed_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes outliers in the dilation speed of the left and right pupils.

    This function takes a DataFrame containing pupil dilation data and removes outliers in the dilation speed
    of the left and right pupils. Outliers are identified based on the presence of the 'PupilLeft_dilation_speed_outlier'
    and 'PupilRight_dilation_speed_outlier' columns in the DataFrame. If an outlier is detected, the corresponding
    pupil dilation value is set to NaN.

    Args:
        df (pd.DataFrame): The input DataFrame containing the pupil dilation data.

    Returns:
        pd.DataFrame: The DataFrame with outliers in the dilation speed removed.
    Note:
        No records (rows) are removed from the DataFrame.
    """
    df.loc[df.PupilLeft_dilation_speed_outlier == True, "PupilLeft"] = np.nan
    df.loc[df.PupilRight_dilation_speed_outlier == True, "PupilRight"] = np.nan
    return df


def remove_pupil_dilation_near_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Masks pupil dilation data as NaN for records near gaps in the dataset.

    This function identifies records in the DataFrame where samples are near gaps (indicated by
    the boolean column 'sample_near_gap' being True) and sets the pupil dilation measurements
    ('PupilLeft' and 'PupilRight') for these records to NaN. This is typically used to prepare
    data for analysis by excluding potentially unreliable measurements adjacent to missing data.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'PupilLeft' and 'PupilRight' columns with pupil 
      dilation measurements, and a 'sample_near_gap' boolean column indicating proximity to gaps.

    Returns:
    - pd.DataFrame: The modified DataFrame with 'PupilLeft' and 'PupilRight' set to NaN for records 
      flagged as near gaps.

    Example:
    ```
    df = pd.DataFrame({
        'PupilLeft': [3, 4, 5, 4],
        'PupilRight': [4, 5, 6, 4],
        'sample_near_gap': [False, True, False, True]
    })
    cleaned_df = remove_pupil_dilation_near_gaps(df)
    ```
    """
    df.loc[df.sample_near_gap == True, ["PupilLeft", "PupilRight"]] = np.nan
    return df


def remove_trend_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces outlier pupil dilation measurements with NaN based on trend analysis.

    This function examines the DataFrame for outlier flags in 'PupilLeft_trend_outlier' and
    'PupilRight_trend_outlier' columns. If an outlier is flagged as True for either the left or
    right pupil dilation measurements ('PupilLeft' or 'PupilRight'), those measurements are set
    to NaN. It is useful for cleaning the data by removing measurements that deviate significantly
    from the trend, potentially due to errors or extreme variations.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'PupilLeft', 'PupilRight', 'PupilLeft_trend_outlier',
      and 'PupilRight_trend_outlier'. The outlier columns contain boolean flags indicating whether
      the corresponding measurement is considered an outlier in the trend.

    Returns:
    - pd.DataFrame: The modified DataFrame with 'PupilLeft' and 'PupilRight' measurements set to NaN
      for records flagged as trend outliers.

    Example:
    ```
    df = pd.DataFrame({
        'PupilLeft': [3, 4, 100, 4],
        'PupilRight': [4, 200, 6, 4],
        'PupilLeft_trend_outlier': [False, False, True, False],
        'PupilRight_trend_outlier': [False, True, False, False]
    })
    cleaned_df = remove_trend_outliers(df)
    ```
    """
    df.loc[df.PupilLeft_trend_outlier ==
           True, "PupilLeft"] = np.nan
    df.loc[df.PupilRight_trend_outlier ==
           True, "PupilRight"] = np.nan
    return df


##################################################
############ Aggregation  functions ##############
##################################################

def prepare_dataset(df: pd.DataFrame, split_time_ms: int = 1250) -> pd.DataFrame:
    """
    Filters and transforms a dataset based on specific emotion and time window criteria.

    This function processes the input DataFrame by performing the following operations:
    1. Removes rows where the Emotion column is labeled as "neutral".
    2. Keeps rows where the segment_type column is "stimulus".
    3. Adds a new column, "time_window", categorizing each row based on the "RelativeTime_ms" column
       into "early" if the time is less than `split_time_ms`, or "late" otherwise.
    4. Filters the DataFrame to keep only the rows categorized as "late" in the "time_window".

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the dataset to be processed.
    - split_time_ms (int, optional): The time in milliseconds used to split the "RelativeTime_ms"
      into "early" or "late" categories. Defaults to 800 ms.

    Returns:
    - pd.DataFrame: The processed DataFrame after applying the filters and transformations.
    """
    df = df[df.Emotion != "neutral"]
    df = df[df.segment_type == "stimulus"]
    df.loc[:, "time_window"] = df["RelativeTime_ms"].apply(
        lambda t: "early" if t < split_time_ms else "late"
    )
    df = df[df.time_window == "late"]
    return df


def aggregate_single_subject_data(df: pd.DataFrame, index_cols) -> pd.DataFrame:
    """
    Aggregates pupil size data for a single subject by computing the mean of baseline-corrected pupil sizes.

    This function takes a DataFrame containing pupil size data, groups the data by specified index columns,
    and calculates the mean of the baseline-corrected pupil sizes within each group. The result is a new
    DataFrame with the mean pupil sizes alongside the index columns used for grouping.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing pupil size data with a column named
      "pupilsize_baseline_corrected" for baseline-corrected pupil sizes.
    - index_cols: A list of column names or a single column name to be used for grouping the data.
      The columns specified by index_cols are used to identify unique subjects or conditions.

    Returns:
    - pd.DataFrame: A DataFrame containing the mean baseline-corrected pupil sizes for each group defined
      by the index_cols, with the group identifiers (index_cols) and the aggregated means.
    """
    agg_df = (
        df.groupby(index_cols)["pupilsize_baseline_corrected"]
        .aggregate(np.mean)
        .reset_index()
    )

    return agg_df


##################################################
############## Luminance functions ###############
##################################################


def verify_image_is_rgb(image: Image.Image) -> None:
    """
    Verifies if the given image is in RGB mode.

    Parameters:
    - image (Image.Image): The image to check.

    Raises:
    - ValueError: If the image is not in RGB mode.

    Returns:
    - None
    """
    if image.mode != "RGB":
        raise ValueError("Image must be in RGB mode")


def get_RGB_channel_arrays(image: Image.Image) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts and returns the RGB channel arrays from a given RGB image.

    This function first verifies that the given image is in RGB mode. It then splits the image into its
    red, green, and blue components, and converts each component into a numpy array.

    Parameters:
    - image (Image.Image): The image to extract RGB channels from.

    Returns:
    - Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing numpy arrays of the red, green,
      and blue channels of the image, in that order.

    Raises:
    - ValueError: If the image is not in RGB mode.
    """
    verify_image_is_rgb(image)
    r, g, b = image.split()
    return np.array(r), np.array(g), np.array(b)


def gamma_encode_rgb(*channels: float) -> List[float]:
    """
    Performs gamma encoding on the given RGB channel values.

    This function takes one or more channel values (assumed to be in the range 0-255 for each
    of the RGB channels) and applies gamma encoding by dividing each value by 255. This is a 
    common step in processing RGB values for various image processing tasks.

    Parameters:
    - channels (float): Variable number of arguments, each representing a channel value in the range 0-255.

    Returns:
    - List[float]: A list of gamma-encoded channel values, scaled to the range 0-1.

    Note:
    - This function does not explicitly perform gamma correction (i.e., it does not apply a non-linear transformation
      based on a gamma value). Instead, it scales the input values to the range 0-1, which is a prerequisite for
      many gamma correction algorithms.
    """
    return [channel / 255.0 for channel in channels]


def linearize_rgb_channels(*channels: np.ndarray) -> List[np.ndarray]:
    """
    Applies linearization to RGB channels to correct for gamma compression.

    This function is based on a common formula for adjusting RGB values to reflect their perceived brightness,
    taking into account the non-linear relationship between the encoded luminance in an image and the actual
    brightness. This process is a form of gamma correction, specifically reversing the gamma compression
    applied to images for storage or transmission.

    Parameters:
    - channels (np.ndarray): Variable number of numpy arrays, each representing an RGB channel's data.

    Returns:
    - List[np.ndarray]: A list of numpy arrays, each containing the linearized data of the corresponding input channel.

    Note:
    The formula used for linearization is based on the assumption that the input values are scaled to the range 0-1.
    It applies a piecewise function where values below 0.04045 are adjusted differently from those above, reflecting
    the standard approach to gamma correction in sRGB images.

    Source:
    The formula for linearizing RGB channels is adapted from a Stack Overflow discussion on determining the perceived
    brightness of an RGB color (https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color).
    """
    def func(chn: np.ndarray) -> np.ndarray:
        return np.where(chn <= 0.04045, chn / 12.92, ((chn + 0.055) / 1.055) ** 2.4)

    return [func(channel) for channel in channels]


def compute_luminance(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Computes the luminance of an image based on its red, green, and blue channel data.

    This function calculates the luminance by applying standard luminance coefficients to
    the RGB channels of an image. The coefficients used are 0.2126 for the red channel, 0.7152
    for the green channel, and 0.0722 for the blue channel, reflecting their relative contributions
    to human-perceived brightness.

    Parameters:
    - r (np.ndarray): The numpy array representing the red channel of the image.
    - g (np.ndarray): The numpy array representing the green channel of the image.
    - b (np.ndarray): The numpy array representing the blue channel of the image.

    Returns:
    - np.ndarray: A numpy array representing the luminance of the image.

    Note:
    The returned luminance values are weighted averages of the RGB channels, scaled according to
    the standard luminance coefficients. This method is widely used in image processing to obtain
    a grayscale representation of an image or to calculate perceived brightness.
    """
    # Luminance coefficients for RGB
    luminance_coefficients = np.array([0.2126, 0.7152, 0.0722])
    luminance = r * luminance_coefficients[0] + g * \
        luminance_coefficients[1] + b * luminance_coefficients[2]

    return luminance


##################################################
############## Plotting  functions ###############
##################################################


def pivot_stimulus_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the stimulus data to have the trial_nr as the index and the RelativeTime_ms as the columns
    """
    stimulus_data = df[df.segment_type == "stimulus"]
    return stimulus_data.pivot(
        index=["trial_nr", "%samples_stimulus",
               "%samples_fixation", "Emotion"],
        columns="RelativeTime_ms",
        values="pupilsize_baseline_corrected",
    )


def round_to_nearest_16ms(ms: Union[int, float]) -> int:
    """
    Rounds the given time in milliseconds to the nearest multiple of 16 milliseconds.

    Parameters:
    ms (float, int): The time in milliseconds to be rounded.

    Returns:
    int: The rounded time in milliseconds.
    """
    return int(round(ms / 50) * 50)
