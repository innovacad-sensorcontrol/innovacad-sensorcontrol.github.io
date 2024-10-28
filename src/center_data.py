"""
Script to center impulse kind of signal inside the window and segment data by such windows.

Data requirements:
    - Data should be in CSV format.
    - Each activity csv files must be in a separate folder named <activity_name> in the raw_data folder (E.g. 'raw_data/swipe_right').
    - Each csv file must be named as p<user_id>_<activity_name>_<session_number>.csv (E.g. 'p1_swipe_right_1.csv').

Constant variables to modify:
TOTAL_WINDOW_SIZE
    Number of samples in the window. If data frequency is 100Hz and activity length is within 1 second, then TOTAL_WINDOW_SIZE = 100

WORK_WINDOW_SIZE
    Usually 95% of the TOTAL_WINDOW_SIZE

WORK_AXIS
    Axis that reflects changes in the signal. Usually, it is the axis with the highest amplitude.

CONFIG
    Dictionary with the actual class names (activities) and their properties:
        - class_code: integer value that represents the class
        - percentage_of_data_to_use: percentage of data to use from the raw data (0-100)
        - signal_nature: type of the signal (continuous or impulse) if impulse, then the signal will be centered and segmented, otherwise, it will be used as is.
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path


RAW_DATA_PATH = Path("../data/raw_data")
OUTPUT_DATA_PATH = RAW_DATA_PATH.parent / "processed_data"

SEARCH_STEP = 1  # how many samples to skip when searching for the next segment
WORK_WINDOW_SIZE = 85  # window size for impulse signals
TOTAL_WINDOW_SIZE = 90  # total window size for impulse signals
THRESHOLD = 0.5  # threshold for zero-crossing detection
WORK_AXIS = "acc_x"  # use acc_x to center signal


CONFIG = {
    "idle": {
        "class_code": 0,
        "percentage_of_data_to_use": 25,
        "signal_nature": "continuous",  # continuous signal does not require centering or segmentation
    },
    "unknown": {
        "class_code": 1,
        "percentage_of_data_to_use": 100,
        "signal_nature": "continuous",
    },
    "swipe_right": {
        "class_code": 2,
        "percentage_of_data_to_use": 100,
        "signal_nature": "impulse",  # impulse signal requires centering and segmentation
    },
    "swipe_left": {
        "class_code": 3,
        "percentage_of_data_to_use": 100,
        "signal_nature": "impulse",
    },
    "double_thumb_tap": {
        "class_code": 4,
        "percentage_of_data_to_use": 100,
        "signal_nature": "impulse",
    },
    "double_knock": {
        "class_code": 5,
        "percentage_of_data_to_use": 100,
        "signal_nature": "impulse",
    },
    "clockwise_rotation": {
        "class_code": 6,
        "percentage_of_data_to_use": 100,
        "signal_nature": "continuous",
    },
    "counterclockwise_rotation": {
        "class_code": 7,
        "percentage_of_data_to_use": 100,
        "signal_nature": "continuous",
    },
}


def clip_start_end_of_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(df.index[0:100]).drop(df.index[-100:])
    df.reset_index(drop=True, inplace=True)
    return df


def offset_signal_to_zero(series: pd.Series) -> np.array:
    """Removal of direct current component from the signal by subtracting the mean value
    of the signal from the signal itself."""
    return (series - series.mean()).values


def amplify_activity_in_windows(data: np.array) -> np.array:
    global SEARCH_STEP, WORK_WINDOW_SIZE

    prepared_data = []
    for i in range(0, len(data) - WORK_WINDOW_SIZE, SEARCH_STEP):
        data_window = data[i : i + WORK_WINDOW_SIZE]
        window_min = np.min(data_window)
        window_max = np.max(data_window)

        prepared_data.append(np.sqrt(np.square(window_min) + np.square(window_max)))
    return prepared_data


def find_segment_start_end(data: np.array):
    global WORK_WINDOW_SIZE, THRESHOLD
    # Calculate the mean of the data
    mean_data = np.mean(data)
    # Set the threshold for zero-crossing detection
    threshold = THRESHOLD * mean_data
    i = 0
    segment_started = False
    start_index = 0
    end_index = 0
    segments = []
    for d in data:
        i += 1
        if d > threshold:
            if not segment_started:
                segment_started = True
                start_index = i
        else:
            if segment_started:
                segment_started = False
                end_index = i
                segments.append(
                    (
                        start_index + WORK_WINDOW_SIZE / 2,
                        end_index + WORK_WINDOW_SIZE / 2,
                    )
                )
    return segments


def center_signal(df: pd.DataFrame, segments_start_end: list) -> pd.DataFrame:
    global TOTAL_WINDOW_SIZE

    index_labled = 0
    columns = df.columns.tolist()
    df_labled = pd.DataFrame(columns=columns)
    for segment in segments_start_end:
        start_index, end_index = segment
        center_index = int((end_index + start_index) / 2)
        for i in range(
            center_index - int(TOTAL_WINDOW_SIZE / 2),
            center_index + int(TOTAL_WINDOW_SIZE / 2),
        ):
            row_data = df.iloc[i]
            df_labled.loc[index_labled] = row_data
            index_labled += 1
    return df_labled


def main():
    global CONFIG, OUTPUT_DATA_PATH

    for class_name in tqdm(CONFIG):
        raw_files_dir = RAW_DATA_PATH / class_name
        for raw_file_path in raw_files_dir.iterdir():

            df = pd.read_csv(raw_file_path)
            df = df.dropna()
            df = clip_start_end_of_df(df)

            session = int(raw_file_path.stem.split("_")[-1])
            user = int(raw_file_path.stem.split("_")[0].replace("p", ""))
            class_code = CONFIG[class_name]["class_code"]

            # limit data by percentage in CONFIG
            data_percentage = CONFIG[class_name]["percentage_of_data_to_use"]
            row_count = int(len(df) * data_percentage / 100)
            df = df.loc[0:row_count]

            if CONFIG[class_name]["signal_nature"] == "impulse":  # needs segmentation
                reference_data = offset_signal_to_zero(df[WORK_AXIS])
                prepered_data = amplify_activity_in_windows(reference_data)
                segments_start_end = find_segment_start_end(prepered_data)
                df = center_signal(df, segments_start_end)

            df["target"] = class_code
            df["user"] = user
            df["session"] = session

            output_file_dir = OUTPUT_DATA_PATH / class_name
            if not os.path.exists(output_file_dir):
                os.makedirs(output_file_dir)
            df.to_csv(output_file_dir / raw_file_path.name, index=False)


if __name__ == "__main__":
    main()
