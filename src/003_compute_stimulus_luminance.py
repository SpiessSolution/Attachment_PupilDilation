# # Compute Stimulus luminance
#
# Purpose of this script is to load all stimuli (from the radboud faces database) and compute the average luminance (+sd) of each stimulus.
# - The approach is derived from: https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color
#
# ### Input <> Output
# - Input are the stimuli shown during the eye tracking experiment. The stimuli are located in `~/Data/stimuli/`
# - Output of this notebook is a dataset with the average (+ standard deviation) luminance  per stimulus. Data is stored in `~/Data/processed/`.
#


# fmt: off
from PIL import Image
from pathlib import Path
import sys
sys.path.append(str(Path().cwd().parent/ "src"))
import modules.utils as utils
import numpy as np
import pandas as pd
# fmt: on


if __name__ == '__main__':
    #################
    ###### I/O ######
    #################
    print("Setting up I/O")

    CURRENT_FILEPATH = Path(__file__)
    DATA_DIR = CURRENT_FILEPATH.parent.parent / "data"
    STIMULI_DIR = DATA_DIR / "stimuli"
    OUTPUT_DIR = DATA_DIR / "processed"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ##################
    ###### main ######
    ##################
    images_path_iter = STIMULI_DIR.glob("*.jpg")
    # 1. Gamma-correct RGB values (i.e., normalize to [0,1] range)
    # 2. Linearize RGB values
    # 3. Compute pixel-wise luminance
    # 4. Aggregate: mean luminance + standard deviation

    data_dict = {"filename": [], "mean_luminance": [], "std_luminance": []}
    for filepath in images_path_iter:
        filename = filepath.parts[0 - 1]
        image = Image.open(filepath)

        r, g, b = utils.get_RGB_channel_arrays(image)
        vR, vG, vB = utils.gamma_encode_rgb(r, g, b)
        vlR, vlG, vlB = utils.linearize_rgb_channels(vR, vG, vB)
        luminance = utils.compute_luminance(vlR, vlG, vlB)

        data_dict["filename"].append(filename)
        data_dict["mean_luminance"].append(np.mean(luminance))
        data_dict["std_luminance"].append(np.std(luminance))

    # Put into single dataframe
    luminance_df = pd.DataFrame(data_dict)

    # Save to excel
    luminance_df.to_excel(
        OUTPUT_DIR / "stimulus_luminance_values.xlsx", index=False)
    print("Done")
