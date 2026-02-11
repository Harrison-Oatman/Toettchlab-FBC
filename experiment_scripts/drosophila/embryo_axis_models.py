import numpy as np
import tifffile
import matplotlib.pyplot as plt

from pyclm import SegmentationMethod, PatternMethod
from pyclm.core.patterns.pattern import PatternContext
from pyclm.core.segmentation.cellpose_segmentation import CellposeSegmentationMethod
from pyclm import run_pyclm

from skimage.transform import resize
from skimage.measure import regionprops, label
from pathlib import Path

BASE_PATH = Path(r"E:\Harrison\fly_revision_experiments\20260129b_inner_hq")
TEMP_FOLDER = BASE_PATH / "TEMP"
TEMP_FOLDER.mkdir(exist_ok=True)

def get_temp_fp(experiment_name):

    temp_name = "_".join(experiment_name.split(".")[1:])

    return TEMP_FOLDER / f"{temp_name}_temp_mask.tif"

class EmbryoSegmentationMethod(CellposeSegmentationMethod):

    def __init__(self, experiment_name, model="embryomodel", use_gpu=True, normlow=0, normhigh=0.15, ideal_size=(100, 100), **kwargs):
        super().__init__(experiment_name, model, use_gpu, normlow, normhigh)

        self.ideal_size = ideal_size

    def segment(self, data):

        downscaled_frame = resize(data, self.ideal_size)
        tifffile.imwrite(TEMP_FOLDER / f"{self.experiment_name}_input.tif", downscaled_frame)
        print(f"data: {downscaled_frame.min()}, {downscaled_frame.max()}, {downscaled_frame.mean()}")

        out = super().segment(downscaled_frame)
        mask = out > 0

        tifffile.imwrite(TEMP_FOLDER / f"{self.experiment_name}_output.tif", mask)
        big_mask = resize(mask.astype(float), data.shape, order=1) > 0.5

        temp_fp = get_temp_fp(self.experiment_name)
        tifffile.imwrite(temp_fp, big_mask)

        return big_mask


class EmbryoOutlineSaver(PatternMethod):

    name = "outline"

    def __init__(self, embryo_channel="638", **kwargs):
        super().__init__(**kwargs)

        self.embryo_channel = embryo_channel
        self.seg_channel_id = None

        self.add_requirement(self.embryo_channel, False, True)

    def generate(self, context: PatternContext) -> np.ndarray:

        xx, _ = self.get_meshgrid()

        return xx*0

class PatternAlongAxis(PatternMethod):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.mask = None
        self.centroid = None
        self.long_axis = None
        self.axis_length = None

    def check_for_mask(self) -> bool:

        temp_fp = get_temp_fp(self.experiment_name)

        print(f"looking for mask at {temp_fp}")

        if not temp_fp.exists():
            return False

        print(f"found mask at {temp_fp}")

        mask = tifffile.imread(temp_fp)
        self.mask = mask

        labeled_mask = label(mask)
        props = regionprops(labeled_mask)

        if len(props) == 0:
            return False

        biggest_prop_area = 0
        for prop in props:
            if prop.area > biggest_prop_area:
                self.mask = labeled_mask == prop.label
                self.centroid = prop.centroid
                self.long_axis = (np.sin(prop.orientation), np.cos(prop.orientation))
                self.axis_length = prop.axis_major_length

                biggest_prop_area = prop.area

        return True

    def apply_magnitude(self, mag) -> np.ndarray:
        raise NotImplementedError

    def generate(self, context: PatternContext):
        if not self.check_for_mask():
            return 0 * self.get_meshgrid()[0]

        y_arange = np.arange(self.pattern_shape[0])
        x_arange = np.arange(self.pattern_shape[1])

        yy, xx = np.meshgrid(y_arange, x_arange)

        mag = (yy - self.centroid[1]) * self.long_axis[0] + (xx - self.centroid[0]) * self.long_axis[1]

        # plt.imshow(mag)
        # plt.show()

        mag = np.abs(mag) / (self.axis_length / 2)

        included = self.apply_magnitude(mag)

        # plt.imshow(included * self.mask)
        # plt.show()

        return included * self.mask

class InnerPatternMethod(PatternAlongAxis):

    def __init__(self, fraction_length=0.1, **kwargs):
        super().__init__(**kwargs)

        self.fraction_length = fraction_length

    def apply_magnitude(self, mag) -> np.ndarray:

        return mag < self.fraction_length


class OuterPatternMethod(PatternAlongAxis):

    def __init__(self, fraction_length=0.1, **kwargs):
        super().__init__(**kwargs)

        self.fraction_length = fraction_length

    def apply_magnitude(self, mag) -> np.ndarray:

        return mag > (1 - self.fraction_length)

segmentation_methods = {
    "embryo": EmbryoSegmentationMethod
}

pattern_methods = {
    "outline": EmbryoOutlineSaver,
    "inner": InnerPatternMethod,
    "outer": OuterPatternMethod,
}

if __name__ == "__main__":

    experiment_directory = BASE_PATH

    run_pyclm(experiment_directory, segmentation_methods=segmentation_methods, pattern_methods=pattern_methods)



