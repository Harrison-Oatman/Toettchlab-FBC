from pyclm import run_pyclm, PatternMethod, SegmentationMethod, PFSPositionMover
import numpy as np
from pyclm.core.experiments import Experiment
from pyclm.core.patterns import AcquiredImageRequest, DataDock, PatternContext
from pyclm.core.segmentation import CellposeSegmentationMethod
from pathlib import Path
from scipy.ndimage import generic_filter
from skimage.measure import regionprops_table, regionprops
from laptrack import LapTrack
import pandas as pd


class TrackingCellposeSegmentationMethod(CellposeSegmentationMethod):

    def __init__(self, experiment_name, max_distance=20, **kwargs):
        super().__init__(experiment_name, **kwargs)

        self.tracks = pd.DataFrame(columns=["label", "frame", "x", "y", "stable_track_id"])
        self.tracker = LapTrack(
            metric="sqeuclidean",
            # The similarity metric for particles. See `scipy.spatial.distance.cdist` for allowed values.
            splitting_metric="sqeuclidean",
            merging_metric="sqeuclidean",
            gap_closing_metric="sqeuclidean",
            # the square of the cutoff distance for the "sqeuclidean" metric
            cutoff=max_distance**2,
            splitting_cutoff=False,  # or False for non-splitting case
            merging_cutoff=False,  # or False for non-merging case
            gap_closing_cutoff=max_distance**2,
            gap_closing_max_frame_count=2,
        )

        self.current_frame = 0

    def add_tracks(self, tracks):
        """
        Adds tracks to the main dataframe
        track_ids with value -1 will be assigned new track_ids
        """
        max_track_id = self.tracks["stable_track_id"].max() if not self.tracks.empty else -1
        new_tracks = tracks.copy()
        needs_track_id = new_tracks["stable_track_id"] == -1
        new_tracks.loc[needs_track_id, "stable_track_id"] = np.arange(max_track_id + 1, max_track_id + 1 + needs_track_id.sum())

        self.tracks = pd.concat([self.tracks, new_tracks], ignore_index=True)

    def remap_labels(self, mask):
        label_map = self.tracks.query("frame == @self.current_frame")[["label", "stable_track_id"]].set_index("label")["stable_track_id"].to_dict()

        mapper = np.arange(mask.max() + 1)
        for label, track_id in label_map.items():
            mapper[label] = track_id

        return mapper[mask]

    def segment(self, data):
        mask = super().segment(data)

        positions = regionprops_table(mask, properties=["label", "centroid"])
        positions_df = pd.DataFrame(positions)
        positions_df = positions_df.rename(columns={"centroid-0": "y", "centroid-1": "x"})
        positions_df["frame"] = self.current_frame

        prev_tracks = self.tracks.query("frame >= @self.current_frame - 2")[["label", "x", "y", "frame", "stable_track_id"]].copy()

        tracking_df = pd.concat([prev_tracks, positions_df], ignore_index=True)

        tracks, _, _ = self.tracker.predict_dataframe(tracking_df,
                                                      coordinate_cols=["x", "y"],
                                                      frame_col="frame"
                                                      )

        t = tracks.groupby("track_id").agg(
            count=("label", "count"),
            label=("label", "last"),
            x=("x", "last"),
            y=("y", "last"),
            frame=("frame", "last"),
            stable_track_id=("stable_track_id", "first"),
        )

        t = t.query("count > 1 and frame == @self.current_frame")

        label_track_id_map = t.set_index("label")["stable_track_id"].to_dict()
        positions_df["stable_track_id"] = positions_df["label"].map(label_track_id_map).fillna(-1).astype(int)

        self.add_tracks(positions_df[["label", "frame", "x", "y", "stable_track_id"]])

        remapped = self.remap_labels(mask)

        self.current_frame += 1

        return remapped


class DynamicRangeControl(PatternMethod):

    name = "dynamicrangecontrol"

    def __init__(self, seg_channel=545, raw_channel=638, warm_up_mins=30, active_mins=60,
                 intensity_regimen=None, min_dynamic_range=0.5, **kwargs):

        super().__init__(**kwargs)

        self.seg_channel = str(seg_channel)
        self.raw_channel = str(raw_channel)

        self.warm_up_mins = warm_up_mins
        self.active_mins = active_mins
        self.intensity_regimen = intensity_regimen

        if self.intensity_regimen is None:
            self.intensity_regimen = [0.5]

        self.min_dynamic_range = min_dynamic_range

        self.add_requirement(self.seg_channel, False, True)
        self.add_requirement(self.seg_channel, False, True)

        self._min_vals = None
        self._max_vals = None

        self.t = 0

        self.df = pd.DataFrame(columns=["label", "time", "frame", "x", "y", "raw_mean", "target_value", "light_delivered"])

    def get_min_max_vals(self):
        if self._min_vals is not None and self._max_vals is not None:
            return self._min_vals, self._max_vals

        min_vals = self.df.groupby("label")["raw_mean"].min()
        max_vals = self.df.groupby("label")["raw_mean"].max()

        good_labels = np.abs(np.log2(max_vals / min_vals)) > self.min_dynamic_range

        print(f"{self.experiment_name} -- good labels: {good_labels.sum()} / {len(good_labels)}")

        self._min_vals = min_vals[good_labels]
        self._max_vals = max_vals[good_labels]

        return self._min_vals, self._max_vals

    @staticmethod
    def fbc_step(self, prop, target_intensity):
        if prop.intensity_mean < target_intensity:
            return prop.image * 0

        return prop.image

    def process_prop(self, prop, target_value):
        min_vals, max_vals = self.get_min_max_vals()

        if prop.label not in min_vals.index:
            return prop.image*0

        target = min_vals[prop.label] + target_value * (max_vals[prop.label] - min_vals[prop.label])

        return self.fbc_step(prop, target)

    def generate_image(self, seg, raw, target_value):
        h, w = self.pattern_shape

        if target_value == 0:
            return np.zeros((int(h), int(w)), dtype=np.float16)
        if target_value == 1:
            return np.ones((int(h), int(w)), dtype=np.float16)

        new_img = np.zeros((int(h), int(w)), dtype=np.float16)

        for prop in regionprops(seg, intensity_image=raw):
            cell_stim = self.process_prop(prop, target_value)

            new_img[prop.bbox[0]: prop.bbox[2], prop.bbox[1]: prop.bbox[3]] += (
                cell_stim
            )

        return np.clip(new_img, 0, 1).astype(np.float16)

    def generate(self, context: PatternContext) -> np.ndarray:

        seg = context.segmentation(self.seg_channel)
        raw = context.raw(self.raw_channel)
        time_s = context.time

        props = regionprops_table(seg, raw, properties=["label", "mean_intensity", "centroid"])

        frame_df = pd.DataFrame(props)
        frame_df["time"] = time_s
        frame_df = frame_df.rename(columns={"mean_intensity": "raw_mean", "centroid-0": "y", "centroid-1": "x"})

        target_value = 0

        if self.warm_up_mins < time_s * 60 <= self.warm_up_mins * 2:
            target_value = 1

        elif time_s * 60 > self.warm_up_mins * 2:
            target_value = self.intensity_regimen[
                np.round((time_s*60 - self.warm_up_mins*2) // self.active_mins) % len(self.intensity_regimen)
            ]

        frame_df["target_value"] = target_value

        self.df = pd.concat([self.df, frame_df], ignore_index=True)

        return self.generate_image(seg, raw, target_value)


BASE_PATH = r""

def main():

    pattern_methods = {"fbc": DynamicRangeControl}
    segmentation_methods = {"tracking_cellpose": TrackingCellposeSegmentationMethod}

    run_pyclm(BASE_PATH, pattern_methods=pattern_methods, segmentation_methods=segmentation_methods, position_mover=PFSPositionMover(), gui=True)

if __name__ == "__main__":
    main()


