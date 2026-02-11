from pyclm import run_pyclm, PatternMethod, SegmentationMethod
import numpy as np
from pyclm.core.experiments import Experiment
from pyclm.core.patterns import AcquiredImageRequest, DataDock, PatternContext
from pathlib import Path
from scipy.ndimage import generic_filter


def image_conversion_helper(image_shape, grid_shape, diameter_ratio):
    """
    generates np arrays to convert between grid coordinates and image coordinates

    returns x_bin, y_bin, and valid (whether it is an active pixel and not a border)
    """
    h, w = image_shape
    rows, cols = grid_shape

    # pixel coordinates
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    bin_height = h / rows
    bin_width = w / cols

    x_bin = (xx // bin_width).astype(int)
    y_bin = (yy // bin_height).astype(int)

    # fractional position inside each bin
    x_border_value = (xx % bin_width) / bin_width
    y_border_value = (yy % bin_height) / bin_height

    edge_width = (1 - diameter_ratio) / 2

    is_border_x = (x_border_value < edge_width) | (x_border_value > (1 - edge_width))
    is_border_y = (y_border_value < edge_width) | (y_border_value > (1 - edge_width))
    is_border = is_border_x | is_border_y

    # mask to keep only interior pixels
    valid = ~is_border

    return x_bin, y_bin, valid

def image_to_grid(image, grid_size, diameter_ratio) -> np.ndarray:

    rows, cols = grid_size, grid_size

    x_bin, y_bin, valid = image_conversion_helper(image.shape, (rows, cols), diameter_ratio)

    flat_bins = y_bin * cols + x_bin
    flat_bins = flat_bins[valid]
    flat_image = image[valid]

    # accumulate sums and counts
    sums = np.bincount(flat_bins, weights=flat_image, minlength=rows * cols)
    counts = np.bincount(flat_bins, minlength=rows * cols)

    # avoid divide-by-zero
    grid = np.zeros(rows * cols)
    nonzero = counts > 0
    grid[nonzero] = sums[nonzero] / counts[nonzero]

    return grid.reshape(rows, cols)

def image_to_max_grid(image, grid_size, diameter_ratio):

    rows, cols = grid_size, grid_size

    x_bin, y_bin, valid = image_conversion_helper(image.shape, (rows, cols), diameter_ratio)

    def bin_max(flat_bins, flat_image, nbins):
        out = np.full(nbins, -np.inf)
        np.maximum.at(out, flat_bins, flat_image)
        return out

    flat_bins = y_bin * cols + x_bin
    flat_bins = flat_bins[valid]
    flat_image = image[valid]

    grid_max = bin_max(flat_bins, flat_image, rows * cols)
    grid_max = grid_max.reshape(rows, cols)

    flat_bins = y_bin * cols + x_bin
    flat_bins = flat_bins[~valid]
    flat_image = image[~valid]

    border_max = bin_max(flat_bins, flat_image, rows * cols)
    border_max = border_max.reshape(rows, cols)

    return grid_max, border_max

def grid_to_image(grid: np.ndarray, image_shape, diameter_ratio) -> np.ndarray:

    diameter_ratio = diameter_ratio

    x_bin, y_bin, valid = image_conversion_helper(image_shape, grid.shape, diameter_ratio)

    bin_value = grid[y_bin, x_bin]

    return valid * bin_value

class SegmentGameOfLife(SegmentationMethod):

    def __init__(self, experiment_name, absolute_threshold=10000, grid_size=50, diameter_ratio=0.7, **kwargs):
        super().__init__(experiment_name=experiment_name, **kwargs)

        self.threshold = absolute_threshold
        self.grid_size = grid_size
        self.diameter_ratio = diameter_ratio


    def detect_state_from_image(self, image: np.ndarray) -> np.ndarray:
        """
        'segmentation' just gets the average value of each bin, and calls it alive if it's above the threshold
        :param image:
        :return:
        """
        grid_avg_values = image_to_grid(image, self.grid_size, self.diameter_ratio)
        grid_max, border_max = image_to_max_grid(image, self.grid_size, self.diameter_ratio)

        neighborhood_on = grid_max > border_max
        neighborhood_on = neighborhood_on * (grid_avg_values > self.threshold)

        return neighborhood_on

    def segment(self, image):

        grid = self.detect_state_from_image(image)
        segmented_grid = grid_to_image(grid, image.shape, self.diameter_ratio)

        return segmented_grid

class GameOfLife(PatternMethod):

    name = "gameoflife"

    def __init__(self, grid_size: int = 50, diameter_ratio: float = 0.7, threshold: int = 4200, **kwargs):
        super().__init__(**kwargs)
        self.grid_size = grid_size
        self.diameter_ratio = diameter_ratio
        self.threshold = threshold

        self.grid = np.random.choice([0, 1], size=(grid_size, grid_size), replace=True)

        # this call to add_requirement is needed to give our stim channel a name
        self.request_stim(seg=True)

        self.turn_zero = True


    @staticmethod
    def advance_gol(grid: np.ndarray):
        """
        implements game of life rules:
        cells are either dead or alive
        a live cell with less than 2 neighbors dies by underpopulation
        a live cell with more than 3 neighbors dies by overpopulation
        a live cell with 2-3 neighbors stays alive
        a dead cell with exactly 3 neighbors is born

        takes current state and returns the next state
        """

        print(np.sum(grid))

        padded = np.pad(grid, (1, 1), mode="wrap")

        num_neighbors = np.zeros_like(grid).astype(int)

        for h in (0, 1, 2):
            for w in (0, 1, 2):
                if h == 1 and w == 1:
                    continue
                num_neighbors += padded[h:padded.shape[0] - (2 - h), w:padded.shape[1] - (2 - w)]

        print(f"num neighbors: {np.unique(num_neighbors)}")
        born = (1 - grid) * (num_neighbors == 3)
        alive = grid * (num_neighbors >= 2) * (num_neighbors <= 3)

        print(np.sum(alive + born))

        return alive + born

    def generate(self, context: PatternContext) -> np.ndarray:

        seg_image = context.stim_seg()

        # on first turn, use random initialization
        if self.turn_zero:
            self.turn_zero = False
            return grid_to_image(self.grid, self.pattern_shape, self.diameter_ratio)


        seg_grid = image_to_grid(seg_image, self.grid_size, self.diameter_ratio)
        current_grid = seg_grid > 0.5

        # diagnostic info
        print(f"Avg accuracy: {np.mean(current_grid == self.grid)}")
        print(f"Number of detected on squares: {np.sum(current_grid)}")
        print(f"Number of detected off squares: {np.sum(1 - current_grid)}")
        print(f"On squares not detected: {np.sum((current_grid == 0) & (self.grid == 1))} / {np.sum(self.grid)}" )
        print(f"Off squares counted as on: {np.sum((current_grid == 1) & (self.grid == 0))} / {np.sum(self.grid == 0)}")

        next_grid = self.advance_gol(current_grid)
        print(f"next step: {np.sum(next_grid)}")
        self.grid = next_grid

        next_pattern = grid_to_image(next_grid, self.pattern_shape, self.diameter_ratio)

        return next_pattern

BASE_PATH = r"C:\Users\Nikon\Desktop\Code\Toettchlab-FBC\test_experiment_outputs\test_gol"

def main():

    pattern_methods = {"game_of_life": GameOfLife}
    segmentation_methods = {"segment_game_of_life": SegmentGameOfLife}

    run_pyclm(BASE_PATH, pattern_methods=pattern_methods, segmentation_methods=segmentation_methods)

if __name__ == "__main__":
    main()


