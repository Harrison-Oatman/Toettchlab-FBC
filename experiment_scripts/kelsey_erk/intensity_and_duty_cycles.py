from pathlib import Path
import numpy as np

from pyclm import run_pyclm
from pyclm import PatternMethod
from pyclm.core.patterns import PatternContext, DataDock

class QuadrantPattern(PatternMethod):
    name = "quadrant"

    def __init__(self, gap_width_um=50, intensities=(5, 20, 50, 100), **kwargs):

        super().__init__(**kwargs)

        self.gap_width_um = gap_width_um
        self.intensities = intensities


    def generate(self, context: PatternContext):

        xx, yy = self.get_um_meshgrid()
        y_center, x_center = self.center_um()

        output = np.zeros_like(xx)

        output[(xx > x_center) & (yy > y_center)] = self.intensities[0]
        output[(xx > x_center) & (yy < y_center)] = self.intensities[1]
        output[(xx < x_center) & (yy > y_center)] = self.intensities[2]
        output[(xx < x_center) & (yy < y_center)] = self.intensities[3]

        output[np.abs(xx - x_center) < self.gap_width_um / 2] = 0
        output[np.abs(yy - y_center) < self.gap_width_um / 2] = 0

        return output / 100

class DutyCyclePattern(PatternMethod):
    name = "duty_cycle"

    def __init__(self, full_duration_mins=20, duty_cycle=0.5, **kwargs):

        super().__init__(**kwargs)

        self.full_duration_mins = full_duration_mins
        self.duty_cycle = duty_cycle

    def generate(self, context: PatternContext) -> np.ndarray:
        # convert from seconds to minutes
        time_mins = context.time / 60

        rescaled_time = time_mins / self.full_duration_mins
        time_fractional_part = rescaled_time % 1.

        is_on = time_fractional_part < self.duty_cycle

        xx, yy = self.get_um_meshgrid()

        return np.ones_like(xx) * is_on

DIR_PATH = Path(r"D:\Kelsey\20260224_intensity_and_duty_cycle_part2")

def main():
    pattern_methods = {
        "quadrant": QuadrantPattern,
        "duty_cycle": DutyCyclePattern
    }

    run_pyclm(DIR_PATH, pattern_methods=pattern_methods)

if __name__ == "__main__":
    main()