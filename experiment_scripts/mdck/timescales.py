from pathlib import Path

import numpy as np
from pyclm import run_pyclm, PFSPositionMover, PatternMethod, PatternContext
from pyclm.core.patterns.fbc_cell_movement import PerCellPatternMethod
from typing import Literal

DIR_PATH = Path(r"E:\Harrison\mdck_mechanics\20260505_directions_and_dutycycles")

class PeriodDirectedPattern(PerCellPatternMethod):

    direction_vec = {
        "u": (1, 0),
        "d": (-1, 0),
        "r": (0, 1),
        "l": (0, -1)
    }

    def __init__(self, sequence: list[Literal['u', 'd', 'l', 'r', 's']] | str | None = None, period_h = 2, **kwargs):
        super().__init__(**kwargs)

        assert sequence, "sequence kwarg required but not provided"
        assert all([val in ["u", "d", "l", "r", "s"] for val in sequence]), \
            f"sequence {sequence} contains unrecognized characters"

        self.sequence = "".join([val for val in sequence])
        self.period_h = period_h

        self.current_direction = self.sequence[0]

    def process_prop(self, prop) -> np.ndarray:

        if self.current_direction == "s":
            return 0 * prop.image

        vec = self.direction_vec.get(self.current_direction, None)

        assert vec is not None, f"direction {self.current_direction} produced an invalid vector"

        return self.prop_vector(prop, vec)

    def generate(self, context) -> np.ndarray:

        t = context.time
        t_h = t / 3600

        step = t_h // self.period_h
        step = int(step % len(self.sequence))

        self.current_direction = self.sequence[step]

        return super().generate(context)


class AlternatingCagePattern(PatternMethod):

    def __init__(self, bar_width_um=50, period_h=1, **kwargs):
        super().__init__(**kwargs)

        self.bar_width_um = bar_width_um
        self.period_h = period_h


    def generate(self, context: PatternContext):

        t = context.time
        t_h = t / 3600

        parity = 0 if (t_h / self.period_h) % 1 < 0.5 else 1

        xx, yy = self.get_um_meshgrid()

        return ((xx // self.bar_width_um) % 2) == parity


class CircleDutyCyclePattern(PatternMethod):

    def __init__(self, radius_um=100, period_m=60, duty_cycle=0.5, **kwargs):
        super().__init__(**kwargs)

        self.radius_um = radius_um
        self.duty_cycle = duty_cycle
        self.period = period_m

    def generate(self, context: PatternContext) -> np.ndarray:

        t_m = context.time / 60
        on = (t_m / self.period) % 1 < self.duty_cycle

        xx, yy = self.get_um_meshgrid()
        x_center, y_center = self.center_um()

        return ((xx - x_center)**2 + (yy - y_center)**2 < self.radius_um**2).astype(np.float32) * on


def main():
    run_pyclm(DIR_PATH, position_mover=PFSPositionMover(), pattern_methods={
        "direction": PeriodDirectedPattern,
        "cage": AlternatingCagePattern,
        "circle_duty": CircleDutyCyclePattern,
    })

if __name__ == "__main__":
    main()