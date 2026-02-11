from pathlib import Path

from pyclm import run_pyclm

DIR_PATH = Path(r"E:\Harrison\cells\RPE\20260122_rotations")

def main():
    run_pyclm(DIR_PATH)

if __name__ == "__main__":
    main()