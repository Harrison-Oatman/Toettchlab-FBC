from pathlib import Path

from pyclm import run_pyclm

DIR_PATH = Path(r"E:\Harrison\mdck_revision_experiments\20260207_speeds_0-075b")

def main():
    run_pyclm(DIR_PATH)

if __name__ == "__main__":
    main()