from pathlib import Path

from pyclm import run_pyclm

DIR_PATH = Path(r"E:\Harrison\mdck_revision_experiments\20260201_control_newcal")

def main():
    run_pyclm(DIR_PATH)

if __name__ == "__main__":
    main()