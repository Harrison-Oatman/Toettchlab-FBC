"""
WARNING: This test uses actual hardware to run
"""

from pyclm import run_pyclm
from pathlib import Path
import sys

TEST_DIR = Path(r"C:\Users\Nikon\Desktop\Code\Toettchlab-FBC\test_experiment_outputs\test_pyclm_runs")
# PYCLM_CONFIG = Path(r"C:\Users\Nikon\Desktop\Code\Toettchlab-FBC\test_pyclm.py")


def main():

    run_pyclm(TEST_DIR)

    print("complete")

    import threading
    import traceback

    for thread in threading.enumerate():
        print(f"\nThread: {thread.name}")
        print(f"  daemon: {thread.daemon}")
        print(f"  alive: {thread.is_alive()}")

        if thread.ident is not None:
            print(thread.ident)
            stack = traceback.format_stack(
                sys._current_frames()[thread.ident]
            )
            print("".join(stack))

    import multiprocessing as mp
    import gc

    def find_active_queue_feeders():
        results = []
        for obj in gc.get_objects():
            if isinstance(obj, mp.queues.Queue):
                t = getattr(obj, "_thread", None)
                if t and t.is_alive():
                    results.append((obj, t.ident))
        return results

    for q, tid in find_active_queue_feeders():
        print(f"Active QueueFeederThread {tid} owned by queue {q}")
        print(q.__dict__)


if __name__ == "__main__":
    main()