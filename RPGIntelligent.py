from sample_generator import Sample
from dataset_preprocessor import Preprocessor
from FineTuner import FineTuner as Ft
from main import Game
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sprites import *
import sys
import time
from multiprocessing import Process
from dataset_collector import gather_samples

s = Sample("prompt", "response")
p = Preprocessor()
tuner = Ft()  # FineTuner class instance

import os

def get_last_checkpoint():
    f = Ft()

    # 1. Find the latest checkpoint manually
    checkpoints = [chk for chk in os.listdir(f.trained_model)
                   if chk.startswith('checkpoint-')]
    if checkpoints:
        latest = max(checkpoints, key=lambda x: int(x.split('-')[1]))
        last_checkpoint = os.path.join(f.trained_model, latest)

        # Escape backslashes for Windows compatibility
        return last_checkpoint.replace("\\", "\\\\")
    else:
        return None

def FineTune(sample, processor, tuner):
    """This runs once and then stops itself."""
    latest_checkpoint = get_last_checkpoint()
    if sample.can_train_model():
        processor.import_to_main_dataset()
        time.sleep(1)
        processor.save_formatted_data()
        processor.tokenize()
        tuner.finetune_model(latest_checkpoint)
        print("FineTune finished and thread exiting.")
    else:
        print("Not enough data to finetune")


def start_finetune_after_delay(delay=10, interval=300):
    """Wait `delay` seconds, then run FineTune every `interval` seconds."""
    time.sleep(delay)
    while True:
        FineTune(s, p, tuner)
        time.sleep(interval)


# ============ MAIN PROGRAM ============
if __name__ == "__main__":
    # Start FineTune process (separate from game)

    finetune_process = Process(target=start_finetune_after_delay, args=(10, 300))
    finetune_process.start()
    #gather_samples()



    g = Game()
    # Game loop in main process
    g.new()
    while g.running:
        g.main()
        if g.won:
            g.game_won()
        else:
            g.game_over()

    pygame.quit()
    sys.exit()   # Game exits, but finetune_process keeps running


#---------------------------------------------------------------------------------------
# To gather data while fine tuning
#--------------------------------------------------------------------------------------
    """
    if __name__ == "__main__":
    # Start FineTune process
    finetune_process = Process(target=start_finetune_after_delay, args=(10, 300))
    finetune_process.start()

    # Start gather_samples in parallel too (if it should run concurrently)
    gather_process = Process(target=gather_samples)
    gather_process.start()

    # Wait for BOTH processes with 1-hour timeout
    finetune_process.join(timeout=200)
    gather_process.join(timeout=0)  # Don't wait if already finished

    # Terminate both if fine-tuning is still running (meaning timeout hit)
    if finetune_process.is_alive():
        print("⏰ Time's up! Stopping all processes...")
        finetune_process.terminate()
        gather_process.terminate()  # Also stop gathering
        finetune_process.join()
        gather_process.join()
"""