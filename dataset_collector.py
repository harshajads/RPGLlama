from Level_generator import *
from sample_generator import *
import random
import threading
import time


def gather_samples():
    while True:
        j = random.randint(1, 7)
        print(f"Difficulty: {j}")
        prompt, map = get_level(j)
        # print(prompt)
        # print(map)
        s = Sample(prompt, map)
        s.create_sample()
        s.save_sample()
        time.sleep(1)




# Create 4 threads
"""threads = []
for i in range(1):
    t = threading.Thread(target=gather_samples, name=f"T{i+1}")
    threads.append(t)
    t.start()

# Wait for all threads to finish
for t in threads:
    t.join()

print("All threads finished.")"""


