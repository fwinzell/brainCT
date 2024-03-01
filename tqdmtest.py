import time
import tqdm

outer_loop = tqdm.tqdm([10, 20, 30, 40, 50], desc=" outer", position=1, leave=False)
for outer in outer_loop:
    outer_loop.set_description(f"Iteration [{outer}/{50}]")
    inner_loop = tqdm.tqdm(range(outer), desc=" inner loop", position=0, leave=False)
    for inner in inner_loop:
        time.sleep(0.05)
print("done!")
