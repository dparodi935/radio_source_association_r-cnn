""" Run from folder containing auto_run.py
"""
import subprocess, itertools, datetime, pathlib

# names and then creates folder to contain the weights and logs
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
logdir = pathlib.Path("runs")
logdir.mkdir(parents=True, exist_ok=True)

for enc, seed, ep_num in itertools.product(["radio3", "radio2_w1", "radio2_w1m", "radio3_w1", "radio3_w1m"], [2], ["10"]):
    # creates the unique names for each run
    tag = f"{enc}_s{seed}_ep{ep_num}"
    #w = logdir / f"weights_{tag}.pt"
    w = logdir/".."/"testing_weights"/f"weights_{tag}.pt"
    for name, cmd in [
        ("infer", ["python", "-u", "infer.py", "--data-root", r"..\cnn_data",
                   "--split", "test", "--weights", str(w)]),
    ]:
        print(f"=== {tag} {name} ===", flush=True)
        
        # opens .log file in write mode
        with open(logdir/ ".." / "test_results" / f"{name}_{tag}_test.log", "w") as fh:
            #executes the command lines and stores the terminal output in the log file    
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
            