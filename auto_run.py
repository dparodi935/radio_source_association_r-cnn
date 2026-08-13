import subprocess, itertools, datetime, pathlib

# names and then creates folder to contain the weights and logs
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
logdir = pathlib.Path("runs") / stamp
logdir.mkdir(parents=True, exist_ok=True)

for enc, seed, ep_num in itertools.product(["radio3", "radio2_w1", "radio2_w1m", "radio3_w1", "radio3_w1m"], [1,2], ["10"]):
    # creates the unique names for each run
    tag = f"{enc}_s{seed}_ep{ep_num}"
    w = logdir / f"weights_{tag}.pt"
    
    for name, cmd in [
        ("train", ["python", "-u", "train.py", "--data-root", r"..\cnn_data",
                   "--encoding", enc, "--torch-seed", str(seed),
                   "--epochs", ep_num, "--out", str(w)]),
        ("infer", ["python", "-u", "infer.py", "--data-root", r"..\cnn_data",
                   "--split", "val", "--weights", str(w)]),
    ]:
        print(f"=== {tag} {name} ===", flush=True)
        
        # opens .log file in write mode
        with open(logdir / f"{name}_{tag}.log", "w") as fh:
            #executes the command lines and stores the terminal output in the log file    
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
            