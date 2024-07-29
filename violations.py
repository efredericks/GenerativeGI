import os, sys
import seaborn as sns

base_dir = "./GPTP-exprs-timed"
expr_names = []
exprs = []
NUM_EXPR = 63



for i in range(1,NUM_EXPR+1):#64):
    expr_names.append(f"EC{i}")
    exprs.append(f"{base_dir}/ec{i}")

for expr in exprs:
    for path, dirs, files in os.walk(expr):
        for file in files:
            if file.endswith(".out"):
                with open(os.path.join(path,file)) as f:
                    print(f"Parsing {file}")
