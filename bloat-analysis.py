import os
import pandas as pd
base_dir = "../results/"

expr_names = ["EC9", "EC31", "EC27", "EC60"]
exprs = [
"./art-only.art_and_num",
"./gc_uc_cc_negsp_artsc",
"./gc_uc_negsp_artsc",
"./pc_gc_uc_negsp_artsc",
]

base_dir=exprs[1]
dfs = []
#for dir in [j for j in os.listdir(base_dir) if os.path.isdir(base_dir + j)]:
#    reps = [i for i in os.listdir(base_dir + dir) if i.isnumeric()]
#    for r in reps:
#        dir_check = f"{base_dir}{dir}/{r}/{r}/{r}/"
#        if not os.path.exists(dir_check):
#            continue

for i in range(len(exprs)):
    expr = exprs[i]
    expr_name = expr_names[i]
    print(f"Parsing {expr_names[i]}:{expr}")

    for path, dirs, files in os.walk(expr):
        for file in files:
            if file.endswith("_population.dat"):
                with open(os.path.join(path,file)) as f:
                    print(f"Parsing {file}")


                    pop_file = f#[i for i in os.listdir(f"{base_dir}{dir}/{r}/{r}/{r}/") if "population" in i][0]
                    dfs.append(pd.read_csv(os.path.join(path,file), sep="\t", header=None, names=["ind_id","fitnesses","genome"]))
                    dfs[-1]["exp"] = path#dir
                    r = path.split('/')[-1]
                    dfs[-1]["rep"] = r

df = pd.concat(dfs)
# Reorder columns
df = df[["exp","rep","ind_id","fitnesses","genome"]]
# Create a column “technique_count” that counts the number of times a technique is used in the genome
df["technique_count"] = df["genome"].apply(lambda x: len(x.split(",")))
df["unique_technique_and_params_count"] = df["genome"].apply(lambda x: len(set(x.split(","))))
df.head()
