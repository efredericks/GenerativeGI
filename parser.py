import os, sys
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import operator as op
import scipy.stats as stats
import pandas as pd
import numpy as np

expr_names = ["EC9", "EC31", "EC27", "EC60"]

exprs = [
"./art-only.art_and_num",
"./gc_uc_cc_negsp_artsc",
"./gc_uc_negsp_artsc",
"./pc_gc_uc_negsp_artsc",

#"./paper_runs/clear_lexicase",
#"./paper_runs/clear_single_pairwise",
#"./paper_runs/no_clear_lexicase",
#"./paper_runs/no_clear_single_pairwise",
#"./paper_runs/random",
]

times = {}
programs = {}


ffs = [
  'pc', 'gc', 'ut', 'cd', 'ns', 'ac'
]
expr_fit_correlation = {
  'EC9': [],
  'EC31': [],
  'EC27': [],
  'EC60': [],
}

def normalize_data(val, _min, _max, which='max', _new_min=0.0, _new_max=1.0):
    if which == 'max': # maximize
        return ((val - _min) / (_max - _min)) * (_new_max - _new_min) + _new_min
    else: # minimize
        return 1.0 - (((val - _min) / (_max - _min)) * (_new_max - _new_min) + _new_min)

def add_fitness_fn(idx, ff):
    expr_fit_correlation[idx].append(ff)

fit_data = {}
for i in range(1,65):
    fit_data[f"EC{i}"] = {}
    for ff in ffs:
        fit_data[f"EC{i}"][ff] = []

# track normalizing values
normalize_ffs = {}
for ff in ffs:
    normalize_ffs[ff] = {'max': -99999.9, 'min': 99999.9}
normalize_ffs['pc']['which'] = 'max'
normalize_ffs['gc']['which'] = 'min'
normalize_ffs['ut']['which'] = 'max'
normalize_ffs['cd']['which'] = 'max'
normalize_ffs['ns']['which'] = 'min'
normalize_ffs['ac']['which'] = 'min'

print(fit_data)
# parse fitnesses - need to correlate configs with ffs as we didn't track that in the runs
add_fitness_fn('EC9', ffs[2])
add_fitness_fn('EC9', ffs[5])

add_fitness_fn('EC31', ffs[1])
add_fitness_fn('EC31', ffs[2])
add_fitness_fn('EC31', ffs[3])
add_fitness_fn('EC31', ffs[4])
add_fitness_fn('EC31', ffs[5])

add_fitness_fn('EC27', ffs[1])
add_fitness_fn('EC27', ffs[2])
add_fitness_fn('EC27', ffs[4])
add_fitness_fn('EC27', ffs[5])

add_fitness_fn('EC60', ffs[0])
add_fitness_fn('EC60', ffs[1])
add_fitness_fn('EC60', ffs[2])
add_fitness_fn('EC60', ffs[4])
add_fitness_fn('EC60', ffs[5])

# fitness normalization and heatmap
"""
for i in range(len(exprs)):
    expr = exprs[i]
    expr_name = expr_names[i]
    print(f"Parsing {expr_names[i]}:{expr}")

    for path, dirs, files in os.walk(expr):
        for file in files:
            if file.endswith("_population.dat"):
                with open(os.path.join(path,file)) as f:
                    print(f"Parsing {file}")

                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        _line = line.split("\t")
                        fits = _line[1].strip()
                        fits = fits[1:] # parens
                        fits = fits[:-1]
                        fits = fits.split(',') # into vals

                        for j in range(len(fits)-1):
                            fit = float(fits[j].strip())
                            fit_id = expr_fit_correlation[expr_name][j]
                            fit_data[expr_name][fit_id].append(fit)

                            # normalize tracking
                            if fit > normalize_ffs[fit_id]['max']:
                                normalize_ffs[fit_id]['max'] = fit
                            if fit < normalize_ffs[fit_id]['min']:
                                normalize_ffs[fit_id]['min'] = fit


#    normalize_ffs[ff] = {'max': 0.0, 'min': 0.0}

# normalize fitness values now
#print(normalize_ffs)
for i in range(len(exprs)):
    expr = exprs[i]
    expr_name = expr_names[i]
    print(f"Normalizing {expr_names[i]}:{expr}")

    for fit_id in fit_data[expr_name]:
        for j in range(len(fit_data[expr_name][fit_id])):
            fit = fit_data[expr_name][fit_id][j]
            fit_data[expr_name][fit_id][j] = normalize_data(fit, normalize_ffs[fit_id]['min'], normalize_ffs[fit_id]['max'], which=normalize_ffs[fit_id]['which'])

# fill numpy array with averaged values to go into heatmap
fit_array = np.nan * np.empty((len(ffs), 64))
for i in range(1,64):#len(exprs)):
    expr_name = f"EC{i}"
    if expr_name in expr_names:
        for fit_id in fit_data[expr_name]:
            if len(fit_data[expr_name][fit_id]) > 0:
                avg = sum(fit_data[expr_name][fit_id]) / len(fit_data[expr_name][fit_id])
                fit_array[ffs.index(fit_id),i-1] = avg

#print(fit_array)

#fit_df = pd.DataFrame(fit_data)
xlabs = [f"EC{i}" for i in range(1,65)]
ylabs = [f"ff_{f}" for f in ffs]
sns.heatmap(data=fit_array, vmin=0.0, vmax=1.0, xticklabels=xlabs, yticklabels=ylabs, cmap='coolwarm', linewidths=0.5)
plt.show()
"""


# timing and program analysis
for expr in exprs:
    times[expr] = []
    programs[expr] = []

    fit_data[expr] = []

    for path, dirs, files in os.walk(expr):
        for file in files:
            if file.endswith(".out"):
                with open(os.path.join(path,file)) as f:
                    print(f"Parsing {file}")

                    start_parsing_progs = False
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()

                        split_line = line.split(" ")
                        if split_line[0] == "9999":
                            start_parsing_progs = True

                        # count number of programs - parse on )
                        if start_parsing_progs and len(split_line) > 4:
                            parsed_line = line.split(')')
                            pline = parsed_line[1].strip().split(',')
                            for _pline in pline:
                                prog = _pline.split(':')
                                if prog[0] != '': 
                                    programs[expr].append(prog[0])

                        # execution time
                        if "real" in line:
                            l = line.split('\t')
                            # there has to be a datetime function for this
                            #t = datetime.strptime(l[1], '%Mm:%Ss')

                            _time = l[1].split('m') # left side is minutes
                            mins = float(_time[0])    # right side is seconds s
                            secs = float(_time[1][:-1])
                            mins += secs / 60.
                            times[expr].append(mins)#(mins*60.)+secs)

# number of programs per config
for i in range(len(exprs)):#expr in exprs:
    expr = exprs[i]
    prog_counts = {x: programs[expr].count(x) for x in programs[expr]}

    maxTechnique = ""
    maxCounts = 0

    #print(expr, expr_names[i])
    for k,v in prog_counts.items():
        if v > maxCounts:
            maxCounts = v
            maxTechnique = k
    #    print(f"{k}:{v}")
    print(expr, "max technique")
    print(f":{maxTechnique}: {v}")
    print("----------")

#for i in range(len(exprs)):
#    print(exprs[i], expr_names[i])
#    print("---")
#    for e, v in programs.items():
#        print(e,v)
#
#    print("---")

# t tests
#print("0-1",stats.wilcoxon(times[exprs[0]], times[exprs[1]]))
#print("0-2",stats.wilcoxon(times[exprs[0]], times[exprs[2]]))
#print("0-3",stats.wilcoxon(times[exprs[0]], times[exprs[3]]))
print("1-2",stats.wilcoxon(times[exprs[1]], times[exprs[2]]))
print("1-3",stats.wilcoxon(times[exprs[1]], times[exprs[3]]))
print("2-3",stats.wilcoxon(times[exprs[2]], times[exprs[3]]))

#https://stackoverflow.com/questions/37576160/how-do-i-add-category-names-to-my-seaborn-boxplot-when-my-data-is-from-a-python
sns.set_theme(style="ticks")#, palette="pastel")
sns.color_palette('colorblind')
# sort keys and values together
sorted_keys, sorted_vals = zip(*sorted(times.items(), key=op.itemgetter(1)))

sns.set(context='notebook', style='whitegrid')
sns.utils.axlabel(xlabel="Experimental Configuration", ylabel="Time (minutes)", fontsize=16)
sns.boxplot(data=sorted_vals, width=.18)
sns.swarmplot(data=sorted_vals, size=6, edgecolor="black", linewidth=.9)
plt.xticks(plt.xticks()[0], expr_names)#sorted_keys)

plt.show()


# program counts
#prog_sorted_keys, prog_sorted_vals = zip(*sorted(programs.items(), key=op.itemgetter(1)))
#sns.set(context='notebook', style='whitegrid')
#sns.utils.axlabel(xlabel="Config", ylabel="Program count", fontsize=16)
#sns.boxplot(data=sorted_vals, width=.18)
#sns.swarmplot(data=sorted_vals, size=6, edgecolor="black", linewidth=.9)
#plt.xticks(plt.xticks()[0], expr_names)#sorted_keys)





