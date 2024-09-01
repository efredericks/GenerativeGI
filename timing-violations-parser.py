import os

timings = {}
for i in range(1,65):
    timings[f"ec{i}"] = {}

for fname in os.listdir("."):
    if fname.endswith(".out"):
        with open(os.path.join(".", fname)) as f:
            lines = f.readlines()
            fname_s = fname.split(".")
            first = fname_s[0].split("-")
            idx = first[-1]


            for line in lines:
                if "Violated timing" in line:
                    sline = line.strip().split(" ")
                    print(int(sline[0]))
                    #try:
                    #    gid = int(sline[0])
#
#                        if gid in timings[idx]:
#                            timings[idx][gid] += 1
#                        else:
#                            timings[idx][gid] = 0
#                    except:
#                        print("Parsing issue:", fname, line, "Violated timing" in line)

#for k,v in timings.items():
#    print(f"{k}: {len(v)}")


