import os

from PIL import Image

#base_dir = "./pc_gc_uc_negsp_artsc/" 
#base_dir = "./gc_uc_negsp_artsc/" 
#base_dir = "./gc_uc_negsp_artsc/" 
#base_dir = "./gc_uc_negsp_artsc/" 

#ecs = [1,5,6,7]
#for i in ecs:
for i in range(1,64):
    #base_dir = f"./GPTP-exprs-timed/ec{i}/" 
    #base_dir = f"./GPTP-exprs-3min-no-FF1/ec{i}/" 
    base_dir = f"./GPTP-exprs-fixedmut-3min/ec{i}/" 
    #base_dir = f"./in-progress/ec{i}/" 

    for dir in os.listdir(base_dir):
        if not dir.endswith(".out") and not dir.endswith(".swp"):
            reps = [i for i in os.listdir(base_dir + dir) if i.isnumeric()]
            print(base_dir+dir)
            #reps = range(1,15)
            for r in reps:
                dir_check = f"{base_dir}{dir}/{r}/{r}/"
                #dir_check = f"{base_dir}{dir}/{r}/{r}/{r}/"
                if not os.path.exists(dir_check):
                    continue
                pics = [i for i in os.listdir(f"{base_dir}{dir}/{r}/{r}/") if "img" in i and int(i.split("-")[1].split(".")[0]) > 200]
                pics.pop()
                print(dir_check, len(pics))
                
                collage = Image.new('RGB', (5000, 5500))
                for i in range(0,5500,500):
                    for j in range(0,5000,500):
                        try:
                            img = Image.open(f"{base_dir}{dir}/{r}/{r}/{pics.pop(0)}")
                            collage.paste(img, (j, i))
                        except:
                            pass
                
                collage.save(f"{base_dir}{dir}_{r}_collage.png")
