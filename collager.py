import os

from PIL import Image

base_dir = "./art-only.art_and_num/" 
#base_dir = "./pc_gc_uc_negsp_artsc/" 
#base_dir = "./gc_uc_negsp_artsc/" 
#base_dir = "./gc_uc_negsp_artsc/" 
#base_dir = "./gc_uc_negsp_artsc/" 

for dir in os.listdir(base_dir):
    if not dir.endswith(".out") and not dir.endswith(".swp"):
        reps = [i for i in os.listdir(base_dir + dir) if i.isnumeric()]
        #reps = range(1,15)
        for r in reps:
            dir_check = f"{base_dir}{dir}/{r}/{r}/"
            #dir_check = f"{base_dir}{dir}/{r}/{r}/{r}/"
            if not os.path.exists(dir_check):
                continue
            pics = [i for i in os.listdir(f"{base_dir}{dir}/{r}/{r}/") if "img" in i and int(i.split("-")[1].split(".")[0]) > 200]
            print(len(pics))
            
            collage = Image.new('RGB', (5000, 5500))
            for i in range(0,5500,500):
                for j in range(0,5000,500):
                    try:
                        img = Image.open(f"{base_dir}{dir}/{r}/{r}/{pics.pop(0)}")
                        collage.paste(img, (j, i))
                    except:
                        pass
            
            collage.save(f"{base_dir}{dir}_{r}_collage.png")
