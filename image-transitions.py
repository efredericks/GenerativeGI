import os
from PIL import Image
import numpy as np

def v_fade(width = 8):
        '''make masks for fading from one image to the next through a vertical sweep.  Does this through numpy slices'''
        global img1                #from img1 = Image.open('one.png')
        n = np.array(img1)        #make a numpy.array the same size as the frame
        n.dtype=np.uint8        #correct the data type
        n[:][:] = 0                #the first mask lets everything through
        #copy the array, else all masks will be the last one
        result=[Image.fromarray(n.copy())]
        for i in range(int(n.shape[0]/width+1)):
            #add vertical strips of width one at a time.
            y=i*(width)
            n[y:y+width][:] = 255
            result.append(Image.fromarray(n.copy()))
        return result

img1, img2 = (Image.open('%s' % x) for x in ['paper-img/drunk.png','paper-img/dither.png'])
#fade from 1 to 2
images = [Image.composite(img2,img1,mask) for mask in v_fade()]
#fast on transition frams but hold end images longer
durations = ['.10']*len(images)
durations[0] = durations[-1] = '2.5'
#fade from 2 to 1
images.extend([Image.composite(img1,img2,mask) for mask in v_fade()])
durations.extend(durations)
# writeGif('my.gif',images, duration=durations)

i = 0
print(images,durations)
for f in images:
      f.save(f"{i}_transition.png")
      i += 1
