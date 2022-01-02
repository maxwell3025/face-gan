import numpy as np
import glob
import PIL
import PIL.Image as Image

data = []
for filename in glob.glob("data/raw/*.png"):
    img = np.array(Image.open(filename).convert("RGB"))
    data.append((img/128.0)-1)
np.save("data/data.npy",np.stack(data))

