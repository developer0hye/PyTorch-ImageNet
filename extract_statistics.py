import os
import cv2
import numpy as np
from pathlib import Path

def dataset_load(root):
    imgs_path = []
    labels_path = []
    
    for r, d, f in os.walk(root):
        for file in f:
            if file.endswith((".png", ".jpg")):
                imgs_path.append(os.path.join(r, file).replace(os.sep, '/'))
            elif file.endswith(".txt"):
                labels_path.append(os.path.join(r, file).replace(os.sep, '/'))
            
    return imgs_path, labels_path

root = ""
imgs, _ = dataset_load(root)

print(len(imgs))

color_table = [(0, 255, 0), (0, 0, 255), (0, 0, 255), (0, 0, 255)]

r_mean_batch = []
g_mean_batch = []
b_mean_batch = []

r_std_batch = []
g_std_batch = []
b_std_batch = []

for img in imgs:
    img_file_name = Path(img).resolve().stem
    
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img /= 255.
    
    r_mean_batch.append(np.mean(img[..., 0]))
    g_mean_batch.append(np.mean(img[..., 1]))
    b_mean_batch.append(np.mean(img[..., 2]))
    
    r_std_batch.append(np.std(img[..., 0]))
    g_std_batch.append(np.std(img[..., 1]))
    b_std_batch.append(np.std(img[..., 2]))

r_mean_batch = np.mean(r_mean_batch)
g_mean_batch = np.mean(g_mean_batch)
b_mean_batch = np.mean(b_mean_batch)

r_std_batch = np.mean(r_std_batch)
g_std_batch = np.mean(g_std_batch)
b_std_batch = np.mean(b_std_batch)

print(r_mean_batch, g_mean_batch, b_mean_batch)
print(r_std_batch, g_std_batch, b_std_batch)
