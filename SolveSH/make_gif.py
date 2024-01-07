import os
import imageio
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

fname = "SHCncvDisc_v2_alpha_25em1_ps_5em1_v2"
temp_dir = os.getcwd()+"/gifs/temp"
gif_dir = os.getcwd()+"/gifs"
data = sio.loadmat(os.getcwd()+"/data/"+fname+".mat")

nx = np.shape(data['xx'])[0]
ny = np.shape(data['yy'])[0]

# x_st = round(nx/4)
# x_end = x_st+2*x_st
# y_st = round(ny/4)
# y_end =y_st+2*y_st
# x_st = 0
# x_end = -1
# y_st = 0
# y_end = -1
x_st = 0
x_end = -1
y_st = 0
y_end = round(ny/2)

fps = 5

for idx in range(np.shape(data['uu'])[2]):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3,3))
    ax.imshow(data['uu'][y_st:y_end,x_st:x_end,idx],cmap='gray')
    plt.savefig(temp_dir+"/sh_"+f"{idx:0>4}"+".png", bbox_inches='tight')
    plt.close()

images = []
for file_name in sorted(os.listdir(temp_dir)):
    if file_name.endswith('.png'):
        file_path = os.path.join(temp_dir, file_name)
        images.append(imageio.v2.imread(file_path))
imageio.mimsave(gif_dir+'/{}.gif'.format(fname), images, duration=int(1000*(1/fps)))

for f in os.listdir(temp_dir):
    if f.startswith('sh'):
        os.remove(os.path.join(temp_dir,f))