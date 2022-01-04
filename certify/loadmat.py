from scipy.io import loadmat
import os

num_partition = 50

dataset = 'fashion_mnist'
mode = 'rob'
t_persample = 1
scale = 200
out_dir = os.path.join('/home/crx/collective/out', dataset+'partition'+str(num_partition),
                       mode)

out_file = os.path.join(
    out_dir, f'poison3to25_scale{scale}_t_persample{t_persample}.mat')
mat = loadmat(out_file)
print(mat)
