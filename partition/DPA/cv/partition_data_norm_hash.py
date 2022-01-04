import torch
import torchvision
import argparse
import numpy as  np
import PIL

parser = argparse.ArgumentParser(description='Partition Data')
parser.add_argument('--dataset', default="fashionMnist", type=str, help='dataset to partition')
parser.add_argument('--portion', default=0.005, type=float, help="subtrain set size")
parser.add_argument('--overlap', default=0, type=int, help='train set overlap number')
parser.add_argument('--partitions', default=50, type=int, help='number of partitions')
args = parser.parse_args()
channels = 3
if (args.dataset == "mnist"):
	data = torchvision.datasets.MNIST(root='./data', train=True, download=True)
	channels = 1

if (args.dataset == 'fashionMnist'):
	data = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True)
	channels = 1

if (args.dataset == 'svhn'):
	data = torchvision.datasets.SVHN(root='./data', split='train', download=True)

if (args.dataset == "cifar"):
	data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

imgs, labels = zip(*data)
finalimgs = torch.stack(list(map((lambda x: torchvision.transforms.ToTensor()(x)), list(imgs))))
for_sorting = (finalimgs*255).int()

idxgroup_final = None
for time in range(args.overlap+1):
    sort_list = (for_sorting.reshape(for_sorting.shape[0],-1).sum(dim=1)+time).numpy().tolist()
    hash_sorting = torch.tensor(list(map(hash, sort_list)))
    intmagessum = hash_sorting % int(1/args.portion*(args.overlap+1))
    idxgroup = list([(intmagessum  == i).nonzero() for i in range(args.partitions)])
    if idxgroup_final == None:
        idxgroup_final = idxgroup
    else:
        idxgroup_final = list([torch.cat((idxgroup_final[i], idxgroup[i]), dim=0) for i in range(args.partitions)])

# idxgroup = list([(intmagessum  == i).nonzero() for i in range(args.partitions)])
#force index groups into an order that depends only on image content  (not indexes) so that (deterministic) training will not depend initial indices
idxgroup = idxgroup_final
idxgroup = list([idxgroup[i][np.lexsort(torch.cat((torch.tensor(labels)[idxgroup[i]].int(),for_sorting[idxgroup[i]].reshape(idxgroup[i].shape[0],-1)),dim=1).numpy().transpose())] for i in range(args.partitions) ])
idxgroupout = list([x.squeeze().numpy() for x in idxgroup])
for i in range(args.partitions):
	print(idxgroupout[i].shape)
means = torch.stack(list([finalimgs[idxgroup[i]].permute(2,0,1,3,4).reshape(channels,-1).mean(dim=1) for i in range(args.partitions) ]))
stds =  torch.stack(list([finalimgs[idxgroup[i]].permute(2,0,1,3,4).reshape(channels,-1).std(dim=1) for i in range(args.partitions) ]))
out = {'idx': idxgroupout,'mean':means.numpy(),'std':stds.numpy() }
torch.save(out, "partitions_hash_mean_" +args.dataset+'_'+str(args.partitions)+'_'+str(args.portion)+'.pth')