import numpy as np
import pandas as pd
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Partition data')
    parser.add_argument('--dataset', default='bank', choices=['bank', 'electricity'])
    parser.add_argument('--num_partition', default=20, type=int)
    parser.add_argument('--portion', default=0.05, type=float)
    parser.add_argument('--overlap', default=0, type=int)
    args = parser.parse_args()
    return args

def main(args):
    filename = f'data/{args.dataset}/train.csv'
    df = pd.read_csv(filename)
    # assert valid args
    assert(int(1/args.portion*(args.overlap+1)) >= args.num_partition)

    # use hash to partition data
    pre_hash_data = df.values[:,:-1]
    idxgroup_final = None
    for time in range(args.overlap+1): # update hash value
        hash_data = [hash(str(data)+str(time)) % (int(1/args.portion*(args.overlap+1))) for data in pre_hash_data]
        idxgroup = [np.nonzero((hash_data == np.array(i)))[0] for i in range(args.num_partition)]
        if idxgroup_final == None:
            idxgroup_final = idxgroup
        else:
            idxgroup_final = [np.concatenate([idxgroup_final[i],idxgroup[i]], axis=0) for i in range(args.num_partition)]
    # no replacement
    idxgroup_final = np.array([np.unique(idxgroup_final[i]) for i in range(args.num_partition)])

    # check array length
    for i in range(args.num_partition):
        print(len(idxgroup_final[i]))

    save_name = f'partitions/{args.dataset}/hash_portion{args.portion}_partition{args.num_partition}_overlap{args.overlap}.npy'
    np.save(save_name, idxgroup_final, allow_pickle=True)

if __name__ == '__main__':
    main(parse_arguments())