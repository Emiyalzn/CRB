from utils import prepare_param, DPA_certify, BAG_certify
import numpy as np
import torch
import gurobipy as gp
import argparse
import csv
import time
import math
from scipy.io import savemat
import os

def data_cb(model, where):
    if where == gp.GRB.Callback.MIP:
        cur_obj = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
        cur_bd = model.cbGet(gp.GRB.Callback.MIP_OBJBND)

        # Did objective value or best bound change?
        if model._obj != cur_obj or model._bd != cur_bd:
            model._obj = cur_obj
            model._bd = cur_bd
            model._data.append([time.time() - model._start, cur_obj, cur_bd])


def gurobi_search(num_consider, num_classifiers, num_classes, num_poison, overlap, numvotes, label, preds, time_limit, verbose):
    m = gp.Model("attack")

    X = m.addVars(num_classifiers, vtype=gp.GRB.BINARY, name="X")
    Y = m.addVars(num_consider, vtype=gp.GRB.BINARY, name="Y")

    m.update()

    m.setObjective(gp.quicksum(Y[i]
                   for i in range(num_consider)), gp.GRB.MAXIMIZE)

    m.addConstr(gp.quicksum(X[i]
                for i in range(num_classifiers)) <= num_poison*(overlap+1))

    # print(numvotes)
    new_true_votes = []
    new_votes = []

    for i in range(num_consider):
        for j in range(num_classes):
            if j == label[i]:
                new_true_votes.append(
                    numvotes[i][j] - gp.quicksum(X[l] if preds[l][i] == j else 0 for l in range(num_classifiers)))
            else:
                new_votes.append(numvotes[i][j] + gp.quicksum(X[l]
                                    if preds[l][i] != j else 0 for l in range(num_classifiers)))

    for i in range(num_consider):
        if label[i] == 1:
            m.addConstr(
                (Y[i]-0.5)*(0.5-(new_votes[i] - new_true_votes[i] + 1)) <= 0)
        else:
            m.addConstr(
                (Y[i]-0.5)*(0.5-(new_votes[i] - new_true_votes[i])) <= 0)

    m.Params.LogToConsole = verbose
    m.Params.TimeLimit = time_limit
    # m.Params.MIPFocus = 3
    m._obj = None
    m._bd = None
    m._data = []
    m._start = time.time()
    m.optimize(callback=data_cb)
    obj = m._data[-1][1]
    upper_bound = m._data[-1][2]
    A = []
    for k, v in m.getAttr('X', X).items():
        if v == 1:
            A.append(k)
    print(A)

    return obj, upper_bound, A

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Certify Max Num of Poisoned Predictions.")
    parser.add_argument('mode', type=str, default='rob', choices=[
                        'rob', 'ca'], help='To certify robustness or to evaluate certified accuracy')
    parser.add_argument('dataset', default='bank',
                        choices=['bank', 'electricity'], help="Test dataset.")
    parser.add_argument('num_partition', type=int,
                        help='Test number of partitions.')
    parser.add_argument('--num_poison', default=1,
                        type=int, help='poison budget.')
    parser.add_argument('--overlap', default=0,
                        type=int, help='partition overlap.')
    parser.add_argument('--portion', default=0.05,
                        type=float, help="data for every partition")
    parser.add_argument('--model', default="bayes",
                        choices=['bayes', 'svm', 'logistic'], help='classic trained models')
    parser.add_argument('--num_classes', default=2,
                        type=int, help='number of classes.')
    parser.add_argument('--out', type=str, default='/home/crx/collective/out')
    args = parser.parse_args()
    return args   

def main(args):
    print(args)
    out_dir = os.path.join(args.out, args.dataset+'partition'+str(args.num_partition),
                           args.mode)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    num_classes = 2
    filein = np.load(f"evaluations/{args.dataset}/model_{args.model}_partition_{args.num_partition}"
                     f"_portion_{args.portion}_overlap_{args.overlap}.pkl", allow_pickle=True)
    labels = np.array(filein[0])
    preds = np.array(filein[1:])

    # if mode == 'ca', we only consider the correct predictions
    # if mode == 'rob', we consider all the predictions
    if args.mode == 'ca':
        numvotes = np.zeros((preds.shape[1], num_classes)) # [num_instance, num_class]
        for preds_subclassifer in preds:
            numvotes += np.eye(num_classes)[preds_subclassifer]
        idxsort = np.argsort(-numvotes, axis=1, kind='stable')
        num_total = (idxsort[:, 0] == labels).sum().item()  # correct
    elif args.mode == 'rob':
        num_total = len(preds[0])

    num_poison = args.num_poison

    scale = 10000
    print(f"================num_poison: {num_poison}==================")
    list_params = prepare_param(mode=args.mode, preds=preds, 
                                labels=labels, num_poison=num_poison, num_classes=args.num_classes, overlap=args.overlap, scale=scale)
    
    num_unattacked_col = num_total
    (num_consider, new_numvotes, new_label, new_preds) = list_params[0]
    obj, upper_bound, A = gurobi_search(num_consider=num_consider, num_classifiers=args.num_partition, num_classes=args.num_classes, num_poison=num_poison,
                                        overlap=args.overlap, numvotes=new_numvotes, label=new_label, preds=new_preds, 
                                        time_limit=600, verbose=False)
    print(f"objective: {math.floor(upper_bound)}, gap: {(num_consider-math.floor(upper_bound))/num_consider:.4f}")
        

if __name__ == '__main__':
    main(parse_arguments())

