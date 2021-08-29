import argparse

from torch.nn import parameter

from loader import MoleculeDataset
from torch_geometric.data import DataLoader
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Batch
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt 
from model import GNN, GNN_graphpred,js_loss
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd
        # state_c=self.js_div(node_representation,guidance_representation)
        # r=self.state_p-self.y_*state_c
        # self.state_p=state_c.item()
from tensorboardX import SummaryWriter

criterion = nn.BCEWithLogitsLoss(reduction = "none")
#criterion_SDG=js_loss(get_softmax=True)
def train(args, model,device, loader, optimizer,optimizer_SDG,guidance_dataset):
    model.train()

    n_j=3;loss_SDG=0;state=torch.randn(1)[0].item();gamma=0.99;target_emb=None;loss_sum=0
    # guidance_dataset=None
    # for batch in val_loader:
    #     guidance_dataset=batch.to(device)
    for l in range(n_j):
        sigma_r = 0;pi_list=[],sigma_r_list=[]
        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            #batch_=batch
            if step==100:
                break
            batch = batch.to(device)

            data_,pred,action,pi,js = model(batch,guidance_dataset)

            y = data_.y.view(pred.shape).to(torch.float64)

            #Whether y is non-null or not.
            is_valid = y**2 > 0
            #Loss matrix
            loss_mat = criterion(pred.double(), (y+1)/2)
            #loss matrix after removing null target
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

            optimizer.zero_grad()
            loss = torch.sum(loss_mat)/torch.sum(is_valid)
            loss.backward(retain_graph=True)

            optimizer.step()
            optimizer.zero_grad()
            #calculate distribution discrepancy
            #loss_js=criterion_SDG(source_emb,target_emb)
            #r(sj−1, aj, sj) ←
            r=state-gamma*js
            sigma_r=sigma_r+ r*gamma
            state=js.item()
            pi_list.append(pi)
            
            #loss_SDG=r+loss
            #loss_sum+=loss_SDG.item()
            # loss_SDG.backward()
            # optimizer.zero_grad()
            # loss.backward()
            #if step%4==0:
            
            #loss_SDG=(loss_SDG/(4))
    for x in pi_list:
        # x*sigma_r
    loss_SDG=loss_SDG/n_j
    optimizer_SDG.step()
    optimizer_SDG.zero_grad()
    return (loss_sum/(step+1)/n_j)


def eval(args, model, device, loader, normalized_weight):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape).cpu())
        y_scores.append(pred.cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_scores = torch.cat(y_scores, dim = 0).numpy()

    roc_list = []
    weight = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))
            weight.append(normalized_weight[i])

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    weight = np.array(weight)
    roc_list = np.array(roc_list)

    return weight.dot(roc_list)
def finetune_train(args, model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()

        optimizer.step()

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=1,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--hidden', type=int, default=30,
                        help='embedding dimensions (default: 30)')
    parser.add_argument('--dropout_ratio', type=float, default=0.2,
                        help='dropout ratio (default: 0.2)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default = 'chembl_filtered', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--guidance_dataset', type=str, default = 'bace', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--output_model_file', type = str, default = 'd:/python workspace/pretrain-gnns/model', help='filename to output the pre-trained model')
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    args = parser.parse_args()


    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    # import pre-training dataset
    #Bunch of classification tasks
    if args.dataset == "chembl_filtered":
        num_tasks = 1310
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    dataset = MoleculeDataset("./dataset/chem_dataset/dataset/" + args.dataset, dataset=args.dataset)
    print(dataset)
    #import guidance_dataset
    if args.guidance_dataset == "tox21":
        guidance_num_tasks = 12
    elif args.guidance_dataset == "hiv":
        guidance_num_tasks = 1
    elif args.guidance_dataset == "pcba":
        guidance_num_tasks = 128
    elif args.guidance_dataset == "muv":
        guidance_num_tasks = 17
    elif args.guidance_dataset == "bace":
        guidance_num_tasks = 1
    elif args.guidance_dataset == "bbbp":
        guidance_num_tasks = 1
    elif args.guidance_dataset == "toxcast":
        guidance_num_tasks = 617
    elif args.guidance_dataset == "sider":
        guidance_num_tasks = 27
    elif args.guidance_dataset == "clintox":
        guidance_num_tasks = 2
    else:
        raise ValueError("Invalid guidance_dataset name.")

    #set up dataset
    guidance_dataset = MoleculeDataset("./dataset/chem_dataset/dataset/"+args.guidance_dataset, dataset=args.guidance_dataset)
    print(guidance_dataset)
    
    # guidance_dataset split into guidance_dataset finetuing dataset and testing dataset
    if args.split == "scaffold":
        smiles_list = pd.read_csv('./dataset/chem_dataset/dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(guidance_dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(guidance_dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv('./dataset/chem_dataset/dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(guidance_dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random scaffold")
    else:
        raise ValueError("Invalid split option.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    guidance_dataset=Batch.from_data_list([valid_dataset[i] for i in range(len(valid_dataset))]).to(device)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)

    #set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, guidance_num_tasks,args.batch_size,args.hidden,device,JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file + ".pth")
    
    model.to(device)

    print(model)

    #set up optimizer
    
    optimizer = optim.Adam([param if name!='SDG' else None for name, param in model.named_parameters()], lr=args.lr, weight_decay=args.decay)
    #optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_SDG = optim.Adam(model.SDG.parameters(), lr=args.lr, weight_decay=args.decay)
    print(optimizer)
    #print(optimizer_SDG)

    loss_list=[]
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))

        loss=train(args, model, device, loader, optimizer,optimizer_SDG,guidance_dataset)
        loss_list.append(loss)


    
    x = np.arange(1,args.epochs+1) 
    y = np.array(loss_list)
    plt.title("Matplotlib demo") 
    plt.xlabel("x axis caption") 
    plt.ylabel("y axis caption") 
    plt.plot(x,y) 
    plt.show()


    if not args.output_model_file == "":
        torch.save(model.gnn.state_dict(), args.output_model_file + ".pth")


if __name__ == "__main__":
    import os
    print(os.getcwd())
    main()
# "--output_model_file","d:/python workspace/pretrain-gnns/model",