"""graph partition with balance and fairness, as described in https://arxiv.org/abs/2304.03093."""
# Author: Cheng-Long Wang <chenglong.wang@kaust.edu.sa>
# License: BSD 3 clause

import os
import time
import copy
import pickle
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from dataclasses import dataclass

import torch
import pandas as pd
import operator as op
from sklearn.cluster import KMeans
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, degree, to_dense_adj, to_scipy_sparse_matrix
import matplotlib.pyplot as plt

@dataclass
class Partition:
    method: str
    ids_shards: dict
    shards_ids: dict
    shards_edges: dict
    score_balance: float
    score_fair: float
    inter_pair: dict

class GUIDE:
    def __init__(self, edge_indexs, labels, k=2, NITER_BASE=10, NITER_SR=10):
        '''
        input:  edge_indexs: edge_indexs of graph
                labels: labels
                k: number of shards
                NITER_BASE: number of global iterations
                NITER_SR: number of iterations for SR method
        '''
        self.edge_indexs = edge_indexs
        self.labels = labels
        self.number_local = k
        self.NITER_BASE = NITER_BASE
        self.NITER_SR = NITER_SR
        self.method = 'Fast'

    def fit(self, method='Fast', alpha_=1e-2, beta_=1e-1, es_threshold = -0.05):
        '''
        input:  method: 'Fast' or 'SR'
                alpha_: hyperparameter for Fast and SR
                beta_: hyerparameter SR
                es_threshold: early stop es_threshold for SR method
        return: a class of Partition
                class Partition:
                    shards_ids: dict
                    shards_edges: dict
                    score_balance: float
                    score_fair: float
                    inter_pair: dict
        '''
        start = time.time()
        self.method = method
        deg = degree(self.edge_indexs[0])
        degreeindex = (deg != 0).nonzero(as_tuple=True)[0]
        self.degree0index = (deg == 0).nonzero(as_tuple=True)[0]
        edge_index_new, _ = subgraph(
            degreeindex, self.edge_indexs, relabel_nodes=True)
        labels_new = self.labels[degreeindex]

        number_nodes = len(labels_new)
        number_colors = len(labels_new.unique())
        # converting sensitive to a vector with entries in [colors] and buiding F %%%
        adjacency = to_dense_adj(
            edge_index_new).squeeze().to(labels_new.device)
        degrees = adjacency.sum(axis=0)
        if degrees.count_nonzero() != degrees.shape[0]:
            raise ValueError(
                'the input graph contains outlier points, leading to a 0-degree')

        if self.method == 'Fast':
            # torch.set_default_dtype(torch.float32)
            W = (adjacency+adjacency.T)/2
            degrees = W.sum(axis=1)
            D12 = torch.diag(degrees.reshape(-1).pow(-1/2))
            W = D12 @ W @ D12

            Fairness = torch.zeros(
                (number_nodes, number_colors))
            for i in range(number_colors):
                color_index = (labels_new == i).nonzero()
                Fairness[color_index, i] = 1

            B_left = Fairness.sum(axis=0).reshape(1, number_colors)
            B_right = torch.zeros((1, self.number_local))
            B_right[:] = torch.sqrt(torch.tensor(
                number_nodes/self.number_local))

            Balance = B_left.T @ B_right / torch.tensor(number_nodes)
            MB_ = alpha_ * Fairness @ Balance
            H, _ = self.GPI(W, alpha_, Fairness, MB_, Niter=self.NITER_BASE)
            km = KMeans(n_clusters=self.number_local, n_init=10, max_iter=50).fit(
                H.cpu().detach().numpy())
            assigns = dict(zip(degreeindex.tolist(), km.labels_))

        elif self.method == 'SR':
           # torch.set_default_dtype(torch.float32)
            W = (adjacency+adjacency.T)/2
            degrees = W.sum(axis=1)
            D12 = degrees.reshape(-1).pow(-1/2).diag()
            W = D12 @ W @ D12

            Fairness = torch.zeros(
                (number_nodes, number_colors))
            for i in range(number_colors):
                color_index = (labels_new == i).nonzero()
                Fairness[color_index, i] = 1

            B_left = Fairness.sum(axis=0).reshape(1, number_colors)
            B_right = torch.zeros((1, self.number_local))
            B_right[:] = torch.tensor(number_nodes/self.number_local).sqrt()

            Balance = B_left.T @ B_right / torch.tensor(number_nodes)
            MB1_ = alpha_ * Fairness @ Balance

            R0 = torch.eye(self.number_local)
            H0 = torch.empty(number_nodes, self.number_local)
            torch.nn.init.orthogonal_(H0)
            Y0 = torch.diag((H0@H0.T).diag()**(-0.5)) @ H0
            _, labels = Y0.max(dim=1)
            for i in range(self.number_local):
                idx = (labels == i).nonzero()
                Y0[:, i] = 0
                Y0[idx, i] = 1

            err = 1.0
            obj = []
            iter = 0
            while (iter < self.NITER_SR) and (err > 1e-3):

                # Rotation-Step (R0)
                M_ = Y0 @ torch.diag((Y0.sum(dim=0) + 1e-10)**(-1/2))
                MR = H0.T @ M_
                u_, _, v_ = torch.linalg.svd(MR, full_matrices=False)
                R0 = u_ @ v_.T

                # Embedding-Step (H0)
                MB_ = MB1_ + beta_ * M_ @ R0.T
                H0, _ = self.GPI(W, alpha_, Fairness, MB_,
                                 H_=H0, Niter=self.NITER_BASE)

                # Indicator-Step (Y0)
                Y0_ = self.SpecRotation(H0, R0, Niter=self.NITER_BASE)

                # # Objective Value
                # M_ = Y0_ @ torch.diag((Y0_.sum(dim=0) + 1e-10)**(-1/2))
                # obj_ = (H0.T @ W @ H0 - alpha_ *
                #         (H0.T @ Fairness) @ (Fairness.T @ H0)).trace() + alpha_*(Fairness.T @
                #                                                                  H0 - Balance).norm()**2 + beta_*(H0 @ R0 - M_).norm()**2
                # err = obj[iter-1]-obj_ if iter > 1 else err
                # if err > 0:
                #     Y0 = Y0_
                #     obj.append(obj_)

                # early stopping by measuring the partition scores
                Y0 = Y0_
                _, cluster_assign = Y0_.max(dim=1)
                assigns = dict(
                    zip(degreeindex.tolist(), cluster_assign.cpu().detach().numpy()))
                assigns_full = self.degree0complete(assigns)
                score_balance, score_fair = self.calculate_score(assigns_full)
                # is sum of score_balance and score_fair smaller than the threshold?
                if score_balance + score_fair > es_threshold:
                    print(score_balance, score_fair, es_threshold)
                    break

                iter = iter + 1

            _, cluster_assign = Y0.max(dim=1)
            assigns = dict(
                zip(degreeindex.tolist(), cluster_assign.cpu().detach().numpy()))

        assigns_full = self.degree0complete(assigns)
        self.p = self.partition_stat(assigns_full)

        end = time.time()
        print(
            f'method:{method}, running time:{end-start}, balance:{-self.p.score_balance}, fairness:{-self.p.score_fair}')
        return self.p

    def SpecRotation(self, H0, R0, Niter=10):
        _, assign_ = H0.max(dim=1)
        Y0 = torch.zeros(H0.shape)
        for i in range(H0.shape[1]):
            idx = (assign_ == i).nonzero()
            Y0[:, i] = 0
            Y0[idx, i] = 1

        HR = H0 @ R0
        sumY = (Y0**2).sum(dim=0)
        sumHY = HR.mul(Y0).sum(dim=0)
        seq_, _ = HR.max(dim=1)
        seq_, _ = HR.max(dim=1)
        _, idxi_ = seq_.sort()
        for t2 in range(Niter):
            converged = True
            for ii in range(H0.shape[0]):
                idxi = idxi_[ii]
                hri = HR[idxi, :]
                yi = Y0[idxi, :]
                _, id0_ = yi.max(dim=0)
                valuei_ = (sumHY + (1-yi).mul(hri)).mul((sumY + 1-yi).sqrt()**(-1)) - \
                    (sumHY - yi.mul(hri)).mul(((sumY-yi).sqrt()+1e-10)**(-1))
                _, id_ = valuei_.max(dim=0)
                if id_ != id0_:
                    converged = False
                    yi = torch.zeros((1, H0.shape[1]))
                    yi[0, id_] = 1
                    Y0[idxi, :] = yi
                    sumY[id0_] = sumY[id0_] - 1
                    sumY[id_] = sumY[id_] + 1
                    sumHY[id0_] = sumHY[id0_] - HR[idxi, id0_]
                    sumHY[id_] = sumHY[id_] + HR[idxi, id_]

            if converged:
                break

        return Y0

    def GPI(self, W, alpha_, Fairness, MB_, H_=None, Niter=10, PlotFlag=False):
        # MA_: n*n, MB_: n*m, n>=m
        # min{W.T @ W = I} Tr(W.T @ MA_ @ W -2 W.T @ MB_)
        # --> max{W.T @ W = I} Tr(W.T @ (alpha*I - MA_) @ W + 2 W.T @ MB_)
        # MA_ = torch.max(MA_, MA_.T)
        n_, m_ = MB_.shape
        if n_ < m_:
            raise ValueError('MB_.shape[0] should be larger than MB_.shape[1]')

        if H_ == None:
            H_ = torch.empty(n_, m_)
            torch.nn.init.orthogonal_(H_)

        ww = torch.ones(n_, 1)
        for i in range(10):
            m1 = W @ ww
            ww = m1/torch.linalg.norm(m1, 2)
        eta_ref = torch.abs(ww.T @ W @ ww)
        eta_ = 1e1**(len(str(int(eta_ref))))
        W_ = eta_*torch.eye(n_) + W

        err, iter = 1, 0
        obj = []
        while (iter < Niter) and (err > 1e-3):
            M = 2 * W_ @ H_ - 2 * alpha_ * \
                Fairness @ (Fairness.t() @ H_) + 2 * MB_
            u, s, v = torch.linalg.svd(M, full_matrices=False)
            H0_ = u @ v.T
            obj_ = torch.trace(H_.T @ W_ @ H_ + alpha_ *
                               (H_.T @ Fairness) @ (Fairness.t() @ H_) + 2 * H_.T @ MB_)
            err = obj_ - obj[iter-1] if iter > 0 else err
            if err > 0:
                H_ = H0_
                obj.append(obj_)

            iter = iter+1

        if PlotFlag:
            plt.figure()
            plt.plot(np.array(obj))

        return H_, err

    def degree0complete(self, assigns):
        if len(self.labels.shape) > 1:
            self.labels = self.labels[:, 0].to(torch.int64)

        queuelabel = self.labels[self.degree0index]
        queuedict = dict(zip(range(self.degree0index.shape[0]), queuelabel))
        n_ = self.labels.shape[0]
        odval = self.labels.bincount()
        balancemean = odval/n_

        shards_key = set(assigns.values())
        shards_number = len(set(assigns.values()))
        shards_index = list(assigns.keys())
        label_ = dict(
            zip(shards_index, self.labels[shards_index].detach().cpu().numpy()))
        shard_loop_ = [label_, assigns]
        shard_loopcomb = {}
        for k in label_.keys():
            shard_loopcomb[k] = tuple(shard[k] for shard in shard_loop_)

        value_list = list(shard_loopcomb.values())
        df = pd.DataFrame(value_list, columns=['True Type', 'Assigned Shard'])
        df = df.groupby(['True Type', 'Assigned Shard']
                        ).size().unstack(fill_value=0)

        count_shards = [op.countOf(assigns.values(), key)
                        for key in shards_key]

        bloss = torch.tensor(count_shards) - n_/shards_number
        sortedb, indicesb = torch.sort(bloss)
        for indb, keyb in enumerate(indicesb):
            if sortedb[indb] < 0:
                floss = torch.tensor(
                    df.values[:, keyb])-balancemean*n_/shards_number
                sortedf, indicesf = torch.sort(floss)
                for indf, keyf in enumerate(indicesf):
                    if torch.round(sortedf[indf]) < 0:
                        insertnum = torch.round(-sortedf[indf])
                        queuelist = (queuelabel == keyf).nonzero(
                            as_tuple=True)[0]
                        for keyq in queuelist:
                            tmp = queuedict.pop(int(keyq), None)
                            if tmp != None:
                                assigns[int(self.degree0index[keyq])
                                        ] = int(keyb)
                                insertnum -= 1
                                if len(queuedict) == 0:
                                    return assigns
                                if insertnum == 0:
                                    break

        queuedict_ = copy.deepcopy(queuedict)
        if len(list(queuedict_.values())) > 0:
            for keyf in set(torch.stack(list(queuedict_.values())).tolist()):
                bloss = torch.tensor(count_shards) - n_/shards_number
                sortedb, indicesb = torch.sort(bloss)
                for indb, keyb in enumerate(indicesb):
                    if sortedb[indb] < 0:
                        floss = torch.tensor(
                            df.values[:, keyb]/count_shards[keyb])-balancemean
                        insertnum = (
                            floss[keyf]*count_shards[keyb]).abs().ceil()
                        for _ in range(int(insertnum)):
                            for keyq, valueq in queuedict_.items():
                                if valueq == keyf:
                                    tmp = queuedict.pop(int(keyq), None)
                                    if tmp != None:
                                        assigns[int(self.degree0index[keyq])] = int(
                                            keyb)
                                        insertnum -= 1
                                        if len(queuedict) == 0:
                                            return assigns
                                        if insertnum == 0:
                                            break

        return assigns

    def partition_stat(self, partitions):
        if self.edge_indexs.max() < self.labels.shape[0]:
            edge_let = torch.cat(
                [torch.tensor([[self.labels.shape[0]]]), torch.tensor([[self.labels.shape[0]]])], dim=0).to(self.edge_indexs.device)
            self.edge_indexs = torch.cat((self.edge_indexs, edge_let), 1)

        parts_key = set(partitions.values())
        parts_nodes_id = {}
        parts_edges_index = {}
        for part_id in parts_key:
            parts_nodes_id[part_id] = [key for key,
                                       value in partitions.items() if value == part_id]
            edge_index_, _ = subgraph(
                parts_nodes_id[part_id], self.edge_indexs, relabel_nodes=True)
            parts_edges_index[part_id] = edge_index_

        spicymat_csr = sp.csr_matrix(to_scipy_sparse_matrix(self.edge_indexs))
        # gather neighbors pair between different shards
        inter_pair = {}
        for part_id in parts_key:
            deselect = np.delete(
                range(self.labels.shape[0]), parts_nodes_id[part_id]).tolist()
            despar = spicymat_csr[parts_nodes_id[part_id]][:, deselect]
            inter_list_ = list(
                zip(sp.coo_matrix(despar).row, sp.coo_matrix(despar).col))
            inter_pair_ = defaultdict(list)
            for k, v in inter_list_:
                inter_pair_[k].append(deselect[v])
            inter_pair[part_id] = dict(inter_pair_)

        score_balance, score_fair = self.calculate_score(partitions)

        partition_proc = Partition(
            self.method, partitions, parts_nodes_id, parts_edges_index, score_balance, score_fair, inter_pair)

        return partition_proc

    def calculate_score(self, partitions, labels=None):
        parts_key = set(partitions.values())
        # n = dataset[0].num_nodes
        if labels == None:
            labels = self.labels
        n = len(labels)
        parts_number = len(parts_key)
        count_shards = [op.countOf(partitions.values(), key)
                        for key in parts_key]
        score_balance_ = torch.abs(
            (torch.tensor(count_shards) - n/parts_number)).sum().detach().cpu().numpy()/(2*n)

        label_ = dict(
            zip(range(labels.shape[0]), labels.detach().cpu().numpy()))
        partitions_ = [label_, partitions]
        partitionscomb = {}
        for k in label_.keys():
            partitionscomb[k] = tuple(shard[k] for shard in partitions_)

        value_list = list(partitionscomb.values())
        df = pd.DataFrame(value_list, columns=['True Type', 'Assigned Shard'])
        df = df.groupby(['True Type', 'Assigned Shard']
                        ).size().unstack(fill_value=0)

        odval = labels.bincount()
        balancemean = odval/n

        subbalance = []
        for i in parts_key:
            subbalance.append(
                (torch.tensor(df.values[:, i]/count_shards[i])-balancemean).abs().sum())

        score_fair_ = torch.tensor(
            subbalance).mean().detach().cpu().numpy()/2
        return score_balance_, score_fair_

    def subgraph_repair(self, x, REPAIR_METHOD='Zero', PATH='checkpoints/', DATA_NAME='Cora_0.8', MULTI_GRAPH=None):
        '''
        Repair the subgraphs with selected method, and save the repaired subgraphs to the PATH
        PATH: PATH+DATA_NAME/part{part_id}/subgraphs/part{part_id}_{PARTITION_METHOD}_partition_{REPAIR_METHOD}_repair.pkl
        REPAIR_METHOD: Zero, Mirror, MixUp, None
        '''
        self.x = x
        suffix = self.method + '_Partition_'+REPAIR_METHOD+'_repair'
        for parts in self.p.shards_ids.keys():
            sub_graph = Data(x=self.x[self.p.shards_ids[parts]],
                             edge_index=self.p.shards_edges[parts], y=self.labels[self.p.shards_ids[parts]])
            base_id = list(range(sub_graph.num_nodes))
            from_ = []
            for key in self.p.inter_pair[parts].keys():
                for value in self.p.inter_pair[parts][key]:
                    from_.append(key)

            if len(from_) == 0:
                sub_graph.train_mask = torch.ones(
                    sub_graph.num_nodes, dtype=torch.bool)
                sub_graph.train_mask = torch.ones(
                    sub_graph.num_nodes, dtype=torch.bool)
                sub_graph.uids = base_id

            else:
                if REPAIR_METHOD == 'Zero':
                    # Zero Feature
                    x_plus = torch.zeros([len(from_), sub_graph.num_features])
                    sub_graph.train_mask = torch.ones(
                        sub_graph.num_nodes, dtype=torch.bool)
                    sub_graph.test_mask = torch.zeros(
                        sub_graph.num_nodes, dtype=torch.bool)
                    subplus_graph = self.patch_withouty(
                        sub_graph, x_plus, from_)
                    subplus_graph.uids = base_id + from_
                elif REPAIR_METHOD == 'Mirror':
                    # Mirror Feature
                    x_plus = sub_graph.x[from_]
                    sub_graph.train_mask = torch.ones(
                        sub_graph.num_nodes, dtype=torch.bool)
                    sub_graph.test_mask = torch.zeros(
                        sub_graph.num_nodes, dtype=torch.bool)
                    subplus_graph = self.patch_withouty(
                        sub_graph, x_plus, from_)
                    subplus_graph.uids = base_id + from_
                elif REPAIR_METHOD == 'MixUp':
                    # Mixup Feature
                    mix0up = torch.zeros([len(from_), sub_graph.num_features])
                    alpha = torch.randn(len(from_), 1).uniform_(0, 1)
                    x_plus = alpha * sub_graph.x[from_] + (1-alpha) * mix0up
                    sub_graph.train_mask = torch.ones(
                        sub_graph.num_nodes, dtype=torch.bool)
                    sub_graph.test_mask = torch.zeros(
                        sub_graph.num_nodes, dtype=torch.bool)
                    subplus_graph = self.patch_withouty(
                        sub_graph, x_plus, from_)
                    subplus_graph.uids = base_id + from_
                elif REPAIR_METHOD == 'None':
                    sub_graph.train_mask = torch.ones(
                        sub_graph.num_nodes, dtype=torch.bool)
                    sub_graph.train_mask = torch.ones(
                        sub_graph.num_nodes, dtype=torch.bool)
                    subplus_graph = sub_graph
                    subplus_graph.uids = base_id                

            if MULTI_GRAPH is not None:
                savename = PATH+'{}/part{}/subgraphs/graph{}_part{}_{}.pt'.format(
                    DATA_NAME, parts, MULTI_GRAPH, parts, suffix)
            else:
                savename = PATH+'{}/part{}/subgraphs/part{}_{}.pt'.format(
                    DATA_NAME, parts, parts, suffix)

            os.makedirs(os.path.dirname(savename), exist_ok=True)
            torch.save(subplus_graph, savename)

        if MULTI_GRAPH is not None:
            self.p.DPATH = PATH+'{}/part{}/subgraphs/graph{}_part{}_{}.pt'.format(
                DATA_NAME, 'id', MULTI_GRAPH, 'id', suffix)
        else:
            self.p.DPATH = PATH+'{}/part{}/subgraphs/part{}_{}.pt'.format(
                DATA_NAME, 'id', 'id', suffix)

        return self.p

    def patch_withouty(self, sub_graph, x_trg, from_):
        x = torch.cat((sub_graph.x, x_trg), 0)
        to_repair = list(np.asarray(torch.arange(
            len(from_))) + sub_graph.num_nodes)
        edge_let = torch.cat(
            [torch.tensor([from_, to_repair]), torch.tensor([to_repair, from_])], dim=1).to(sub_graph.x.device)
        edge_index = torch.cat((sub_graph.edge_index, edge_let), 1)
        patch_graph = Data(x=x, edge_index=edge_index, y=sub_graph.y)
        patch_graph.train_mask = torch.cat([sub_graph.train_mask, torch.zeros(
            len(from_), dtype=torch.bool)])

        return patch_graph
