from TripleNetGCN import TripletGCNModel
from network_RelNet import RelNetFeat, RelNetCls
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

import sys
import os
sys.path.append("..")
import op_utils
from Datasets.MosDatasets import MosDataset
import numpy as np


class GCMModel(nn.Module):
    def __init__(self, name:str,hy_param):
        super(GCMModel, self).__init__()
        self.name = name
        self.relation_info_num = hy_param['dim_rel']
        # Build model
        models = dict()
        #각 노드마다 속성의 개수가 다 다를 수 있기 때문에 encoding을 통해 차원을 맞춰줌
        models['obj_Node_encoder'] = nn.Sequential(nn.Linear(hy_param['dim_obj'],16),
                                                    nn.ReLU(True),
                                                    nn.Linear(16,hy_param['dim_node'])
                                                    )
        models['rel_encoder'] = nn.Sequential(nn.Linear(hy_param['dim_rel'],16),
                                               nn.ReLU(True),
                                               nn.Linear(16,hy_param['dim_edge'])
                                               )
        # Triplet GCN
        models['triplet_gcn'] = TripletGCNModel(
            num_layers=hy_param['num_layer'],
            dim_node=hy_param['dim_node'],
            dim_edge=hy_param['dim_edge'],
            dim_hidden=hy_param['gcn_dim_hidden']
        )
        #classification module
        models['node_cls'] = nn.Sequential(nn.Linear(hy_param['gcn_dim_hidden'],16),
                                           nn.ReLU(True),
                                           nn.Linear(16,hy_param['num_node']),
                                           nn.Softmax(dim=1)
                                           )

        models['rel_cls'] = nn.Sequential(nn.Linear(hy_param['gcn_dim_hidden'], 32),
                                           nn.ReLU(True),
                                           nn.Linear(32, hy_param['rel_num']),
                                           nn.Softmax(dim=1)
                                           )
        # --------------------------------
        params = list()
        print('==trainable parameters==')
        for name, model in models.items():
            self.add_module(name, model)
            params += list(model.parameters())
            print(name,op_utils.pytorch_count_params(model))
        print('')
        self.optimizer = optim.Adam(params=params, lr=hy_param['lr'],)
        self.optimizer.zero_grad()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, node_feature, rel_feature, edges_index):

        obj_feature = self.obj_Node_encoder(node_feature)

        edge_feature = self.rel_encoder(rel_feature)  # relationship 피쳐 추출
        gcn_obj_feature, gcn_rel_feature = self.triplet_gcn(obj_feature, edge_feature, edges_index)

        pred_node = self.node_cls(gcn_obj_feature)
        pred_rel = self.rel_cls(gcn_rel_feature)

        predict_value ={
            'pred_node' : pred_node,
            'pred_rel' : pred_rel
        }
        return obj_feature, edge_feature, gcn_rel_feature,gcn_obj_feature,predict_value

    def getRelFeat(self,node_feature,edge_index):
        temp_relative_info = torch.zeros((edge_index.size(1), self.relation_info_num))
        for i in range(edge_index.size(1)):
            subject_idx = edge_index[0][i]
            object_idx = edge_index[1][i]
            sub_pos = node_feature[subject_idx][:2]
            obj_pos = node_feature[object_idx][:2]
            relative_distance = math.sqrt(
                (obj_pos[0] - sub_pos[0]) ** 2 + (obj_pos[1] - sub_pos[1]) ** 2)  # 두 점 사이 거리 계산
            rad = math.atan2(obj_pos[1] - sub_pos[1], obj_pos[0] - sub_pos[0])  # 두 점 사이 각 계산
            relative_degree = (rad * 180) * math.pi
            relative_orientation = (x[object_idx][2] - x[subject_idx][2])
            temp_relative_info[i] = torch.tensor([relative_distance, relative_degree, relative_orientation])

        return temp_relative_info

    def process(self,mode,node_feature,edges_index,gt_value, weights_obj=None, weights_rel=None):

        #relation_feature 계산
        relation_feature = self.getRelFeat(node_feature,edges_index) #12,3

        obj_feature, edge_feature, gcn_rel_feature, gcn_obj_feature,predict_value = self(node_feature, relation_feature, edges_index)

        #gt : [12,3] -- predict : [12,3]
        rel_loss = F.binary_cross_entropy(predict_value['pred_rel'],gt_value.type(torch.FloatTensor))
        # print(predict_value['pred_rel'])
        # print(rel_loss)
        if mode =='train':
            self.backward(rel_loss)
        logs = ("Loss/total_loss", rel_loss.detach().item())
        return logs,predict_value

    def backward(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()



if __name__ == '__main__':

    hy_param = {}
    hy_param['num_layer'] = 2
    hy_param['dim_node'] = 32
    hy_param['dim_obj'] = 6
    hy_param['dim_rel'] = 3
    hy_param['dim_edge'] = 32
    hy_param['gcn_dim_hidden'] = 32
    hy_param['rel_num'] = 3
    hy_param['lr'] = 0.0001
    #num_node

    hy_param['num_node'] = 4
    # build model
    network = GCMModel('GCMModel', hy_param)

    train_datasets = MosDataset('../../mos_train_jsons')
    from torch.utils.data import DataLoader
    import math
    trainDataLoader = DataLoader(dataset=train_datasets, batch_size=1, shuffle=True)

    for i in range(0,100):
        for item in trainDataLoader:
            x = item['x'].squeeze(dim=0)
            edge_index = item['edge_index'].squeeze(dim=0)
            gt_label = item['meta']['GT'].squeeze(dim=0)
            logs, predict_value = network.process('train', x, edge_index, gt_label)
            print(logs)
            # print(predict_value)

