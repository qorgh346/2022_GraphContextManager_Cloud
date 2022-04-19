import os,sys

import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy
#임시 임베딩 value
temp_embedding_vector = {'Move':0,'Ready':1,'init_Load':2,'init_load':2,'none':3}
class MosDataset(Dataset):
    def __init__(self,root,split_path,mode='train',Normalization=True):
        ### 22. 04.19 ###
        # Normalization code
        # MinMaxScaler 객체 생성 --> 정규화 작업 사용
        self.norm_flag = Normalization


        #### 22.04.17 ####
        # train_test split code #
        data_list_file = os.path.join(split_path,'{}_list.txt'.format(mode))

        list_file = open(data_list_file, "r")
        split_data_list = [i.rstrip() for i in list_file.readlines()]
        list_file.close()


        self.data_lists = [os.path.join(root,i) for i in split_data_list]
        print(self.data_lists)
        ###
        self.robot_mappint_idx = dict()
        # -> {'AMR_LIFT1': 0, 'AMR_LIFT2': 1, 'AMR_TOW1': 2, 'AMR_TOW2': 3}
        self.edge_mapping_idx = dict()
        # -> #{'0_1': 0, '0_2': 1, '0_3': 2, '1_0': 3, '1_2': 4, '1_3': 5, '2_0': 6, '2_1': 7, '2_3': 8, '3_0': 9, '3_1': 10, '3_2': 11}
        self.relation_list = ['nearBy','isBehindOf','faceToFace']
        self.relation_mapping_idx = {v:k for k,v in enumerate(self.relation_list)}
        #-> {#nearBy : 0 ,isBehindOf : 1 .... }
    def __getitem__(self, idx):
        data_path = self.data_lists[idx]
        with open(data_path, "r") as d_json:
            data = json.load(d_json)
        #Node
        node_data = self.getNode_info(data['robots'])
        #Get Edge Index
        edge_index = self.getEdgeIndex(data['robots'])
        #GT
        gt_label = self.getGT_label(edge_index.shape[1],data['labels'])

        node_data , edge_index, gt_label = torch.from_numpy(node_data).float(),torch.from_numpy(edge_index).long(),\
                                           torch.from_numpy(gt_label).long()

        meta_data = {'GT':gt_label,'robot_mappint_idx':self.robot_mappint_idx,'edge_mapping_idx':self.edge_mapping_idx,
                     'relation_mapping_idx':self.relation_mapping_idx}
        return {'x':node_data,'edge_index':edge_index,'meta':meta_data}

    def __len__(self):
        return len(self.data_lists)

    def getNode_info(self,data):
        node_data = np.array([])
        for robot_info in data:
            xyz_list = robot_info['pos']  # type : list
            xyz_list.append(robot_info['velocity'])
            xyz_list.append(robot_info['battery'])
            xyz_list.append(temp_embedding_vector[robot_info['status']])
            node_data = np.concatenate([node_data, np.array(xyz_list)])
        node_data = node_data.reshape(len(data), -1)
        return node_data

    def getGT_label(self,row_size,data):

        gt_label = np.zeros([row_size, len(self.relation_list)])
        for rel in data:
            subject = rel[0]  # "AMR_LIFT2" -->row
            object = rel[2]  # AMR_TOW1" -->row
            relation = rel[1]  # nearby --> col

            row_num = "{}_{}".format(self.robot_mappint_idx[subject], self.robot_mappint_idx[object])
            col_num = self.relation_mapping_idx[relation]
            gt_label[self.edge_mapping_idx[row_num], col_num] = 1
        return gt_label
    def getEdgeIndex(self,data):
        #edge_index - fully connected
        edge_index = list()
        temp_idx = 0
        for i in range(len(data)):
            for j in range(len(data)):
                if i == j: continue
                edge_index.append([i,j])
                self.robot_mappint_idx[data[i]['name']] = i
                # --> {'AMR_LIFT1': 0, 'AMR_LIFT2': 1, 'AMR_TOW1': 2, 'AMR_TOW2': 3}
                self.edge_mapping_idx["{}_{}".format(i,j)] = temp_idx

                temp_idx += 1
        edge_index = np.array(edge_index).T #size = [2,num_node*num_node-1 ]
        return edge_index



if __name__ == '__main__':
    path = '../../mos_datasets_jsons'
    train_test_path = '../split_dataset_list'
    a = MosDataset(root=path,split_path=train_test_path,mode='test')
    b = DataLoader(dataset=a,batch_size=1)
    for i in b:
        print(i)
        # sys.exit()
