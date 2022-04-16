from typing import Optional
import torch
from torch import Tensor
# from networks_base import BaseNetwork, mySequential
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter
import torch.nn as nn
import torch.nn.functional as F
import sys

def build_mlp(dim_list, activation='relu', do_bn=False,
              dropout=0, on_last=False):
    layers = []
    for i in range(len(dim_list) - 1):
        dim_in, dim_out = dim_list[i], dim_list[i + 1]
        layers.append(torch.nn.Linear(dim_in, dim_out))
        final_layer = (i == len(dim_list) - 2)
        if not final_layer or on_last:
            if do_bn:
                layers.append(torch.nn.BatchNorm1d(dim_out))
            if activation == 'relu':
                layers.append(torch.nn.ReLU())
            elif activation == 'leakyrelu':
                layers.append(torch.nn.LeakyReLU())
        if dropout > 0:
            layers.append(torch.nn.Dropout(p=dropout))
    return torch.nn.Sequential(*layers)


# Message Passing

class TripletGCN(MessagePassing):  # Message Passing
    def __init__(self, dim_node, dim_edge, dim_hidden, aggr='add', use_bn=True):  # 2,
        super().__init__(aggr=aggr)
        self.dim_node = dim_node  # 256
        self.dim_edge = dim_edge
        self.dim_hidden = dim_hidden  # 512
        self.nn1 = build_mlp([dim_node * 2 + dim_edge, dim_hidden, dim_hidden * 2 + dim_edge],
                             do_bn=use_bn, on_last=True)
        print(self.nn1)
        # dim_node *2 + dim_edge 이란?
        # source Node 속성 개수 + Target Node 속성 개수 + edgeFeature 속성 개수

        self.nn2 = build_mlp([dim_hidden, dim_hidden, dim_node], do_bn=use_bn)
                                                        #최종적으로는 원래 노드의 개수가 나와야됌

    def forward(self, x, edge_feature, edge_index):
        # print(' TripletGCN의 forward 함수 시작 ')
        # print('노드 관련 :',x.size())
        # print('edge_feature :',edge_feature.size())
        # print('edge_index :',edge_index.size())
        gcn_x, gcn_e = self.propagate(edge_index, x=x, edge_feature=edge_feature)  # 메시지 전파 시작을 위한 초기 호출
        # print('propagate 후')
        x = self.nn2(gcn_x)  ### mlp 2단계 : 특정 노드의 경우 집합 단계에서 해당 노드(sub, obj)의 모든 가능한 연결에서 오는 신호들은 함께 평균화됨
        #2022.02.15
        # gcn feature 에서
        # gcn_x 는 Node 분류에 사용할 때 사용하는거고
        # gcn_e 는 이제 edge graph context feature 인데 이걸로
        # 공간관계 분류기, hand 관계분류기, hand_obj 관계 분류기로 들어가 output을 출력하도록 하자!

        # print('nn2 후 최종 return')
        # print('gcn 후 MLP에 들어간 후 노드들 = ',x.size())
        # print('gcn 후 Edge Feature 들 = ',gcn_e.size())

        return x, gcn_e

    def message(self, x_i, x_j, edge_feature):  # 노드에 대한 메시지 구성
        # print('--- message 패싱 함수 시작 ---')
        # print('source Node 사이즈 : ',x_i.size())
        # print('target Node 사이즈 : ',x_j.size())
        # print('Edge Feature 사이즈 : ',edge_feature.size())
        x = torch.cat([x_i, edge_feature, x_j], dim=1)
        x = self.nn1(x)  # .view(b,-1) ### mlp 1단계 : triplet ij 정보 전달을 위해 mlp에 제공됨
        # print('첫번째 nn1 후 : ',x.size())
        new_x_i = x[:, :self.dim_hidden] #전체 18개의 edge_index 관계들 중에 0~128개까지의 가중치만 사용
        new_e = x[:, self.dim_hidden:(self.dim_hidden + self.dim_edge)] #이건 128 부터 168까지 사용 -> 딱 edge_feature의 개수만큼
        new_x_j = x[:, (self.dim_hidden + self.dim_edge):] #이건 168 부터 다 사용 -> 그럼 이것도 총 128개임
        x = new_x_i + new_x_j #업데이트 된 source Node의 속성들(128개) + 업데이트 된 Target Node의 속성들(128개) = 요소합이라서
                                                                                            # (edge_index 개수 ,128)임
        # print('Message 패싱 끝 -> return 하면 aggregate 가서 애들 모을거임')
        return [x, new_e]

    def aggregate(self, x: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        # print('어그리게이트 함수 실행 \n aggregate 전 텐서 사이즈 : ',x[0].size())
        #메세지 페싱할 때는 edge_index 에 연결되어 있는 노드들한테 패싱하기 때문에 사이즈는 [노드--노드 연결된 개수 ,128개]
        #원래 노드의 개수 4개로 바꿔야됌. add하냐 max할거냐 avg할거냐의 방법을 써서 원래 노드의 개수 4개로 맞춰줌.아래 dim=원래 노드의 개수 넣으면 됌
        x[0] = scatter(x[0], index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        # print('어그리게이트 끝나고 aggregate 후 텐서 사이즈 : ',x[0].size())
        return x


class TripletGCNModel(nn.Module):
    """ A sequence of scene graph convolution layers  """

    ''' --- rel_num 인수로 받아오도록 추가 --- '''
    def __init__(self, num_layers,**kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.gconvs = torch.nn.ModuleList()
        ''' 이 부분 추가 '''
        # self.rel_cls = relNetCls(rel_num=rel_num, in_size=kwargs['dim_edge'], batch_norm=True, drop_out=True)
        # ''' ---------- '''

        for _ in range(self.num_layers):  # num_layer=2 -> 2홉. 1 ---2 ---3
            self.gconvs.append(TripletGCN(**kwargs))  # dim node=256, dim_edge=256, dim_hidden=512

    def forward(self, node_feature, edge_feature, edges_indices):
        result = {}
        for i in range(self.num_layers):
            # print('@@' * 10)
            # print('{} 번째 홉 시작'.format(i + 1))
            gconv = self.gconvs[i]
            node_feature, edge_feature = gconv(node_feature, edge_feature, edges_indices)
            # print('{} 번째 홉 끝'.format(i+1))
            # print('@@'*10)
            if i < (self.num_layers - 1):
                node_feature = torch.nn.functional.relu(node_feature)
                edge_feature = torch.nn.functional.relu(edge_feature)
                # temp_feature = torch.nn.functional.relu(temp_feature)
            # a = self.rel_cls(edge_feature)
            # print('rel_cls 결과 : ',a)

        # relation_result = self.rel_cls(edge_feature)
        result['node_feature'] = node_feature
        result['edge_feature'] = edge_feature
        # result['relation_result'] = relation_result
        # return result
        return node_feature, edge_feature

''' 이 부분 추가 '''
# class relNetCls(nn.Module):
#     def __init__(self, rel_num=6, in_size=40, batch_norm=True, drop_out=True):
#         super().__init__()
#         self.name = 'relcls'
#         self.in_size = in_size
#         self.rel_num = rel_num
#         self.use_batch_norm = batch_norm  # true
#         self.use_drop_out = drop_out
#         self.fc1 = nn.Linear(in_size, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, rel_num)
#         if drop_out:  # true
#             self.dropout = nn.Dropout(p=0.3)
#         if self.use_batch_norm:  # true
#             self.bn1 = nn.BatchNorm1d(128)
#             self.bn2 = nn.BatchNorm1d(128)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.fc1(x)
#         if self.use_batch_norm:
#             x = self.bn1(x)
#         x = self.relu(x)
#
#         x = self.fc2(x)
#         if self.use_drop_out:
#             x = self.dropout(x)
#         if self.use_batch_norm:
#             x = self.bn2(x)
#         x = self.relu(x)
#
#         x = self.fc3(x)
#
#         result = nn.Softmax(dim=1)
#
#         return result(x) #F.log_softmax(x, dim=1)


if __name__ == '__main__':
    num_layers = 2
    dim_node = 32
    dim_edge = 64
    dim_hidden = 128
    num_node = 6
    num_edge = 4

    heads = 2
    x = torch.rand(4, 32) #[numnode, node_embedding_feature]

    edge_feature = torch.rand([12, 32], dtype=torch.float) #[edge_num,edge_embedding_feature]

    edge_index = torch.randint(0, 4, [12, 2]) #temp_edge_index
    print(edge_index.size())
    edge_index = edge_index.t().contiguous() #[2,12]
    print(edge_index)
    #4,32
    net = TripletGCNModel(2, dim_node=32, dim_edge=32, dim_hidden=32)
    print(net)
    y = net(x, edge_feature, edge_index)
    print(y[0].size())
    print(y[1].size())
    print(y)