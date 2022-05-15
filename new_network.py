
#2022.05.15 --> 시-공간 반영한 GCN 모델 구조

# Input
# 1. Node : [ batchsize, Node_number, Node_attribute Number, timestep]
# 2. Edge_Index [2,Node_num * (Node_num-1) ] -- fully connected
# 3. 현재 Edge Attribute 는 모델 내부에서 계산함.
# 4. 정답 라벨값 : time step이 12면 [ 0 ~ 12 시나리오에서의 맨 마지막 시나리오의 정답을 사용해야 할 듯 ]


from typing import Optional, Tuple
import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn
import torch
import torch.nn.functional as F

# Message Passing

def build_mlp(dim_list, activation='relu', do_bn=False,
              dropout=0, on_last=False):
    layers = []
    for i in range(len(dim_list) - 1):
        dim_in, dim_out = dim_list[i], dim_list[i + 1]
        layers.append(torch.nn.Linear(dim_in, dim_out))
        final_layer = (i == len(dim_list) - 2)
        if not final_layer or on_last:
            if do_bn:
                layers.append(torch.nn.LayerNorm(dim_out)) #batchNorm1d
            if activation == 'relu':
                layers.append(torch.nn.ReLU())
            elif activation == 'leakyrelu':
                layers.append(torch.nn.LeakyReLU())
        if dropout > 0:
            layers.append(torch.nn.Dropout(p=dropout))
    return torch.nn.Sequential(*layers)

from torch_scatter import scatter

class TripletGCN(MessagePassing):  # Message Passing
    def __init__(self, dim_node, dim_edge, dim_hidden, aggr='add', use_bn=True):  # 2,
        #         self.conv_h = TripletGCN(dim_node=self.in_channels,dim_edge=self.in_channels,dim_hidden=self.in_channels)
        super().__init__(aggr=aggr)
        self.dim_node = dim_node  # 256
        self.dim_edge = dim_edge
        self.dim_hidden = dim_hidden  # 512
        self.nn1 = build_mlp([dim_node * 2 + dim_edge, dim_hidden, dim_hidden * 2 + dim_edge],
                             do_bn=use_bn, on_last=True)
        print(self.nn1)
        # dim_node *2 + dim_edge 이란?
        # source Node 속성 개수 + Target Node 속성 개수 + edgeFeature 속성 개수

        self.nn2 = build_mlp([dim_hidden, dim_hidden, dim_hidden], do_bn=use_bn)
                                                        #최종적으로는 원래 노드의 개수가 나와야됌
        self.nn3 = build_mlp([dim_edge, dim_hidden, dim_hidden], do_bn=use_bn)


    def forward(self, x, edge_feature, edge_index):
        # print(' TripletGCN의 forward 함수 시작 ')
        # print('노드 관련 :',x.size())
        # print('edge_feature :',edge_feature.size())
        # print('edge_index :',edge_index.size())
        gcn_x, gcn_e = self.propagate(edge_index, x=x, edge_feature=edge_feature)  # 메시지 전파 시작을 위한 초기 호출
        # print('propagate 후')
        x = self.nn2(gcn_x)  ### mlp 2단계 : 특정 노드의 경우 집합 단계에서 해당 노드(sub, obj)의 모든 가능한 연결에서 오는 신호들은 함께 평균화됨
        gcn_e = self.nn3(gcn_e)
        #2022.02.15
        # gcn feature 에서
        # gcn_x 는 Node 분류에 사용할 때 사용하는거고
        # gcn_e 는 이제 edge graph context feature 인데 이걸로
        # 공간관계 분류기, hand 관계분류기, hand_obj 관계 분류기로 들어가 output을 출력하도록 하자!

        # print('nn2 후 최종 return')
        print('gcn 후 MLP에 들어간 후 노드들 = ',x.size())
        print('gcn 후 Edge Feature 들 = ',gcn_e.size())

        return x, gcn_e

    def message(self, x_i, x_j, edge_feature):  # 노드에 대한 메시지 구성
        # print('--- message 패싱 함수 시작 ---')
        # print('source Node 사이즈 : ',x_i.size())
        # print('target Node 사이즈 : ',x_j.size())
        # print('Edge Feature 사이즈 : ',edge_feature.size())
        x = torch.cat([x_i, edge_feature, x_j], dim=2)
        print('concat x : ',x.size())
        print('self.nn1 : ',self.nn1)
        # x = self.nn1(x.squeeze(dim=0))  # .view(b,-1) ### mlp 1단계 : triplet ij 정보 전달을 위해 mlp에 제공됨
        x = self.nn1(x)
        print('첫번째 nn1 후 : ',x.size())
        new_x_i = x[:,:, :self.dim_hidden] #전체 18개의 edge_index 관계들 중에 0~128개까지의 가중치만 사용
        new_e = x[:,:, self.dim_hidden:(self.dim_hidden + self.dim_edge)] #이건 128 부터 168까지 사용 -> 딱 edge_feature의 개수만큼
        new_x_j = x[:,:, (self.dim_hidden + self.dim_edge):] #이건 168 부터 다 사용 -> 그럼 이것도 총 128개임
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


class TGCN2(torch.nn.Module):
    r"""An implementation THAT SUPPORTS BATCHES of the Temporal Graph Convolutional Gated Recurrent Cell.
    For details see this paper: `"T-GCN: A Temporal Graph ConvolutionalNetwork for
    Traffic Prediction." <https://arxiv.org/abs/1811.05320>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        batch_size (int): Size of the batch.
        improved (bool): Stronger self loops. Default is False.
        cached (bool): Caching the message weights. Default is False.
        add_self_loops (bool): Adding self-loops for smoothing. Default is True.
    """

    def __init__(self, in_channels: int, out_channels: int, batch_size: int, improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True):
        super(TGCN2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.batch_size = batch_size
        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):
        self.conv_z = TripletGCN(dim_node=self.in_channels,dim_edge=self.in_channels,dim_hidden=self.out_channels)

        # self.conv_z = GCNConv(in_channels=self.in_channels,  out_channels=self.out_channels, improved=self.improved,
        #                       cached=self.cached, add_self_loops=self.add_self_loops )

        self.linear_z_node = torch.nn.Linear(2 * self.out_channels, self.out_channels)
        self.linear_z_edge = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_r = TripletGCN(dim_node=self.in_channels,dim_edge=self.in_channels,dim_hidden=self.out_channels)

        # self.conv_r = GCNConv(in_channels=self.in_channels, out_channels=self.out_channels, improved=self.improved,
        #                       cached=self.cached, add_self_loops=self.add_self_loops )
        self.linear_r_node = torch.nn.Linear(2 * self.out_channels, self.out_channels)
        self.linear_r_edge = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_h = TripletGCN(dim_node=self.in_channels,dim_edge=self.in_channels,dim_hidden=self.out_channels)

        # self.conv_h = GCNConv(in_channels=self.in_channels, out_channels=self.out_channels, improved=self.improved,
        #                       cached=self.cached, add_self_loops=self.add_self_loops )
        self.linear_h_node = torch.nn.Linear(2 * self.out_channels, self.out_channels)
        self.linear_h_edge = torch.nn.Linear(2 * self.out_channels, self.out_channels)
    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X,edge_feature, H):
        H_node = torch.zeros(self.batch_size, X.shape[1], self.out_channels)  # (b, 207, 32)
        H_edge = torch.zeros(self.batch_size, edge_feature.shape[1], self.out_channels)
        # if H is None:
        #     H_node = torch.zeros(self.batch_size,X.shape[1], self.out_channels) #(b, 207, 32)
        #     H_edge = torch.zeros(self.batch_size,edge_feature.shape[1], self.out_channels)
        return H_node,H_edge
    #     def forward(self, x, edge_feature, edge_index):
    def _calculate_update_gate(self, X, edge_index, edge_feature, H_Node,H_Edge):
        gcn_node,gcn_edge =self.conv_z(X, edge_feature, edge_index)

        Z_node = torch.cat([gcn_node,H_Node],axis=2)
        Z_edge = torch.cat([gcn_edge,H_Edge],axis=2)

        Z_node = self.linear_z_node(Z_node) # (b, 4, 32)
        Z_edge = self.linear_z_edge(Z_edge)  #(b,12,32)
        Z_node = torch.sigmoid(Z_node)
        Z_edge = torch.sigmoid(Z_edge)

        return Z_node,Z_edge

    def _calculate_reset_gate(self, X, edge_index, edge_feature, H_Node,H_Edge):
        gcn_node,gcn_edge =self.conv_r(X, edge_feature, edge_index)

        R_node = torch.cat([gcn_node, H_Node], axis=2)
        R_edge = torch.cat([gcn_edge, H_Edge], axis=2)

        R_node = self.linear_r_node(R_node) # (b, 207, 32)
        R_edge = self.linear_r_edge(R_edge)
        R_node = torch.sigmoid(R_node)
        R_edge = torch.sigmoid(R_edge)

        return R_node,R_edge

    def _calculate_candidate_state(self, X, edge_index, edge_feature, H_Node,H_Edge, R_node,R_edge):
        gcn_node,gcn_edge = self.conv_h(X, edge_feature, edge_index)
        H_tilde_node = torch.cat([gcn_node,H_Node*R_node],axis=2)
        H_tilde_edge = torch.cat([gcn_edge, H_Edge * R_edge], axis=2)

        H_tilde_node = self.linear_h_node(H_tilde_node) # (b, 4, 32)
        H_tilde_edge = self.linear_h_edge(H_tilde_edge)  # (b, 12, 32)

        H_tilde_node = torch.tanh(H_tilde_node)
        H_tilde_edge = torch.tanh(H_tilde_edge)

        return H_tilde_node,H_tilde_edge

    def _calculate_hidden_state(self, Z_Node,Z_Edge, H_Node,H_Edge, H_tilde_Node,H_tilde_Edge):
        H_node = Z_Node * H_Node + (1 - Z_Node) * H_tilde_Node   # # (b, 207, 32)
        H_edge = Z_Edge * H_Edge + (1 - Z_Edge) * H_tilde_Edge  # # (b, 207, 32)

        return H_node,H_edge

    def forward(self,X: torch.FloatTensor, edge_index: torch.LongTensor, edge_feature: torch.FloatTensor,
                H: torch.FloatTensor = None ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
        """
        H_Node,H_Edge = self._set_hidden_state(X,edge_feature,H)
        print('H_node = ',H_Node.size())
        print('H_Edge = ',H_Edge.size())
        Z_Node,Z_Edge = self._calculate_update_gate(X, edge_index, edge_feature, H_Node,H_Edge)
        R_Node,R_Edge = self._calculate_reset_gate(X, edge_index, edge_feature, H_Node,H_Edge)
        H_tilde_Node,H_tilde_Edge = self._calculate_candidate_state(X, edge_index, edge_feature,H_Node,H_Edge, R_Node,R_Edge)
        H_node,H_edge = self._calculate_hidden_state(Z_Node,Z_Edge, H_Node,H_Edge, H_tilde_Node,H_tilde_Edge) # (b, 207, 32)
        return (H_node,H_edge)



class A3TGCN2(torch.nn.Module):
    r"""An implementation THAT SUPPORTS BATCHES of the Attention Temporal Graph Convolutional Cell.
    For details see this paper: `"A3T-GCN: Attention Temporal Graph Convolutional
    Network for Traffic Forecasting." <https://arxiv.org/abs/2006.11583>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        periods (int): Number of time periods.
        improved (bool): Stronger self loops (default :obj:`False`).
        cached (bool): Caching the message weights (default :obj:`False`).
        add_self_loops (bool): Adding self-loops for smoothing (default :obj:`True`).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        periods: int,
        batch_size:int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True):
        super(A3TGCN2, self).__init__()

        self.in_channels = in_channels  # 2
        self.out_channels = out_channels # 32
        self.periods = periods # 12
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.batch_size = batch_size
        self._setup_layers()

    def _setup_layers(self):
        self._base_tgcn = TGCN2(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=self.batch_size,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops)

        device = torch.device('cpu')
        self._attention = torch.nn.Parameter(torch.empty(self.periods, device=device))
        torch.nn.init.uniform_(self._attention)

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_feature: torch.FloatTensor,
        H: torch.FloatTensor = None
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** (PyTorch Float Tensor): Node features for T time periods.
            * **edge_index** (PyTorch Long Tensor): Graph edge indices.
            * **edge_weight** (PyTorch Long Tensor, optional)*: Edge weight vector.
            * **H** (PyTorch Float Tensor, optional): Hidden state matrix for all nodes.

        Return types:
            * **H** (PyTorch Float Tensor): Hidden state matrix for all nodes.
        """
        H_nodeMatrix = 0
        H_edgeMatrix = 0
        probs = torch.nn.functional.softmax(self._attention, dim=0)

        for period in range(self.periods):

            H_nodeMatrix = H_nodeMatrix + probs[period] * self._base_tgcn( X[:, :, :, period], edge_index, edge_feature[:,:,:,period], H)[0] #([32, 207, 32]
            H_edgeMatrix = H_edgeMatrix + probs[period] * self._base_tgcn(X[:, :, :, period], edge_index, edge_feature[:,:,:,period],H)[1]
        return H_nodeMatrix,H_edgeMatrix


# Making the model
class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods, batch_size,out_channel):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN2(in_channels=node_features, out_channels=out_channel, periods=periods,
                            batch_size=batch_size)  # node_features=2, periods=12
        print(self.tgnn)
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(32, periods)

    def forward(self, x, edge_index,edge_feature):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        import sys
        h = self.tgnn(x, edge_index,edge_feature)  # x [b, 207, 2, 12]  returns h [b, 207, 12]
        print('tgnn --> h size', h[0].size(),h[1].size())
        #이 다음에 Node 분류, Edge 분류기의 입력으로 들어가야함.
        sys.exit()
        h = F.relu(h)
        h = self.linear(h)
        return h


class test_model(nn.Module):
    def __init__(self,node_in,out_channel,rel_in):
        super().__init__()
        self.Node_embedding = nn.Linear(node_in,out_channel)
        self.Edge_embedding = nn.Linear(rel_in,out_channel)
    def forward(self,X,pair_X):
        #change timestep
        X = X.permute(2,0,1)
        pair_X = pair_X.permute(2,0,1)

        node_embed = self.Node_embedding(X)
        edge_embed = self.Edge_embedding(pair_X)

        return node_embed.permute(1,2,0),edge_embed.permute(1,2,0)

# Create model and optimizers
if __name__=='__main__':

    num_nodefeature = 6
    timestep = 18
    node_num = 4
    rel_attr_num = 3
    node_feat = 32
    out_feature_num = 128
    batcsize = 1
    DataX = torch.randn((node_num,num_nodefeature,timestep))
    DataEdge = torch.randn((node_num*(node_num-1),rel_attr_num,timestep))
    Data_EdgeIndex = torch.randint(0,4,(2,12))
    print(Data_EdgeIndex)

    #Node Embedding , Edge Embedding
    model = test_model(node_in=num_nodefeature,out_channel=node_feat,rel_in=rel_attr_num)

    #Temporal - Messagepassing
    gcn_model = TemporalGNN(node_features=node_feat, periods=timestep, batch_size=batcsize,out_channel=out_feature_num)

    nodeFeature,edgeFeature = model(DataX,DataEdge)
    print('nodeFeature  = ',nodeFeature.size())
    print('edgeFeature = ',edgeFeature.size())
    print(nodeFeature.unsqueeze(dim=0).size())
    #Input size - Node [batchsize,node_num,node_feature_num,time_step]
    # Input size - Edge [batchsize,edge_num,edge_feature_num,time_step]

    print(gcn_model(nodeFeature.unsqueeze(dim=0),Data_EdgeIndex,edgeFeature.unsqueeze(dim=0)))
