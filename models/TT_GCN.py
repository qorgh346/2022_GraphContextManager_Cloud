#TT GCN : Temporal TripleNet Graph Convolution Network
import torch
import sys
sys.path.append("..")
from models.TripleNetGCN import TripletGCN

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
        # print('H_node = ',H_Node.size())
        # print('H_Edge = ',H_Edge.size())
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
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(32, periods)

    def forward(self, x, edge_index,edge_feature):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        import sys
        h = self.tgnn(x, edge_index,edge_feature)  # x [b, 207, 2, 12]  returns h [b, 207, 12]
        return h[0],h[1]
