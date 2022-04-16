from torch import nn
import torch

class RelNetFeat(nn.Module):
    def __init__(self, input_size, output_size, batch_norm=True, init_weights=True):
        super(RelNetFeat, self).__init__()
        self.name = 'RelFeature'
        self.use_batch_norm = batch_norm
        self.input_size = input_size
        self.out_size = output_size

        self.fc1 = nn.Linear(self.input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, output_size)

        if batch_norm:
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.r = nn.Tanh()
        # if init_weights:
        #     self.init_weights('constant', 1, target_op = 'BatchNorm')
        #     self.init_weights('xavier_normal', 1)

    def forward(self, rel_feature):
        x = rel_feature
        temp = x
        x = self.fc1(x)
        if self.use_batch_norm:
            self.bn1(x)
        x = self.relu(x)

        x = self.fc2(x)
        if self.use_batch_norm:
            self.bn2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)
        # print(x.size())
        return x

''' 이 부분 추가 '''
class RelNetCls(nn.Module):
    def __init__(self, rel_num=6, in_size=40, batch_norm=True, drop_out=True):
        super().__init__()
        self.name = 'relcls'
        self.in_size = in_size
        self.rel_num = rel_num
        self.use_batch_norm = batch_norm  # true
        self.use_drop_out = drop_out
        self.fc1 = nn.Linear(in_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, rel_num)
        if drop_out:  # true
            self.dropout = nn.Dropout(p=0.3)
        if self.use_batch_norm:  # true
            self.bn1 = nn.BatchNorm1d(128)
            self.bn2 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.fc1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.relu(x)

        x = self.fc2(x)
        if self.use_drop_out:
            x = self.dropout(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.relu(x)

        x = self.fc3(x)

        result = nn.Softmax(dim=1)

        return result(x) #F.log_softmax(x, dim=1)

if __name__ == '__main__':

    num_node = 4
    dim_node = 25
    num_edge = num_node * (num_node - 1)
    input_size = 16
    output_size = 256

    nodes_feat = torch.rand([num_node, dim_node], dtype=torch.float)

    # 입력 rel_feature 랜덤 값
    # rel_feature = torch.rand([num_edge, input_size], dtype=torch.float)

    # node feature로 만든 rel_feature
    rel_feature = list()
    for n in range(num_node):  # class id list
        for m in range(num_node):
            if n == m: continue
            sub_feat = nodes_feat[n]
            obj_feat = nodes_feat[m]
            pair_rel = torch.concat([sub_feat, obj_feat], dim=0)
            rel_feature.append(pair_rel)
    rel_feature = torch.stack(rel_feature, 0)

    input_size = rel_feature[2].size()[0]
    net = RelNetFeat(input_size, output_size)
    edge_feature = net(rel_feature)
    print(edge_feature)
    print(edge_feature.size())