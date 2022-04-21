import torch
import torch_geometric
from models.CloudGCM_Network import *
from Datasets.MosDatasets import MosDataset
from torch.utils.data import DataLoader
import copy
def init_param():
    hy_param = {}
    hy_param['num_layer'] = 2
    hy_param['dim_node'] = 32
    hy_param['dim_obj'] = 6
    hy_param['dim_rel'] = 3
    hy_param['dim_edge'] = 32
    hy_param['gcn_dim_hidden'] = 32
    hy_param['rel_num'] = 3
    hy_param['lr'] = 0.01
    # num_node
    hy_param['num_node'] = 4
    hy_param['path'] = '../mos_datasets_jsons'
    hy_param['train_test_path'] = './split_dataset_list'
    hy_param['epochs'] = 100
    return hy_param


hy_param = init_param()
# build model
network = GCMModel('GCMModel', hy_param,norm_flag=True)
print(network)
#build Datasets
train_datasets = MosDataset(root=hy_param['path'], split_path=hy_param['train_test_path'], mode='train',Normalization=True)
test_datasets = MosDataset(root=hy_param['path'], split_path=hy_param['train_test_path'], mode='test',Normalization=True)
val_datasets = MosDataset(root=hy_param['path'], split_path=hy_param['train_test_path'], mode='val',Normalization=True)
#build Dataloader
trainDataLoader = DataLoader(dataset=train_datasets, batch_size=1, shuffle=True)
testDataLoader = DataLoader(dataset=test_datasets, batch_size=1, shuffle=False)
valDataLoader = DataLoader(dataset=val_datasets, batch_size=1, shuffle=True)

print(valDataLoader)

def run_process(mode,model_path='./save_models'):
    if mode == 'train':
        best_acc = 0.0
        for epoch in range(0, hy_param['epochs']):
            running_loss = 0.0
            running_corrects = 0
            num_cnt = 0

            #Train Start
            network.train()
            for item in trainDataLoader:
                x = item['x'].squeeze(dim=0)
                edge_index = item['edge_index'].squeeze(dim=0)
                gt_label = item['meta']['GT'].squeeze(dim=0)
                # print('id : ', item['meta']['id'])

                logs, predict_value = network.process('train', x, edge_index, gt_label)

                running_loss += logs[1]
                running_corrects += logs[3]
                num_cnt += 1
                # print(logs)
                # print(predict_value)
            epoch_train_loss = running_loss / num_cnt
            epoch_train_acc = running_corrects / num_cnt
            # 에폭 loss & acc 계산
            print('{}/{} train(loss) = {:.5f} \t train(acc) = {:.5f}'.format(epoch, hy_param['epochs'], epoch_train_loss,
                                                               epoch_train_acc))

            #Validation Start
            # print('############ Start Val ############## ')
            running_loss = 0.0
            running_corrects = 0
            num_cnt = 0
            for item in valDataLoader:
                with torch.no_grad():
                    network.eval()
                    x = item['x'].squeeze(dim=0)
                    edge_index = item['edge_index'].squeeze(dim=0)
                    gt_label = item['meta']['GT'].squeeze(dim=0)
                    logs, predict_value = network.process('val', x, edge_index, gt_label)
                    running_loss += logs[1]
                    running_corrects += logs[3]
                    num_cnt += 1
            epoch_val_loss = running_loss / num_cnt
            epoch_val_acc = running_corrects / num_cnt


            #에폭 loss & acc 계산
            print('{}/{} val(loss) = {:.5f} \t val(acc) = {:.5f}'.format(epoch, hy_param['epochs'], epoch_val_loss,
                                                               epoch_val_acc))

            if epoch_val_acc > best_acc:
                best_idx = epoch
                best_acc = epoch_val_acc
                best_model_wts = copy.deepcopy(network.state_dict())
                print('==> best model saved - %d / %.1f' % (best_idx, best_acc))
                check_point = torch.save({'epoch': epoch,'model_state_dict': network.state_dict(),
                'loss': epoch_val_loss}, './last_checkPoint')
            if epoch % 10 == 0:
                torch.save(network.state_dict(),
                       '{}/{}'.format('./save_models', 'model_epoch{}.pt'.format(epoch)))
        network.load_state_dict(best_model_wts)

        torch.save(network.state_dict(),
                   '{}/{}'.format('./save_models','bestmodel.pt'))
        print('model saved')


    else:
        # Test Start
        #load_model
        model_file_name = 'bestmodel.pt' #추후에 best model로 변경
        network.load_state_dict(torch.load(os.path.join(model_path,model_file_name)))
        # print(network)
        #
        running_loss = 0.0
        running_corrects = 0
        num_cnt = 0

        for item in testDataLoader:
            print('Start')
            with torch.no_grad():
                network.eval()
                x = item['x'].squeeze(dim=0)
                print('id : ', item['meta']['id'])


                edge_index = item['edge_index'].squeeze(dim=0)
                gt_label = item['meta']['GT'].squeeze(dim=0)
                logs, predict_value = network.process('test', x, edge_index, gt_label)
                print(logs)
                running_loss += logs[1]
                running_corrects += logs[3]
                num_cnt += 1

                relation_mapping_idx ={ v.item():k for k,v in item['meta']['relation_mapping_idx'].items()}
                edge_mapping_idx = {v.item(): k for k, v in item['meta']['edge_mapping_idx'].items()}
                robot_mappint_idx = {v.item(): k for k, v in item['meta']['robot_mappint_idx'].items()}
                # print(robot_mappint_idx)
                # print(relation_mapping_idx)
                # print(predict_value['pred_rel'])
                # print(edge_mapping_idx)

                max_pred_argmax = torch.argmax(predict_value['pred_rel'], dim=1)
                for idx,result in enumerate(max_pred_argmax):
                    sub_obj = edge_mapping_idx[idx].split('_')
                    sub = robot_mappint_idx[int(sub_obj[0])]
                    obj =robot_mappint_idx[int(sub_obj[1])]
                    predicate = relation_mapping_idx[result.item()]
                    print('{}_{}_{}'.format(sub,predicate,obj))

        test_loss = running_loss / num_cnt
        test_acc = running_corrects / num_cnt

        # Test loss & acc 계산
        print('test(loss) = {:.5f} \t test(acc) = {:.5f}'.format(test_loss,
                                                                     test_acc))

if __name__ == '__main__':
    #process Start
    # run_process(mode='train')
    run_process(mode='test')
