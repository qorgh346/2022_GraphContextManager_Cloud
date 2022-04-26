import json
import os,sys
import numpy as np
import matplotlib.pyplot as plt

predicate = {'isBehindOf':0,'nearBy':0,'faceToFace':0}
class_list = ['AMR_LIFT1','AMR_LIFT2','AMR_TOW1','AMR_TOW2']

mapping_relationship_idx = {}
c = 0

cols = len(predicate.keys())
for i_idx,i in enumerate(class_list):
    for j_idx,j in enumerate(class_list):
        if i_idx == j_idx:continue
        mapping_relationship_idx['{}_{}'.format(i_idx,j_idx)] = c
        c+=1
mapping_class = {v:k for k,v in enumerate(class_list)} #AMR_LIFT1 : 0, AMR_LIFT2 : 1 , AMR_TOW1 : 2...
mapping_rel = {v:k for k,v in enumerate(predicate.keys())} # ' isBehindOf' : 0 , ...
distribution_table = np.zeros((len(mapping_relationship_idx.keys()),cols))


def predicate_distribution(path):
    #관계 빈도수에 따른 분포도
    datalists = os.listdir(path)
    for name in datalists:
        file_path = os.path.join(path,name)
        data = read_json(file_path)
        # print(data)
        for result in data['labels']:
            src = result[0]
            obj = result[2]
            pred = result[1]
            predicate[pred] += 1
            node_pair_distribution(src,pred,obj)

def node_pair_distribution(src,predicate,obj):
    #두 노드간 관계의 전체 분포도
    row = mapping_relationship_idx['{}_{}'.format(mapping_class[src],mapping_class[obj])]
    col = mapping_rel[predicate]
    distribution_table[row,col] += 1

def frequency_visualization(save_path,name):
    # 관계빈도수 그래프
    data = predicate.values()
    plt.bar(predicate.keys(),data)
    plt.xlabel('Predicate', fontsize=12)
    plt.ylabel('Frequency',fontsize=12)
    plt.title('Frequency distribution')
    # plt.show()

    createFolder(save_path)
    plt.savefig('{}/{}.png'.format(save_path,name))

def node_pair_visualization(save_path,name):
    rows = distribution_table.shape[0]
    plt.figure(figsize=(15,15))
    convert_mapping_rel = {v:k for k,v in mapping_relationship_idx.items()}

    for i in range(rows):
        plt.subplot(4,3,i+1)
        plt.bar(predicate.keys(),distribution_table[i],width=0.4,color=['r', 'g', 'b'])
        src_idx = convert_mapping_rel[i].split('_')[0]
        obj_idx = convert_mapping_rel[i].split('_')[1]
        plt.title('{} = {}'.format(class_list[int(src_idx)], class_list[int(obj_idx)]), fontsize=15)
        plt.ylim([0, 200])
    plt.savefig('{}/{}.png'.format(save_path, name))

def read_json(file):
    with open(file,'r') as f:
        json_data = json.dumps(json.load(f))
        gt_data = json.loads(json_data)
    return gt_data

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

if __name__ == '__main__':
    json_paths = 'C:/Users/ailab/Desktop/2022_GCM_cloud/mos_datasets_jsons'
    save_path = 'C:/Users/ailab/Desktop/2022_GCM_cloud/visualization'
    predicate_distribution(json_paths)
    frequency_visualization(save_path,'Frequency_Distribution_1')
    node_pair_visualization(save_path,'Pair_Distribution_1')