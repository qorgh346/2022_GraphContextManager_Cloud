import os,sys,json
import numpy as np
import shutil
#github : 배형 wooca12

save_json_root = '../mos_datasets_jsons'
object_predicate = ['currentRobotPose','hasStatus','robotVelocity','batteryRemained']
rel_predicate = ['nearBy','isBehindOf','faceToFace']

temp_flag = {'currentRobotPose':[3,6],'hasStatus':[3,4],'robotVelocity':[3,4],'batteryRemained':[3,4]}
object_info = {'hasStatus':dict(),'robotVelocity':dict(),'batteryRemained':dict(),'currentRobotPose':dict()}
relation_info = {'relationships':list()}

robot_names = ['AMR_LIFT1','AMR_LIFT2','AMR_TOW1','AMR_TOW2']

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
def write_txt(data,file_name,save_dir):
    createFolder(save_dir)
    with open(os.path.join(save_dir,'{}.txt'.format(file_name)), "w") as file:
        file.writelines(data)
    file.close()


def write_json(relation_info, object_info,file_name):


    js_dict = dict()
    a = 3
    js_dict['id'] = file_name
    js_dict['labels'] = list()
    js_dict['robots'] = list()
    # print(relation_info)
    # (js_dict['labels'].append(i) for i in relation_info['relationships'])

    [js_dict['labels'].append(i.split(' ')) for i in relation_info['relationships']]



    for robot in robot_names:
        temp_obj = dict()
    #zz
        try:
            #로봇 정보 추가하고 싶으면 아래와 같이 추가
            temp_obj['name'] = robot
            temp_obj['status'] = object_info['hasStatus'][robot]
            temp_obj['velocity'] = float(object_info['robotVelocity'][robot])
            temp_obj['battery'] = float(object_info['batteryRemained'][robot])
            temp_obj['pos'] = object_info['currentRobotPose'][robot]

            # print(robot,'-->',[robot])
        except:
            object_info['currentRobotPose'][robot] = [0.0,0.0,0.0] #초기 위치로!

        js_dict['robots'].append(temp_obj)

    with open(os.path.join(save_json_root,'{}.json'.format(file_name)), 'w', encoding='utf-8') as make_file:
        json.dump(js_dict, make_file, indent="\t")
    print('Save Json!')

def train_test_val_split(data_list,test_size=0.2,val_size=0.1, shuffle=True, random_state=1004):
    test_num = int(data_list.shape[0] * test_size)
    val_num = int(data_list.shape[0] * val_size)
    train_num = data_list.shape[0] - test_num - val_num
    print('####### total_datasets Number #######\n --> ',data_list.shape[0])
    print('####### train_num ##########\n -->',train_num)
    print('####### test_num ###########\n -->',test_num)
    print('####### val_num ############\n -->',val_num)
    if shuffle:
        np.random.seed(random_state)
        shuffled = np.random.permutation(data_list.shape[0])
        X = data_list[shuffled]
        X_train = X[:train_num]
        X_test = X[train_num:train_num+test_num]
        X_val = X[train_num+test_num:val_num+train_num+test_num]
    else:
        X_train = data_list[:train_num]
        X_test = data_list[train_num:test_num+train_num]
        X_val = data_list[train_num+test_num:val_num+test_num+train_num]
    # print(X_val)
    # sys.exit()
    return X_train, X_test,X_val

def collect_data(root):
    # data_folders = os.listdir(root)

    file_lists = sorted(os.listdir(root),key = lambda x:int(x.split('_')[2].split('.')[0]))
    print(file_lists)
    ######## 22.04.17 ############
    ######## Train ,Test txt 파일로 나누기 #############
    train_list,test_list,val_list = train_test_val_split(np.array(file_lists),test_size=0.2,val_size=0.1,shuffle=False)

    train_list = ['data_{}.json'.format(i.split('_')[2][:-4])+'\n' for i in train_list ]
    test_list = ['data_{}.json'.format(i.split('_')[2][:-4])+'\n' for i in test_list ]
    val_list = ['data_{}.json'.format(i.split('_')[2][:-4])+'\n' for i in val_list ]

    write_txt(train_list,file_name='train_list',save_dir='./split_dataset_list')
    write_txt(test_list, file_name='test_list', save_dir='./split_dataset_list')
    write_txt(val_list, file_name='val_list', save_dir='./split_dataset_list')

    #########################################

    for idx,file in enumerate(file_lists):
        data_path = os.path.join(root,file)
        # with open(data_path,'r')
        print("===========\tREAD\t============\n\t\t{}".format(file))
        f = open(data_path, "r")
        datas = f.read().split('\n')
        for data in datas:
            # print(data.split(' '))
            data_list = data.split(' ')
            if len(data_list) == 1:
                continue
            predicate = data_list[1].replace('(',' ').lstrip()
            # print(data_list)
            # print(predicate)
            if predicate in object_predicate:
                #Object Node json 작성
                robot_name = data_list[2].split("#")[-1][:-1]
                temp_data = data_list[temp_flag[predicate][0]:temp_flag[predicate][1]]

                if len(temp_data) == 1:
                    value = temp_data[0].replace('"', " ").lstrip()[:-3]

                    object_info[predicate][robot_name] = value

                else:
                    x = float(temp_data[0][2:-1])
                    y = float(temp_data[1][:-1])
                    z = float(temp_data[2][:-4])
                    object_info[predicate][robot_name] =[x,y,z]
            elif predicate in rel_predicate:
                #관계 작성
                # print(predicate)
                src = data_list[2].split('#')[1][:-1]
                tar = data_list[3].split('#')[1][:-3]
                # print(src,tar)
                relation_info['relationships'].append('{} {} {}'.format(src,predicate,tar))


        write_json(relation_info,object_info,'data_{}'.format(idx))
        # 초기화
        relation_info['relationships'] = list()

def move_data(root):
    folder_lists = os.listdir(root)
    c = 0
    for idx,folder in enumerate(folder_lists):
        if idx == 0:
            continue
        src = os.path.join(root,folder)
        dir = '../mos_train_datasets'
        files = os.listdir(src)
        # print(files)
        for file in files:
            filename = 'data_info_{}.txt'.format(c)
            a = os.path.join(dir, filename)
            shutil.move(os.path.join(src,file), os.path.join(dir ,filename))
            c += 1
    print('끝')
    # sys.exit()

if __name__ =='__main__':
    root = '../CM_Datasets'
    createFolder(save_json_root)
    # move_data(root)
    collect_data('../mos_train_datasets')