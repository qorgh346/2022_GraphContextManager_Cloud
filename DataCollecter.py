import os,sys,json
#github : 배형 wooca12

save_json_root = './mos_train_jsons'
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

def write_json(relation_info, object_info,file_name):


    js_dict = dict()
    a = 3
    js_dict['id'] = file_name
    js_dict['labels'] = list()
    js_dict['robots'] = list()
    # print(relation_info)
    # (js_dict['labels'].append(i) for i in relation_info['relationships'])

    [js_dict['labels'].append(i.split(' ')) for i in relation_info['relationships']]

    temp_obj = dict()

    for robot in robot_names:
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

def collect_data(root):
    file_lists = sorted(os.listdir(root),key = lambda x:int(x.split('_')[2].split('.')[0]))

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

if __name__ =='__main__':
    root = './mos_train_datasets'
    createFolder(save_json_root)
    collect_data(root)