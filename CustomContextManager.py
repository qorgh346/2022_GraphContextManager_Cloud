import os,sys
import math
import json

class CustomCM:
    def __init__(self,root_dir,save_dir):
        self.root = root_dir
        self.save_dir = save_dir
        sample_lists = os.listdir(root_dir)
        self.items = dict()
        for idx,i in enumerate(sample_lists):
            data_root = os.path.join(root_dir,i)
            self.items[idx] = [os.path.join(data_root,i) for i in os.listdir(data_root)]
        self.count_temp = 0
    def init(self):
        self.json_datas = dict()
        self.json_datas['labels'] = list()
        self.json_datas['robots'] = list()


    def file_Read_Write(self,dataset):

        temp_dict = dict()
        for i in dataset:
            with open( i,'r') as file:
                datas = file.readline().split(' ')[1:]
                temp_dict[datas[0]] = datas

        self.objectInfo(temp_dict)
        self.prolog_python(temp_dict)
        with open(os.path.join(self.save_dir, 'data_{}.json'.format(self.count_temp)), 'w', encoding='utf-8') as make_file:
            json.dump(self.json_datas, make_file, indent="\t")

        self.count_temp+=1
        self.init()

    def process(self):
        print('process')
        print(self.items)
        for count,datalists in self.items.items():

            file_lists = sorted(datalists,key = lambda x:int(x.split('.')[1].split('_')[-1]))
            start = 0
            timestamp = 4
            print(file_lists)
            for i in range(4,len(file_lists),4):
                self.init()
                self.file_Read_Write(file_lists[start:i])
                start = i

    ############  Prolog Python ###############
    def isBehindOf(self,Robot1,Robot2):
    #   Diff_Theta is abs(Robot1theta - Robot2theta),
    # 	Diff_X is abs(Robot1X - Robot2X),
    # 	Diff_Y is abs(Robot1Y - Robot2Y),
    # 	(Diff_Theta < 2, Diff_X < 1, Diff_Y < 4).
        Diff_X = abs(int(Robot1[1]) - int(Robot2[1]))
        Diff_Y = abs(int(Robot1[2]) - int(Robot2[2]))
        Diff_Theta = abs(int(Robot1[3]) - int(Robot2[3]))
        if Diff_X < 2 and Diff_Y < 3: #Diff_Theta < 2 and
            return True
        return False
    def nearBy(self,Robot1,Robot2):
        #   Distance = sqrt((Robot1X - Robot2X)**2 + (Robot1Y - Robot2Y)**2),
        # 	Distance<4.
        Distance = math.sqrt((int(Robot1[1])-int(Robot2[1]))**2 + (int(Robot1[2])-int(Robot2[2]))**2)
        if Distance < 3:
            return True
        return False

    def faceToface(self,Robot1,Robot2):
        #Robot1X = Robot2X.
	    # %abs(Robot1theta - Robot2theta) >= 179, abs(Robot1theta - Robot2theta) < 181.
        if int(Robot1[1]) == int(Robot2[1]):
            if abs(int(Robot1[3])-int(Robot2[3])) >= 179 and abs(int(Robot1[3])-int(Robot2[3])) < 181:
                return True
        return False
    ###################################################################################

    def objectInfo(self,robot_infos):

        for robot_id,value in robot_infos.items():
            temp_dict = dict()
            temp_dict['name'] = value[0]
            temp_dict['status'] = value[-1]
            temp_dict['velocity'] = float(value[5])
            temp_dict['battery'] = float(value[4])
            temp_dict['pos'] = [float(i) for i in value[1:4]]

            self.json_datas['robots'].append(temp_dict)

    def prolog_python(self,robot_infos):
        #robot_infos --> dict()
        k = 0
        for sub_idx in range(0,len(robot_infos.keys())):
            for obj_idx in range(k,len(robot_infos.keys())):
                if sub_idx == obj_idx : continue
                subject = list(robot_infos.keys())[sub_idx]
                object = list(robot_infos.keys())[obj_idx]
                # print(self.json_datas)

                if self.isBehindOf(robot_infos[subject],robot_infos[object]):
                    #참이면 두 로봇 관계가 isBehindOf 임.
                    self.json_datas['labels'].append([subject,'isBehindOf',object])
                if self.nearBy(robot_infos[subject],robot_infos[object]):
                    self.json_datas['labels'].append([subject,'nearBy',object])
                if self.faceToface(robot_infos[subject],robot_infos[object]):
                    self.json_datas['labels'].append([subject, 'faceToFace', object])

            k+=1

        if len(self.json_datas['labels']) == 0:
            k = 0
            for sub_idx in range(0, len(robot_infos.keys())):
                for obj_idx in range(k, len(robot_infos.keys())):
                    if sub_idx == obj_idx: continue
                    subject = list(robot_infos.keys())[sub_idx]
                    object = list(robot_infos.keys())[obj_idx]
                    self.json_datas['labels'].append([subject, 'None', object])
                k+=1


        # sys.exit()

if __name__ == '__main__':
    print('Start')
    root = '../CM_Datasets'
    save_dir = '../mos_datasets_jsons'
    a = CustomCM(root,save_dir)
    a.process()