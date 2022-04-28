import json
import os

class GraphVIS:
    def __init__(self,save_dir):
        self.options = {'AMR_LIFT1':{"shape":"ellipse","color":"#97C2FC","id":"h''AMR_LIFT1''",
                                "label":"''AMR_LIFT1''"},
                   'AMR_LIFT2': {"shape": "ellipse", "color": "#ede6e6", "id": "h''AMR_LIFT2''",
                                 "label": "''AMR_LIFT2''"},
                   'AMR_TOW1': {"shape": "ellipse", "color": "#fad4d4", "id": "h''AMR_TOW1''",
                                 "label": "''AMR_TOW1''"},
                   'AMR_TOW2': {"shape": "ellipse", "color": "#917c7c", "id": "h''AMR_TOW2''",
                                 "label": "''AMR_TOW2''"},
                   }
        self.count = 0
        self.save_dir = save_dir
    def init_json(self):
        self.fsd_json = {"nodes":list(),"edges":list()}

    def set_Node(self,node):
        data = self.options[node]
        self.fsd_json['nodes'].append(data)

    def set_Edge(self,subject,predicate,object):

        data = {'color':"#C2FAAC","from":"h''{}''".format(subject),'to':"h''{}''".format(object),
                'label':predicate
                }
        self.fsd_json['edges'].append(data)
    def write_json(self):
        # print(os.path.join(self.save_dir, 'MosHobeGraph.json'))
        with open(os.path.join(self.save_dir, 'MosHobeGraph.json'), 'w',encoding='utf-8') as make_file:
            json.dump(self.fsd_json, make_file, indent="\t")
        import time
        time.sleep(0.5)



