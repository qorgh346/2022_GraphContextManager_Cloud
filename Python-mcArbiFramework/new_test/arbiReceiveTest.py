from arbi_agent.agent.arbi_agent import ArbiAgent
from arbi_agent.configuration import BrokerType
from arbi_agent.agent import arbi_agent_excutor

import time


class arbiAgent(ArbiAgent):
    def __init__(self):
        super().__init__()

    def on_data(self, sender: str, data: str):
        print(self.agent_url + "\t-> receive data : " + data)
    
    def on_request(self, sender: str, request: str) -> str:
        print(self.agent_url + "\t-> receive request : " + request)
        return "(request ok)"
    
    """
    def on_notify(self, content):
        gl_notify = GLFactory.new_gl_from_gl_string(content)
    """
    def on_query(self, sender: str, query: str) -> str:
        print(self.agent_url + "\t-> receive query : " + query)
        return "(query ok)"

    def execute(self, broker_url, agent_name, agent, broker_type=2):
        arbi_agent_excutor.excute(broker_url, agent_name, agent, broker_type)
        print(agent_name + " ready")


broker_url = "tcp://127.0.0.1:61616"
#start an agent
sender_agent = arbiAgent()
sender_agent_name = "agent://www.arbi.com/MAPFagent"
arbi_agent_excutor.excute(broker_url, sender_agent_name, sender_agent, broker_type=2)

while True:
    time.sleep(1)
