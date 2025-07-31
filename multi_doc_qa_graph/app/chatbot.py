from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

class MultiDocRAGBot(ChatOpenAI):
    def __init__(self, model):
        load_dotenv()
        super().__init__(model=model)
        self.tools = []
    def add_tool(self, tool):
        self.tools.append(tool)
    def run(self):
        while True:

