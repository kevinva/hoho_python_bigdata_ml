### 07 Building Custom Tools for LLM Agents

from typing import Optional, List, Mapping, Any
import requests

from langchain.chains import LLMMathChain
from langchain.agents import Tool
from langchain.llms.base import LLM
from langchain.agents import initialize_agent
from langchain.tools import BaseTool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from math import pi
from typing import Union


def askv2(query, agent):
    try:
        response = agent(query)
        print(response)
    except ValueError as e:
        response = str(e)
        print(f"exception: {response}")

        if not response.startswith("Could not parse LLM output: `"):
            raise e
        response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
        
        



HOST = 'localhost:5000'
URI = f'http://{HOST}/api/v1/generate'

class AlpacaLLM(LLM):
    
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # print(f"AlpacaLLM _call! prompt: {prompt}, stop: {stop}")
        
        if isinstance(stop, list):
            stop = stop + ["\n###","\nObservation:", "\nObservations:"]

        response = requests.post(
            URI,
            json={
                "prompt": prompt,
                "temperature": 0.7,
                "max_new_tokens": 256,
                "early_stopping": True,
                "stopping_strings": stop,
                'do_sample': True,
                'top_p': 0.1,
                'typical_p': 1,
                'repetition_penalty': 1.18,
                'top_k': 40,
                'min_length': 0,
                'no_repeat_ngram_size': 0,
                'num_beams': 1,
                'penalty_alpha': 0,
                'length_penalty': 1,
                'seed': -1,
                'add_bos_token': True,
                'truncation_length': 2048,
                'ban_eos_token': False,
                'skip_special_tokens': True,
            },
        )
        response.raise_for_status()
        return response.json()['results'][0]['text']

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}
    


class CircumferenceTool(BaseTool):
    name = "Circumference calculator"
    description = "use this tool when you need to calculate a circumference using the radius of a circle"

    def _run(self, radius: Union[int, float]):
        return float(radius)*2.0*pi

    def _arun(self, radius: int):
        raise NotImplementedError("This tool does not support async")
    


image_caption_desc = (
    "use this tool when given the URL of an image that you'd like to be "
    "described. It will return a simple caption describing the image."

)

class ImageCaptionTool(BaseTool):
    name = "Image captioner"             # 重要：影响prompt的内容
    description = image_caption_desc     # 重要：影响prompt的内容

    def _run(self, url: str):
        # download the image and convert to PIL object
        image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
        # preprocess the image
        inputs = processor(image, return_tensors="pt").to(device)
        # generate the caption
        out = model.generate(**inputs, max_new_tokens=20)
        # get the caption
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

    

if __name__ == "__main__":
    llm = AlpacaLLM()

    # conversational_memory = ConversationBufferWindowMemory(
    #         memory_key='chat_history',
    #         k=5,
    #         return_messages=True
    # )

    # # conversational_memory = ConversationBufferMemory(memory_key = "chat_history")

    # # tools = [CircumferenceTool()]

    # agent = initialize_agent(
    #     agent='chat-conversational-react-description',
    #     tools=tools,
    #     llm=llm,
    #     verbose=True,
    #     max_iterations=3,
    #     early_stopping_method='generate',
    #     memory=conversational_memory
    # )

    # agent = initialize_agent(
    #     agent="zero-shot-react-description", 
    #     tools=tools, 
    #     llm=llm,
    #     verbose=True,
    #     max_iterations=3,
    # )

    # print(agent.agent.llm_chain.prompt.messages[0].prompt.template)

    # askv2("can you calculate the circumference of a circle that has a radius of 7.81mm", agent)


    # sys_msg = """Assistant is a large language model trained by OpenAI.

    #             Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

    #             Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

    #             Unfortunately, Assistant is terrible at maths. When provided with math questions, no matter how simple, assistant always refers to it's trusty tools and absolutely does NOT try to answer math questions by itself

    #             Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
    #         """
    
    # new_prompt = agent.agent.create_prompt(
    #     system_message = sys_msg,
    #     tools = tools
    # )
    # agent.agent.llm_chain.prompt = new_prompt

    # askv2("can you calculate the circumference of a circle that has a radius of 7.81mm", agent)

    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration

    hf_model = "Salesforce/blip-image-captioning-large"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = BlipProcessor.from_pretrained(hf_model)
    model = BlipForConditionalGeneration.from_pretrained(hf_model).to(device)

    import requests
    from PIL import Image

    img_url = 'https://images.unsplash.com/photo-1616128417859-3a984dd35f02?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2372&q=80' 
    # image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
     
    # inputs = processor(image, return_tensors = "pt").to(device)
    # out = model.generate(**inputs, max_new_tokens = 20)
    # print(processor.decode(out[0], skip_special_tokens = True))

    # tools = [ImageCaptionTool()]
    tools = [CircumferenceTool()]

    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
    )

    ### 注意： agent用'chat-conversational-react-description'可能会有问题
    agent = initialize_agent(
        agent='conversational-react-description',
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=3,
        early_stopping_method='generate',
        memory=conversational_memory
    )

    # sys_msg = """Assistant is a large language model trained by OpenAI.

    # Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

    # Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

    # Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
    # """

    # new_prompt = agent.agent.create_prompt(
    #     system_message=sys_msg,
    #     tools=tools
    # )
    # agent.agent.llm_chain.prompt = new_prompt

    # update the agent tools
    agent.tools = tools
    print(agent.agent.llm_chain.prompt.template)
    # askv2(f"What does this image show?\n{img_url}", agent)
    askv2("can you calculate the circumference of a circle that has a radius of 7.81mm", agent)

    

