import os
from typing import Optional, List, Mapping, Any
import requests

from langchain import HuggingFaceHub, LLMChain
from langchain import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.llms.base import LLM
import transformers


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
    


llm = AlpacaLLM()


### demo

prompt = """Answer the question based on the context below. If the
question cannot be answered using the information provided answer
with "I don't know".

Context: Large Language Models (LLMs) are the latest models used in NLP.
Their superior performance over smaller models has made them incredibly
useful for developers building NLP enabled applications. These models
can be accessed via Hugging Face's `transformers` library, via OpenAI
using the `openai` library, and via Cohere using the `cohere` library.

Question: Which libraries and model providers offer LLMs?

Answer: """

# print(llm(prompt))



### Conversational memory

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationSummaryMemory, ConversationBufferWindowMemory

# conversation = ConversationChain(llm = llm)

# see the prompt template
# print(conversation.prompt.template)

# conversation = ConversationChain(llm = llm, memory = ConversationBufferMemory())
# print(conversation_buf("Good morning AI!"))

# conversation = ConversationChain(
#     llm = llm,
#     memory = ConversationSummaryMemory(llm = llm)
# )

conversation = ConversationChain(llm = llm, memory = ConversationBufferWindowMemory(k = 1))

### 还有ConversationSummaryBufferMemory： ConversationSummaryMemory和ConversationBufferWindowMemory的集合体

query = "Good morning AI!"
print(f"I: {query}")
print(f"AI: {conversation.run(query)}")

print("==============================")
query = "My interest here is to explore the potential of integrating Large Language Models with external knowledge"
print(f"I: {query}")
print(f"AI: {conversation.run(query)}")


print("==============================")
query = "I just want to analyze the different possibilities. What can you think of?"
print(f"I: {query}")
print(f"AI: {conversation.run(query)}")

print("==============================")
query = "Which data source types could be used to give context to the model?"
print(f"I: {query}")
print(f"AI: {conversation.run(query)}")

print("==============================")
query = "What is my aim again?"
print(f"I: {query}")
print(f"AI: {conversation.run(query)}")


print("#########################")
print(f"{conversation.memory.buffer}")

