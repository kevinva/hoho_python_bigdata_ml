from typing import Optional, List, Mapping, Any
import requests

from langchain.chains import LLMMathChain
from langchain.agents import Tool
from langchain.llms.base import LLM
from langchain.agents import initialize_agent


def askv2(query, agent):
    try:
        response = agent(query)
        print(response)
    except ValueError as e:
        response = str(e)
        if not response.startswith("Could not parse LLM output: `"):
            raise e
        response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
        
        print(f"exception: {response}")



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


llm_math = LLMMathChain(llm = llm)

math_tool = Tool(
    name = "Calculator",
    func = llm_math.run,
    description = "Useful for when you need to answer questions about math."
)

tools = [math_tool]


zero_shot_agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3
)

askv2("what is (4.5 *2.1)^2.2 ?", zero_shot_agent)