from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from mltunex.config.llm_config import GroqConfig
from mltunex.ai_handler.prompt import HyperparameterResponsePrompt

from json_repair import repair_json


load_dotenv()

class GroqHyperparamGenerator:
    def __init__(self, hyperparameter_framework: str = "Optuna", config: GroqConfig = GroqConfig) -> None:
        self.hyperparameter_framework = hyperparameter_framework
        self.llm = ChatGroq(model = config.model, temperature = config.temperature)
        self.output_parser = JsonOutputParser()
        self.prompt_template = PromptTemplate(
            template = config.SYSTEM_PROMPT,
            input_variables = ["Data_Profile", "Top_Models", "ModelHyperparameter_Schema", "HyperparameterResponsePrompt"]
        )
        self.chain = self.prompt_template | self.llm

    def generate_response(self, data_profile: str, top_models: str, model_hyperparameter_schema: str) -> str:
        response = self.chain.invoke(
            {"Data_Profile" : data_profile,
            "Top_Models" : top_models,
            "ModelHyperparameter_Schema" : model_hyperparameter_schema,
            "HyperparameterResponsePrompt" : HyperparameterResponsePrompt.get_hyperparameter_response_prompt(self.hyperparameter_framework)}
        )

        # Format the response to ensure it is valid JSON
        response = self.response_formatter(response = response.content) #type: ignore
        return response
    
    
    def response_formatter(self, response: str) -> str:
        """
        Format the response to ensure it is valid JSON.

        Parameters
        ----------
        response : str
            The raw response string from the LLM.

        Returns
        -------
        str
            A valid JSON string.
        """
        try:
            # Attempt to parse the response as JSON
            response = response.split("</think>")[-1]
            response = repair_json(response)
            response = self.output_parser.parse(response)
            return response
        except Exception as e:
            raise ValueError(f"Invalid JSON response: {e}") from e