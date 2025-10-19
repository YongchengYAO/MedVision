import json
import backoff
import ast
from PIL import Image
from tqdm import tqdm
from typing import List, Optional, Tuple, Dict, Any, Union

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, create_model


@register_model("gemini__2_5_woTool")
class Gemini__2_5_woTool(lmms):
    """
    Gemini 2.5: https://ai.google.dev/gemini-api/docs/models#model-variations

    Thinking support: Gemini 2.5 Pro and 2.5 Flash come with thinking on by default.
        - Ref: https://ai.google.dev/gemini-api/docs/thinking

    Args:
        thinkingBudget:
            - ref: https://ai.google.dev/gemini-api/docs/thinking#set-budget
            - * The thinkingBudget is only supported in Gemini 2.5 Flash, 2.5 Pro, and 2.5 Flash-Lite
            - A higher token count generally allows for more detailed reasoning, which can be beneficial for tackling more complex tasks.
            - Disable thinking by setting thinkingBudget to 0.
            - Cannot disable thinking for 2.5 Pro
            - * Setting the thinkingBudget to -1 turns on dynamic thinking, meaning the model will adjust the budget based on the complexity of the request.
            - range of thinkingBudget:
                - 2.5 Pro: 128 to 32768
                - 2.5 Flash: 0 to 24576
                - 2.5 Flash-Lite: 512 to 24576
            - Default: -1 (dynamic thinking)
        use_tool: [bool] = True
            - code execution, https://ai.google.dev/gemini-api/docs/code-execution#enable-code-execution
        json_output: [bool] = True
            - structured output, https://ai.google.dev/gemini-api/docs/structured-output#generating-json
            - json output is not supported for code execution (use_tool=True)
        json_fields: List[str], str, Dict[str, str]
            - It will be converted to a dictionary defining the fields for the structured JSON output.
        ignore_thoughts: bool=False
            - If True, the model will ignore the thoughts in the output and only return the final answer.
        ignore_code: bool=True
            - If True, the model will ignore code blocks in the output (only for use_tool=True).
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",  # gemini-2.5-pro, gemini-2.5-flash
        thinkingBudget: Optional[int] = -1,  # dynamic thinking
        use_tool: Optional[bool] = True,  # code execution
        json_output: Optional[bool] = True,  # structured output
        json_fields: Optional[Union[List[str], str, Dict[str, str]]] = ["Thought", "Answer"],
        ignore_thoughts: Optional[bool] = False,  # only return final answer in the output
        ignore_code: Optional[bool] = True,  # ignore code blocks in the output (only for use_tool=True)
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_name = model
        self.thinkingBudget = thinkingBudget
        self.use_tool = use_tool
        if self.use_tool:
            self.ignore_code = ignore_code

        # NOTE: json output is not supported for code execution (use_tool=True)
        if self.use_tool:
            self.json_output = False
        else:
            self.json_output = json_output

        # Ensure "Answer" key exists in json_fields since it's required for _generate_content_with_retry()
        if isinstance(json_fields, str):
            try:
                json_fields = ast.literal_eval(json_fields)
            except json.JSONDecodeError:
                raise ValueError(f"Failed to parse string for argument 'json_fields'")
        if isinstance(json_fields, list):
            self.json_fields = {field: str for field in json_fields}
        if "Answer" not in self.json_fields:
            raise ValueError("The 'Answer' key must be present in json_fields")

        self.ignore_thoughts = ignore_thoughts

        # Prepare the Gemini client and configuration
        self.prepare_model()

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def prepare_model(self):
        # Create JSON format class based on json_fields or use default
        if self.json_output:
            # Dynamically create BaseModel class with custom fields using create_model
            field_definitions = {}
            for field_name, field_type in self.json_fields.items():
                field_definitions[field_name] = (field_type, Field(...))
            JSON_format = create_model("JSON_format", **field_definitions)

        # Get model
        self.client = genai.Client()
        if self.use_tool:
            self.config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=self.thinkingBudget),
                response_mime_type="application/json" if self.json_output else "text/plain",
                response_schema=JSON_format if self.json_output else None,
                tools=[types.Tool(code_execution=types.ToolCodeExecution)],
            )
        else:
            self.config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=self.thinkingBudget),
                response_mime_type="application/json" if self.json_output else "text/plain",
                response_schema=JSON_format if self.json_output else None,
            )

    @backoff.on_exception(backoff.expo, Exception, max_tries=10, base=1.0, jitter=backoff.random_jitter)
    def _generate_content_with_retry(self, visual, contexts):
        response = self.client.models.generate_content(model=self.model_name, contents=[visual, contexts], config=self.config)
        if self.json_output:
            resp_json = response.parsed
            if self.ignore_thoughts:
                return getattr(resp_json, "Answer")
            else:
                final_answer = ""
                # Iterate through all keys except "Answer"
                for key in [k for k in self.json_fields.keys() if k != "Answer"]:
                    final_answer += f"{key}\n: {getattr(resp_json, key, '<not provided>')}\n"
                # Put the final answer at the end
                final_answer += f"Answer\n: {getattr(resp_json, 'Answer', '<not provided>')}\n"
                return final_answer
        elif self.use_tool:
            # ref: https://ai.google.dev/gemini-api/docs/code-execution#enable-code-execution
            final_answer = ""
            part_counter_text = 0
            part_counter_code = 0
            part_counter_code_result = 0
            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    part_counter_text += 1
                    final_answer += f"Text Block {part_counter_text}:\n{part.text}\n"
                if part.executable_code is not None:
                    part_counter_code += 1
                    if not self.ignore_code:
                        final_answer += f"Code Block {part_counter_code}:\n{part.executable_code.code}\n"
                    else:
                        final_answer += f"Code Block {part_counter_code}:\n<code block ignored>\n"
                if part.code_execution_result is not None:
                    part_counter_code_result += 1
                    final_answer += f"Code Result Block {part_counter_code_result}:\n{part.code_execution_result.output}\n"
            return final_answer
        else:
            return response.text if response.text is not None else ""

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # Image inputs
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            if len(visuals) == 1 and isinstance(visuals[0], Image.Image):
                visual = visuals[0]
            else:
                raise ValueError("We only support 1 image input for now and it should be of Image.Image type.")

            # Get model response with retry mechanism
            if self.json_output:
                question = contexts
            else:
                question = contexts + "\n\Please put the final answer after 'Answer:'\n\n"
            resp = self._generate_content_with_retry(visual, question)
            res.append(resp)
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for BiomedGPT")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for BiomedGPT")
