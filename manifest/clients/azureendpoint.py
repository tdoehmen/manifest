"""OpenRouter client."""

import copy
import logging
import os
from typing import Any, Dict, Optional
import time 
from manifest.clients.client import Client
from manifest.request import LMRequest
import urllib.request
import json
import os
import ssl

logger = logging.getLogger(__name__)
def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.


class AzureEndpointClient(Client):
    """OpenRouter client."""

    # Params are defined in https://openrouter.ai/docs/parameters
    PARAMS = {
        "engine": ("model", "meta-llama/codellama-70b-instruct"),
        "max_tokens": ("max_tokens", 1000),
        "temperature": ("temperature", 0.1),
        "top_k": ("k", 0),
        "frequency_penalty": ("frequency_penalty", 0.0),
        "presence_penalty": ("presence_penalty", 0.0),
        "stop_sequences": ("stop", None),
    }
    REQUEST_CLS = LMRequest
    NAME = "azureendpoint"
    IS_CHAT = True

    def connect(
        self,
        connection_str: Optional[str] = None,
        client_args: Dict[str, Any] = {},
    ) -> None:
        """
        Connect to the OpenRouter server.

        connection_str is passed as default OPENROUTER_API_KEY if variable not set.

        Args:
            connection_str: connection string.
            client_args: client arguments.
        """

        self.host = os.environ.get("AZURE_HOST")
        # Replace this with the primary/secondary key, AMLToken, or Microsoft Entra ID token for the endpoint
        self.api_key = os.environ.get("AZURE_API_KEY")
        if not self.api_key:
            raise Exception("A key should be provided to invoke the endpoint")
        for key in self.PARAMS:
            setattr(self, key, client_args.pop(key, self.PARAMS[key][1]))

    def close(self) -> None:
        """Close the client."""

    def get_generation_header(self) -> Dict[str, str]:
        """
        Get generation header.

        Returns:
            header.
        """
        return {'Content-Type':'application/json', 'Authorization':('Bearer '+ self.api_key), 'azureml-model-deployment': 'duckdb-nsql-v2-phi-medium-1' }

    def get_generation_url(self) -> str:
        """Get generation URL."""
        return self.host + "/score"

    def supports_batch_inference(self) -> bool:
        """Return whether the client supports batch inference."""
        return False

    def supports_streaming_inference(self) -> bool:
        """Return whether the client supports streaming inference.

        Override in child client class.
        """
        return True

    def get_model_params(self) -> Dict:
        """
        Get model params.

        By getting model params from the server, we can add to request
        and make sure cache keys are unique to model.

        Returns:
            model params.
        """
        return {"model_name": AzureEndpointClient.NAME, "engine": getattr(self, 'engine')}

    def preprocess_request_params(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess request params.

        Args:
            request: request params.

        Returns:
            request params.
        """
        # Format for chat model
        request = copy.deepcopy(request)
        prompt = request.pop("prompt")
        data = {"input_data": {"input_string": [{"role": "user", "content": prompt}], "parameters": {"stop":"\n```", "max_tokens": 500}}}

        #body = str(str.encode(json.dumps(data)))
        return super().preprocess_request_params(data)

    def postprocess_response(self, response: Dict, request: Dict) -> Dict[str, Any]:
        """
        Format response to dict.

        Args:
            response: response
            request: request

        Return:
            response as dict
        """
        new_choices = []
        response = copy.deepcopy(response)
        if "output" in response:
            new_choices.append({"text": response["output"]})
        else:
            new_choices.append({"text": ""})
        response["choices"] = new_choices
        return super().postprocess_response(response, request)
