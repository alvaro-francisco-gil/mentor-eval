import os
from typing import Dict, List, Optional, Union
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OpenAIClient:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key (str, optional): OpenAI API key. If not provided, will try to get from environment variable.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable or pass it directly.")
        
        openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key)

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict:
        """
        Make a chat completion request to OpenAI API.
        
        Args:
            messages (List[Dict[str, str]]): List of message dictionaries with 'role' and 'content'
            model (str): Model to use for completion
            temperature (float): Controls randomness (0.0 to 1.0)
            max_tokens (int, optional): Maximum number of tokens to generate
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            Dict: API response
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response
        except Exception as e:
            raise Exception(f"Error making chat completion request: {str(e)}")

    def completion(
        self,
        prompt: str,
        model: str = "gpt-3.5-turbo-instruct",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict:
        """
        Make a completion request to OpenAI API.
        
        Args:
            prompt (str): The prompt to complete
            model (str): Model to use for completion
            temperature (float): Controls randomness (0.0 to 1.0)
            max_tokens (int, optional): Maximum number of tokens to generate
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            Dict: API response
        """
        try:
            response = self.client.completions.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response
        except Exception as e:
            raise Exception(f"Error making completion request: {str(e)}")

    def embeddings(
        self,
        text: Union[str, List[str]],
        model: str = "text-embedding-ada-002",
        **kwargs
    ) -> Dict:
        """
        Get embeddings for text using OpenAI API.
        
        Args:
            text (Union[str, List[str]]): Text or list of texts to get embeddings for
            model (str): Model to use for embeddings
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            Dict: API response with embeddings
        """
        try:
            response = self.client.embeddings.create(
                model=model,
                input=text,
                **kwargs
            )
            return response
        except Exception as e:
            raise Exception(f"Error getting embeddings: {str(e)}") 