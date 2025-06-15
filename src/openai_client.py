from typing import List, Optional, Dict, Tuple, Callable, TypeVar, Generic
import json
import time
import os
from pydantic import BaseModel, Field
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
import hashlib
from difflib import SequenceMatcher
import numpy as np
from collections import defaultdict

# Load environment variables
load_dotenv()

# Generic type for the response parser
T = TypeVar('T')

class ScoringResponse(BaseModel):
    score: int = Field(ge=1, le=6)  # Score between 1 and 6
    justification: str

class OpenAIClient(Generic[T]):
    def __init__(
        self, 
        parser: Callable[[str], T],
        api_key: Optional[str] = None, 
        model: Optional[str] = None,
        similarity_threshold: float = 0.8,
        temperature: float = 0.3,
        response_format: Optional[Dict] = None  # Make response format configurable
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env file or provide it directly.")
        
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self._cache: Dict[str, T] = {}
        self._prompt_cache: Dict[str, str] = {}
        self.similarity_threshold = similarity_threshold
        self.parser = parser
        self.temperature = temperature
        self.response_format = response_format

    def _get_cache_key(self, prompt: str) -> str:
        """Generate a cache key for the prompt."""
        return hashlib.md5(prompt.encode()).hexdigest()

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings using SequenceMatcher."""
        return SequenceMatcher(None, str1, str2).ratio()

    def _find_similar_prompt(self, prompt: str) -> Optional[str]:
        """
        Find a similar prompt in the cache that exceeds the similarity threshold.
        Returns the cache key if found, None otherwise.
        """
        for cached_prompt, cache_key in self._prompt_cache.items():
            similarity = self._calculate_similarity(prompt, cached_prompt)
            if similarity >= self.similarity_threshold:
                return cache_key
        return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def _make_api_call(self, prompt: str, use_cache: bool = True) -> T:
        if use_cache:
            cache_key = self._get_cache_key(prompt)
            if cache_key in self._cache:
                return self._cache[cache_key]
            
            similar_cache_key = self._find_similar_prompt(prompt)
            if similar_cache_key and similar_cache_key in self._cache:
                return self._cache[similar_cache_key]

        try:
            # Add explicit instruction about JSON format
            enhanced_prompt = f"{prompt}\n\nPlease ensure your response is a complete, valid JSON object with both opening and closing braces."
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": enhanced_prompt}
                ],
                "temperature": self.temperature
            }
            
            if self.response_format:
                payload["response_format"] = self.response_format

            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                error_msg = f"API call failed with status {response.status_code}: {response.text}"
                print(f"Error details: {error_msg}")
                raise Exception(error_msg)

            response_data = response.json()
            content = response_data["choices"][0]["message"]["content"]
            
            # Clean the content by removing markdown code block formatting
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.startswith("```"):
                content = content[3:]  # Remove ```
            if content.endswith("```"):
                content = content[:-3]  # Remove ```
            content = content.strip()
            
            # Check if JSON is complete
            if not content.endswith("}"):
                print("Warning: Incomplete JSON detected, retrying with more explicit instructions")
                # Retry with more explicit instructions
                payload["messages"][0]["content"] = f"{prompt}\n\nIMPORTANT: Your response must be a complete, valid JSON object. Make sure to include the closing brace '}}' at the end of your response."
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                response_data = response.json()
                content = response_data["choices"][0]["message"]["content"]
                # Clean the content again
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
            
            try:
                result = self.parser(content)
            except Exception as parse_error:
                print(f"Parser error for content: {content}")
                print(f"Parse error details: {str(parse_error)}")
                raise Exception(f"Failed to parse response: {str(parse_error)}")
            
            if use_cache:
                self._cache[cache_key] = result
                self._prompt_cache[prompt] = cache_key
            
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"Request error: {str(e)}")
            raise
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            raise

    def process_batch(
        self, 
        prompts: List[str], 
        batch_size: int = 5, 
        delay: float = 1.0,
        use_cache: bool = True
    ) -> List[T]:
        """
        Process a batch of prompts with automatic similarity detection and caching.
        """
        results = []
        cache_hits = 0
        total_prompts = len(prompts)
        last_progress = 0
        processed_count = 0
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_results = []
            
            for prompt in batch:
                try:
                    result = self._make_api_call(prompt, use_cache=use_cache)
                    batch_results.append(result)
                    if use_cache and self._find_similar_prompt(prompt):
                        cache_hits += 1
                except Exception as e:
                    print(f"Error processing prompt: {str(e)}")
                    print(f"Problematic prompt: {prompt[:100]}...")  # Print first 100 chars of the prompt
                    batch_results.append(None)
                processed_count += 1
            
            results.extend(batch_results)
            
            current_progress = (processed_count / total_prompts) * 100
            if current_progress - last_progress >= 5:
                cache_hit_rate = (cache_hits / processed_count) * 100
                print(f"Progress: {current_progress:.1f}% ({processed_count}/{total_prompts})")
                print(f"Cache hit rate: {cache_hit_rate:.2f}%")
                last_progress = current_progress
            
            if i + batch_size < len(prompts):
                time.sleep(delay)
        
        return results

    def clear_cache(self):
        """Clear all caches."""
        self._cache.clear()
        self._prompt_cache.clear()

# Example usage:
if __name__ == "__main__":
    client = OpenAIClient(similarity_threshold=0.8)  # Adjust threshold as needed
    
    # Example prompts (no need to pre-group them)
    prompts = [
        "Evaluate the following response: 'The sky is blue'",
        "Evaluate the following response: 'The sky is blue and clear'",
        "Evaluate the following response: 'The sky is blue today'",
        "Score this answer: '2+2=4'",
        "Score this answer: '2+2=4 is correct'",
        # ... more prompts
    ]
    
    # Process all prompts
    results = client.process_batch(prompts)
    
    # Print results
    for prompt, result in zip(prompts, results):
        if result:
            print(f"Prompt: {prompt}")
            print(f"Score: {result.score}")
            print(f"Justification: {result.justification}")
            print("---")