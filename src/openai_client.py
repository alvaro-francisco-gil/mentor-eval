from typing import List, Optional, Dict, Tuple
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

class ScoringResponse(BaseModel):
    score: int = Field(ge=1, le=6)  # Score between 1 and 6
    justification: str

class OpenAIClient:
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: Optional[str] = None,
        similarity_threshold: float = 0.7
    ):
        # Load from environment variables if not provided
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env file or provide it directly.")
        
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        # Initialize cache
        self._cache: Dict[str, ScoringResponse] = {}
        self._prompt_cache: Dict[str, str] = {}  # Stores prompt -> cache_key mapping
        self.similarity_threshold = similarity_threshold

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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _make_api_call(self, prompt: str, use_cache: bool = True) -> ScoringResponse:
        if use_cache:
            # First check for exact match
            cache_key = self._get_cache_key(prompt)
            if cache_key in self._cache:
                return self._cache[cache_key]
            
            # Then check for similar prompts
            similar_cache_key = self._find_similar_prompt(prompt)
            if similar_cache_key and similar_cache_key in self._cache:
                return self._cache[similar_cache_key]

        # If no cache hit, make the API call
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a scoring assistant that provides scores and justifications in JSON format."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "response_format": {"type": "json_object"}
        }

        response = requests.post(
            self.base_url,
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"API call failed with status {response.status_code}: {response.text}")

        try:
            response_data = response.json()
            content = response_data["choices"][0]["message"]["content"]
            result = ScoringResponse.parse_raw(content)
            
            # Cache the result
            if use_cache:
                self._cache[cache_key] = result
                self._prompt_cache[prompt] = cache_key
            
            return result
        except (KeyError, json.JSONDecodeError) as e:
            raise Exception(f"Failed to parse API response: {str(e)}")

    def process_batch(
        self, 
        prompts: List[str], 
        batch_size: int = 5, 
        delay: float = 1.0,
        use_cache: bool = True
    ) -> List[ScoringResponse]:
        """
        Process a batch of prompts with automatic similarity detection and caching.
        """
        results = []
        cache_hits = 0
        total_prompts = len(prompts)
        
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
                    batch_results.append(None)
            
            results.extend(batch_results)
            
            # Print progress and cache hit rate
            processed = min(i + batch_size, total_prompts)
            cache_hit_rate = (cache_hits / processed) * 100 if processed > 0 else 0
            print(f"Processed {processed}/{total_prompts} prompts. Cache hit rate: {cache_hit_rate:.2f}%")
            
            # Add delay between batches
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