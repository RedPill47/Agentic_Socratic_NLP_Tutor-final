"""
Response Strength Scorer

Hybrid approach for scoring student responses:
1. Fast heuristic (catches 70-80% of cases instantly)
2. LLM self-grading (for uncertain cases)

Used in onboarding binary search to determine if response is strong/weak.
"""

import os
from typing import Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv
import json
import re

load_dotenv()


class ResponseStrengthScorer:
    """
    Scores student responses on a scale of 0-1.
    
    Uses hybrid approach:
    - Heuristic for obvious cases (fast, free)
    - LLM for nuanced cases (accurate, but slower)
    """
    
    def __init__(self):
        self.llm_client: Optional[AsyncOpenAI] = None
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        # Initialize LLM client if API key available
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.llm_client = AsyncOpenAI(api_key=api_key)
    
    async def score(self, response: str, question: str) -> float:
        """
        Score a student response on a scale of 0-1.
        
        Args:
            response: Student's response text
            question: The question they were answering
            
        Returns:
            Score between 0.0 (weak) and 1.0 (strong)
        """
        if not response or not response.strip():
            return 0.0
        
        # Step 1: Fast heuristic (catches 70-80% of cases)
        heuristic_score = self._heuristic_score(response)
        
        # If clearly weak or strong, return immediately
        if heuristic_score < 0.3:  # Clearly weak
            return heuristic_score
        elif heuristic_score > 0.8:  # Clearly strong
            return heuristic_score
        else:  # Uncertain - use LLM for accuracy
            if self.llm_client:
                return await self._llm_score(response, question)
            else:
                # Fallback to heuristic if LLM unavailable
                return heuristic_score
    
    def _heuristic_score(self, response: str) -> float:
        """
        Fast heuristic scoring based on length, keywords, and patterns.
        No LLM call - instant evaluation.
        """
        score = 0.5  # Start neutral
        response_lower = response.lower().strip()
        
        # Length check
        word_count = len(response.split())
        if word_count < 5:
            score -= 0.4  # Too short - likely weak
        elif word_count < 10:
            score -= 0.2  # Short - probably weak
        elif word_count > 50:
            score += 0.2  # Detailed response - likely strong
        elif word_count > 100:
            score += 0.3  # Very detailed - probably strong
        
        # Confidence indicators
        weak_indicators = [
            "i think", "maybe", "not sure", "i don't know", 
            "i'm not sure", "unsure", "probably", "perhaps",
            "?", "i guess", "not really"
        ]
        strong_indicators = [
            "because", "since", "for example", "specifically",
            "this means", "in other words", "essentially",
            "the key is", "important to note", "crucially"
        ]
        
        weak_count = sum(1 for ind in weak_indicators if ind in response_lower)
        strong_count = sum(1 for ind in strong_indicators if ind in response_lower)
        
        score += (strong_count * 0.1) - (weak_count * 0.15)
        
        # Technical terms (indicates understanding)
        technical_terms = [
            "algorithm", "model", "neural", "vector", "embedding",
            "token", "sequence", "attention", "transformer",
            "network", "layer", "activation", "gradient"
        ]
        tech_count = sum(1 for term in technical_terms if term in response_lower)
        score += min(tech_count * 0.05, 0.2)  # Cap at 0.2
        
        # Question marks (uncertainty)
        if response.count("?") > 1:
            score -= 0.1
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))
    
    async def _llm_score(self, response: str, question: str) -> float:
        """
        LLM-based scoring for nuanced cases.
        Uses lightweight model for speed.
        """
        if not self.llm_client:
            return 0.5  # Fallback to neutral
        
        prompt = f"""Evaluate this student response on a scale of 0-1.

Question: {question}
Response: {response}

Consider:
1. Correctness (0-1): Is the answer correct or on the right track?
2. Depth (0-1): Does it show understanding beyond surface level?
3. Evidence (0-1): Are there specific examples, explanations, or reasoning?

Calculate: score = (correctness * 0.4) + (depth * 0.3) + (evidence * 0.3)

Return ONLY a JSON object with this exact format:
{{"score": 0.0-1.0, "reasoning": "brief one-sentence explanation"}}

Do not include any other text."""

        try:
            completion = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an educational evaluator. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            content = completion.choices[0].message.content.strip()
            
            # Extract JSON
            json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = content
            
            result = json.loads(json_str)
            score = float(result.get("score", 0.5))
            
            # Validate score is in range
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            # Fallback to heuristic if LLM fails
            print(f"⚠️ LLM scoring failed: {e}, using heuristic")
            return self._heuristic_score(response)
    
    def classify(self, score: float) -> str:
        """
        Classify score into category.
        
        Returns:
            "strong" if score > 0.7
            "weak" if score < 0.4
            "moderate" otherwise
        """
        if score > 0.7:
            return "strong"
        elif score < 0.4:
            return "weak"
        else:
            return "moderate"

