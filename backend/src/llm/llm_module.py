"""
LLM Module for Construction RAG System
=======================================

This module handles LLM inference for generating grounded responses.
It implements prompt engineering, hallucination control, and response formatting.

The "G" in RAG - Generation
----------------------------
This is where retrieved context meets language model capabilities.
The LLM synthesizes information from multiple chunks into coherent,
accurate answers.

Key Challenges:
1. Hallucination: LLM generating facts not in the context
2. Attribution: Ensuring answers cite their sources
3. Relevance: Keeping answers focused on the question
4. Consistency: Same question should give similar answers

Our Approach:
- Carefully engineered prompts
- Temperature = 0.1 for consistency
- Explicit grounding instructions
- Source citation requirements

LLM Fundamentals Connection:
----------------------------
1. Tokenization: Text -> Tokens -> IDs
   - LLMs don't see text, they see token IDs
   - Subword tokenization (BPE, SentencePiece)
   - Affects prompt design (token limits)

2. Attention Mechanism:
   - How context is "attended to" for generation
   - Long context may dilute attention on key facts
   - Why chunk quality matters

3. Temperature:
   - Controls randomness in generation
   - temp=0: Greedy (deterministic)
   - temp=1: Sample from distribution
   - For RAG, low temperature = factual

4. Prompt Engineering:
   - Zero-shot: Just the question
   - Few-shot: Examples of Q&A pairs
   - Chain-of-thought: Step-by-step reasoning
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Generator
from dataclasses import dataclass, field
from enum import Enum
import json

from loguru import logger

from ..config import settings
from ..retrieval import RetrievalResult


class ResponseType(Enum):
    """Types of responses the LLM can generate."""
    ANSWER = "answer"
    CLARIFICATION = "clarification"
    NO_INFORMATION = "no_information"
    ERROR = "error"


@dataclass
class LLMResponse:
    """
    Structured response from the LLM.
    
    Contains the answer plus metadata for evaluation
    and debugging.
    """
    answer: str
    response_type: ResponseType
    sources_used: List[str]
    confidence: float
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Token usage for cost tracking
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "answer": self.answer,
            "response_type": self.response_type.value,
            "sources_used": self.sources_used,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "token_usage": {
                "prompt": self.prompt_tokens,
                "completion": self.completion_tokens,
                "total": self.total_tokens
            }
        }


class PromptTemplate:
    """
    Prompt templates for different RAG scenarios.
    
    Well-designed prompts are critical for RAG quality.
    Each template is crafted for specific use cases.
    """
    
    # Main RAG prompt - highly engineered for accuracy
    MAIN_RAG_PROMPT = """You are an expert construction industry assistant with access to official company documents. Your role is to provide accurate, helpful answers based ONLY on the provided context.

CRITICAL INSTRUCTIONS:
1. ONLY use information from the provided context
2. If the context doesn't contain the answer, say "I don't have enough information in the available documents to answer this question."
3. NEVER make up facts, statistics, or procedures not in the context
4. Cite the source document when providing specific information
5. If information seems incomplete, acknowledge this

CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION: {question}

RESPONSE FORMAT:
- Provide a clear, direct answer
- Reference specific documents when stating facts
- If multiple sources give different information, mention this
- Include relevant safety warnings if applicable

Your response:"""

    # Prompt for when no relevant context is found
    NO_CONTEXT_PROMPT = """You are a construction industry assistant. The user asked a question, but no relevant information was found in the company documents.

USER QUESTION: {question}

Please respond by:
1. Acknowledging that you couldn't find relevant information in the documents
2. Suggesting what type of document might contain this information
3. Recommending the user consult with an appropriate team member

Your response:"""

    # Prompt for clarification when query is ambiguous
    CLARIFICATION_PROMPT = """You are a construction industry assistant. The user's question may need clarification.

USER QUESTION: {question}
POTENTIAL CONTEXTS FOUND: {num_contexts}

The question could relate to multiple topics. Please:
1. Ask for clarification about which specific aspect they're asking about
2. List the possible interpretations
3. Keep it professional and helpful

Your response:"""

    # Prompt for summarizing multiple sources
    SUMMARIZATION_PROMPT = """You are summarizing information from multiple construction documents.

SOURCES:
{context}

TASK: Provide a comprehensive summary that:
1. Identifies key points from each source
2. Highlights any differences or contradictions
3. Organizes information logically
4. Notes which source each point comes from

Summary:"""

    # Safety-focused prompt for construction queries
    SAFETY_PROMPT = """You are a construction safety expert. SAFETY IS PARAMOUNT.

SAFETY DOCUMENTATION:
{context}

SAFETY QUESTION: {question}

CRITICAL: When answering safety questions:
1. Always err on the side of caution
2. Cite specific regulations (OSHA, ANSI, etc.)
3. Include all relevant warnings
4. Recommend consulting safety personnel for complex situations
5. NEVER downplay safety requirements

Your safety-focused response:"""


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream response tokens."""
        pass


class OpenAILLM(BaseLLM):
    """
    OpenAI GPT implementation.
    
    Supports GPT-4, GPT-4 Turbo, and GPT-3.5 Turbo.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.1
    ):
        """
        Initialize OpenAI LLM.
        
        Args:
            model_name: OpenAI model to use
            api_key: API key (or from settings)
            temperature: Default temperature
        """
        self.model_name = model_name or settings.openai_model
        self.default_temperature = temperature
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key or settings.openai_api_key)
            logger.info(f"OpenAI LLM initialized: {self.model_name}")
        except ImportError:
            logger.error("openai not installed. Run: pip install openai")
            raise
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using OpenAI API."""
        temperature = temperature if temperature is not None else self.default_temperature
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful construction industry assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            answer = response.choices[0].message.content
            
            return LLMResponse(
                answer=answer,
                response_type=ResponseType.ANSWER,
                sources_used=[],  # Will be populated by RAG pipeline
                confidence=1.0,  # Placeholder
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            )
            
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            return LLMResponse(
                answer=f"Error generating response: {str(e)}",
                response_type=ResponseType.ERROR,
                sources_used=[],
                confidence=0.0
            )
    
    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream response tokens."""
        temperature = temperature if temperature is not None else self.default_temperature
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful construction industry assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                **kwargs
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            yield f"Error: {str(e)}"


class RAGGenerator:
    """
    Main RAG generation pipeline.
    
    Combines retrieved context with LLM generation
    to produce grounded, accurate responses.
    """
    
    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        prompt_template: Optional[str] = None,
        max_context_length: int = 4000,
        enable_hallucination_check: bool = True
    ):
        """
        Initialize RAG generator.
        
        Args:
            llm: Language model to use
            prompt_template: Custom prompt template
            max_context_length: Maximum context characters
            enable_hallucination_check: Enable post-generation checks
        """
        self.llm = llm or OpenAILLM()
        self.prompt_template = prompt_template or PromptTemplate.MAIN_RAG_PROMPT
        self.max_context_length = max_context_length
        self.enable_hallucination_check = enable_hallucination_check
        
        logger.info("RAG Generator initialized")
    
    def _format_context(
        self,
        retrieval_results: List[RetrievalResult]
    ) -> str:
        """Format retrieved chunks into context string."""
        if not retrieval_results:
            return ""
        
        context_parts = []
        total_length = 0
        
        for result in retrieval_results:
            source = result.source.split('/')[-1]  # Just filename
            chunk_text = f"[Source: {source}]\n{result.content}"
            
            if total_length + len(chunk_text) > self.max_context_length:
                break
            
            context_parts.append(chunk_text)
            total_length += len(chunk_text)
        
        return "\n\n---\n\n".join(context_parts)
    
    def _extract_sources(
        self,
        retrieval_results: List[RetrievalResult]
    ) -> List[str]:
        """Extract unique sources from results."""
        sources = []
        seen = set()
        
        for result in retrieval_results:
            source = result.source.split('/')[-1]
            if source not in seen:
                sources.append(source)
                seen.add(source)
        
        return sources
    
    def _detect_safety_question(self, question: str) -> bool:
        """Detect if question is safety-related."""
        safety_keywords = {
            'safety', 'hazard', 'danger', 'ppe', 'protection',
            'injury', 'accident', 'emergency', 'osha', 'fatal',
            'fall', 'electrical', 'chemical', 'fire'
        }
        
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in safety_keywords)
    
    def _check_for_hallucination(
        self,
        response: str,
        context: str
    ) -> float:
        """
        Basic hallucination check.
        
        Returns confidence score (1.0 = likely grounded, 0.0 = likely hallucinated).
        
        This is a simple heuristic-based check. For production,
        consider using NLI-based entailment checking.
        """
        if not self.enable_hallucination_check:
            return 1.0
        
        # Simple check: Are key phrases in the response also in context?
        response_lower = response.lower()
        context_lower = context.lower()
        
        # Extract numbers and specific claims from response
        import re
        numbers_in_response = set(re.findall(r'\d+', response_lower))
        numbers_in_context = set(re.findall(r'\d+', context_lower))
        
        # Check if numbers in response are in context
        if numbers_in_response:
            numbers_grounded = len(numbers_in_response & numbers_in_context)
            number_confidence = numbers_grounded / len(numbers_in_response)
        else:
            number_confidence = 1.0
        
        # Check for hedging phrases (sign of uncertainty)
        hedging_phrases = [
            "i believe", "i think", "probably", "might be",
            "it's possible", "generally speaking"
        ]
        has_hedging = any(phrase in response_lower for phrase in hedging_phrases)
        
        # Combine signals
        confidence = number_confidence * (0.9 if has_hedging else 1.0)
        
        return confidence
    
    def generate(
        self,
        question: str,
        retrieval_results: List[RetrievalResult],
        use_safety_prompt: bool = False
    ) -> LLMResponse:
        """
        Generate a response using RAG.
        
        Args:
            question: User question
            retrieval_results: Retrieved context chunks
            use_safety_prompt: Force safety-focused prompt
        
        Returns:
            LLMResponse with answer and metadata
        """
        # Handle empty context
        if not retrieval_results:
            prompt = PromptTemplate.NO_CONTEXT_PROMPT.format(
                question=question
            )
            response = self.llm.generate(prompt)
            response.response_type = ResponseType.NO_INFORMATION
            return response
        
        # Format context
        context = self._format_context(retrieval_results)
        sources = self._extract_sources(retrieval_results)
        
        # Select appropriate prompt template
        if use_safety_prompt or self._detect_safety_question(question):
            template = PromptTemplate.SAFETY_PROMPT
        else:
            template = self.prompt_template
        
        # Build prompt
        prompt = template.format(
            context=context,
            question=question
        )
        
        # Generate response
        response = self.llm.generate(prompt)
        
        # Add sources
        response.sources_used = sources
        
        # Check for hallucination
        response.confidence = self._check_for_hallucination(
            response.answer, context
        )
        
        # Add warning if low confidence
        if response.confidence < 0.7:
            response.metadata["warning"] = (
                "Response may contain information not fully grounded in documents"
            )
        
        logger.info(
            f"Generated response: {len(response.answer)} chars, "
            f"confidence={response.confidence:.2f}, "
            f"tokens={response.total_tokens}"
        )
        
        return response
    
    def generate_stream(
        self,
        question: str,
        retrieval_results: List[RetrievalResult]
    ) -> Generator[str, None, None]:
        """
        Stream response generation.
        
        Useful for real-time UI feedback.
        """
        context = self._format_context(retrieval_results)
        
        if not context:
            yield "I couldn't find relevant information in the documents."
            return
        
        if self._detect_safety_question(question):
            template = PromptTemplate.SAFETY_PROMPT
        else:
            template = self.prompt_template
        
        prompt = template.format(
            context=context,
            question=question
        )
        
        yield from self.llm.generate_stream(prompt)


class ConversationalRAG(RAGGenerator):
    """
    RAG generator with conversation history support.
    
    Maintains conversation context for follow-up questions.
    """
    
    def __init__(self, max_history: int = 5, **kwargs):
        """
        Initialize conversational RAG.
        
        Args:
            max_history: Maximum conversation turns to keep
            **kwargs: Arguments for RAGGenerator
        """
        super().__init__(**kwargs)
        self.max_history = max_history
        self.history: List[Dict[str, str]] = []
    
    def _build_conversation_context(self) -> str:
        """Build context from conversation history."""
        if not self.history:
            return ""
        
        parts = ["Previous conversation:"]
        for turn in self.history[-self.max_history:]:
            parts.append(f"User: {turn['question']}")
            parts.append(f"Assistant: {turn['answer'][:200]}...")
        
        return "\n".join(parts)
    
    def generate(
        self,
        question: str,
        retrieval_results: List[RetrievalResult],
        **kwargs
    ) -> LLMResponse:
        """Generate with conversation context."""
        # Add conversation history to context
        conv_context = self._build_conversation_context()
        
        if conv_context:
            # Prepend conversation context
            for result in retrieval_results:
                result.metadata["conversation_context"] = conv_context
        
        # Generate response
        response = super().generate(question, retrieval_results, **kwargs)
        
        # Add to history
        self.history.append({
            "question": question,
            "answer": response.answer
        })
        
        # Trim history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        return response
    
    def clear_history(self):
        """Clear conversation history."""
        self.history = []


# Factory function
def create_llm(provider: str = "openai", **kwargs) -> BaseLLM:
    """Create an LLM instance."""
    providers = {
        "openai": OpenAILLM
    }
    
    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}")
    
    return providers[provider](**kwargs)


if __name__ == "__main__":
    print("=" * 60)
    print("LLM Module Demo")
    print("=" * 60)
    
    # This requires an OpenAI API key
    print("\nTo use this module, set OPENAI_API_KEY in .env file")
    print("\nExample usage:")
    print("""
    from llm_module import RAGGenerator, RetrievalResult
    
    generator = RAGGenerator()
    
    results = [
        RetrievalResult(
            content="Fall protection is required above 6 feet.",
            score=0.92,
            chunk_id="chunk_1",
            source="safety_manual.txt",
            metadata={}
        )
    ]
    
    response = generator.generate(
        question="When is fall protection required?",
        retrieval_results=results
    )
    
    print(response.answer)
    """)
