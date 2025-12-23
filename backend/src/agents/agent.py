"""
Agentic AI Module for Construction RAG System
==============================================

This module extends the RAG system with agentic capabilities:
- Multi-step reasoning
- Dynamic tool selection
- Planning and execution
- Self-reflection and correction

What is an AI Agent?
--------------------
An AI Agent is an LLM that can:
1. REASON: Think through problems step-by-step
2. PLAN: Decide what actions to take
3. ACT: Use tools to accomplish tasks
4. OBSERVE: Learn from the results
5. REFLECT: Correct mistakes and improve

The REACT Framework:
-------------------
REACT = Reasoning + Acting

Loop:
1. Thought: What should I do next?
2. Action: Choose and execute a tool
3. Observation: What was the result?
4. [Repeat until done]
5. Final Answer: Synthesize and respond

Why Agents for RAG?
-------------------
Standard RAG: Query -> Retrieve -> Generate

Agent RAG:
- Decide IF retrieval is needed
- Break complex queries into sub-queries
- Validate retrieved information
- Cross-reference multiple sources
- Ask for clarification when needed

This handles complex, multi-step questions that simple RAG cannot.

Example:
"Compare the PPE requirements in the safety manual with what's 
specified in the Smith Construction contract"

Agent approach:
1. Query 1: Get PPE requirements from safety manual
2. Query 2: Get PPE requirements from Smith contract
3. Analysis: Compare the two sets of requirements
4. Synthesis: Generate comparison response
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import re

from loguru import logger

from ..config import settings
from ..retrieval import Retriever, RetrievalResult
from ..llm import OpenAILLM, LLMResponse, ResponseType


class ToolType(Enum):
    """Available tool types for the agent."""
    SEARCH = "search"
    RETRIEVE = "retrieve"
    CALCULATE = "calculate"
    COMPARE = "compare"
    SUMMARIZE = "summarize"
    CLARIFY = "clarify"
    ANSWER = "answer"


@dataclass
class Tool:
    """
    Represents a tool the agent can use.
    
    Tools are the agent's interface to external capabilities.
    Each tool has a name, description, and function to execute.
    """
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_prompt_description(self) -> str:
        """Format tool for inclusion in agent prompt."""
        param_str = ", ".join(
            f"{k}: {v}" for k, v in self.parameters.items()
        )
        return f"- {self.name}: {self.description}\n  Parameters: {param_str}"


@dataclass
class AgentStep:
    """
    Represents a single step in the agent's reasoning.
    
    Captures the full REACT cycle for debugging and analysis.
    """
    step_number: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    is_final: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step_number,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
            "is_final": self.is_final
        }


@dataclass
class AgentResponse:
    """
    Complete agent response with reasoning trace.
    
    Includes both the final answer and the steps taken to get there.
    This transparency is crucial for debugging and trust.
    """
    answer: str
    steps: List[AgentStep]
    retrieval_used: bool
    sources: List[str]
    total_steps: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "steps": [s.to_dict() for s in self.steps],
            "retrieval_used": self.retrieval_used,
            "sources": self.sources,
            "total_steps": self.total_steps,
            "metadata": self.metadata
        }


class BaseAgent(ABC):
    """Abstract base class for agents."""
    
    @abstractmethod
    def run(self, query: str) -> AgentResponse:
        """Execute the agent on a query."""
        pass


class ConstructionRAGAgent(BaseAgent):
    """
    Agentic RAG for construction domain.
    
    This agent can:
    1. Decide if retrieval is needed for a query
    2. Break complex queries into sub-queries
    3. Use multiple tools to gather information
    4. Synthesize answers from multiple sources
    5. Handle follow-up questions
    """
    
    # Agent system prompt
    AGENT_PROMPT = """You are an intelligent construction industry assistant with access to company documents.

You have access to the following tools:

{tools}

To use a tool, respond EXACTLY in this format:
Thought: [Your reasoning about what to do]
Action: [Tool name]
Action Input: {{"param1": "value1", "param2": "value2"}}

After receiving an observation, continue with another Thought/Action/Action Input, 
OR if you have enough information, respond with:
Thought: [Final reasoning]
Final Answer: [Your complete response to the user]

IMPORTANT:
- Break complex questions into simpler sub-queries
- Always cite sources when stating facts
- If you can't find information, say so clearly
- For safety questions, be extra thorough

User Query: {query}

{context}

Begin your reasoning:"""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        llm: Optional[OpenAILLM] = None,
        max_iterations: int = 5
    ):
        """
        Initialize the construction RAG agent.
        
        Args:
            retriever: Retriever for document search
            llm: Language model for reasoning
            max_iterations: Maximum reasoning steps
        """
        self.retriever = retriever or Retriever()
        self.llm = llm or OpenAILLM()
        self.max_iterations = max_iterations
        
        # Initialize tools
        self.tools = self._create_tools()
        
        logger.info(f"Agent initialized with {len(self.tools)} tools")
    
    def _create_tools(self) -> Dict[str, Tool]:
        """Create the tools available to the agent."""
        tools = {}
        
        # Document search tool
        tools["search_documents"] = Tool(
            name="search_documents",
            description="Search construction documents for relevant information",
            function=self._tool_search,
            parameters={
                "query": "string - What to search for",
                "top_k": "integer - Number of results (default: 5)"
            }
        )
        
        # Specific document retrieval
        tools["get_document_section"] = Tool(
            name="get_document_section",
            description="Get specific section from a document",
            function=self._tool_get_section,
            parameters={
                "document": "string - Document name",
                "section": "string - Section to retrieve"
            }
        )
        
        # Comparison tool
        tools["compare_sources"] = Tool(
            name="compare_sources",
            description="Compare information from multiple sources",
            function=self._tool_compare,
            parameters={
                "topic": "string - Topic to compare",
                "sources": "list - Document names to compare"
            }
        )
        
        # Calculator tool (for construction calculations)
        tools["calculate"] = Tool(
            name="calculate",
            description="Perform construction-related calculations",
            function=self._tool_calculate,
            parameters={
                "expression": "string - Mathematical expression",
                "context": "string - What this calculation is for"
            }
        )
        
        # No retrieval needed - answer from knowledge
        tools["direct_answer"] = Tool(
            name="direct_answer",
            description="Answer simple factual questions without retrieval",
            function=self._tool_direct_answer,
            parameters={
                "answer": "string - Your direct answer"
            }
        )
        
        return tools
    
    def _tool_search(
        self,
        query: str,
        top_k: int = 5
    ) -> str:
        """Execute document search tool."""
        results = self.retriever.retrieve(query, top_k=top_k)
        
        if not results:
            return "No relevant documents found for this query."
        
        output_parts = [f"Found {len(results)} relevant chunks:\n"]
        
        for i, result in enumerate(results, 1):
            source = result.source.split('/')[-1]
            preview = result.content[:300].replace('\n', ' ')
            output_parts.append(
                f"{i}. [Source: {source}] (Score: {result.score:.2f})\n"
                f"   {preview}...\n"
            )
        
        return "\n".join(output_parts)
    
    def _tool_get_section(
        self,
        document: str,
        section: str
    ) -> str:
        """Get specific section from a document."""
        # Search with document filter
        query = f"{section} in {document}"
        results = self.retriever.retrieve(
            query,
            top_k=3,
            filter_dict={"source": {"$contains": document}}
        )
        
        if not results:
            return f"Could not find section '{section}' in document '{document}'"
        
        return f"Section from {document}:\n\n{results[0].content}"
    
    def _tool_compare(
        self,
        topic: str,
        sources: List[str]
    ) -> str:
        """Compare information across sources."""
        comparisons = []
        
        for source in sources:
            results = self.retriever.retrieve(
                f"{topic}",
                top_k=2,
                filter_dict={"source": {"$contains": source}}
            )
            
            if results:
                comparisons.append(f"\n{source}:\n{results[0].content[:500]}")
        
        if not comparisons:
            return f"Could not find information about '{topic}' in specified sources"
        
        return f"Comparison of '{topic}' across sources:\n" + "\n---\n".join(comparisons)
    
    def _tool_calculate(
        self,
        expression: str,
        context: str
    ) -> str:
        """Perform safe calculations."""
        try:
            # Safe evaluation of mathematical expressions
            allowed_names = {
                'sqrt': __import__('math').sqrt,
                'pow': pow,
                'abs': abs,
                'round': round,
                'min': min,
                'max': max,
                'pi': 3.14159265359
            }
            
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return f"Calculation: {expression} = {result}\nContext: {context}"
            
        except Exception as e:
            return f"Calculation error: {str(e)}"
    
    def _tool_direct_answer(self, answer: str) -> str:
        """Pass through for direct answers."""
        return answer
    
    def _format_tools_for_prompt(self) -> str:
        """Format tools for inclusion in agent prompt."""
        return "\n".join(
            tool.to_prompt_description() for tool in self.tools.values()
        )
    
    def _parse_agent_response(
        self,
        response: str
    ) -> Tuple[Optional[str], Optional[str], Optional[Dict], Optional[str]]:
        """
        Parse the agent's response to extract thought, action, and final answer.
        
        Returns:
            Tuple of (thought, action, action_input, final_answer)
        """
        thought = None
        action = None
        action_input = None
        final_answer = None
        
        # Extract thought
        thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|Final Answer:|$)', response, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
        
        # Check for final answer
        final_match = re.search(r'Final Answer:\s*(.+?)$', response, re.DOTALL)
        if final_match:
            final_answer = final_match.group(1).strip()
            return thought, None, None, final_answer
        
        # Extract action
        action_match = re.search(r'Action:\s*(\w+)', response)
        if action_match:
            action = action_match.group(1).strip()
        
        # Extract action input
        input_match = re.search(r'Action Input:\s*(\{.+?\})', response, re.DOTALL)
        if input_match:
            try:
                action_input = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                action_input = {"query": input_match.group(1)}
        
        return thought, action, action_input, final_answer
    
    def _should_use_retrieval(self, query: str) -> bool:
        """
        Decide if retrieval is needed for this query.
        
        Some queries don't need document retrieval:
        - Simple greetings
        - Questions about the system
        - General knowledge questions
        """
        no_retrieval_patterns = [
            r'^(hi|hello|hey|thanks|thank you)',
            r'how are you',
            r'what can you do',
            r'help me understand',
        ]
        
        query_lower = query.lower().strip()
        
        for pattern in no_retrieval_patterns:
            if re.match(pattern, query_lower):
                return False
        
        # Default: use retrieval for most queries
        return True
    
    def run(self, query: str) -> AgentResponse:
        """
        Execute the agent reasoning loop.
        
        This is the main entry point that implements the REACT loop.
        """
        logger.info(f"Agent processing query: {query[:50]}...")
        
        steps: List[AgentStep] = []
        context = ""
        sources: List[str] = []
        retrieval_used = False
        
        # Quick check: does this need retrieval?
        if not self._should_use_retrieval(query):
            # Simple query, direct response
            response = self.llm.generate(
                f"Respond helpfully to: {query}"
            )
            return AgentResponse(
                answer=response.answer,
                steps=[AgentStep(1, "Direct response without retrieval", is_final=True)],
                retrieval_used=False,
                sources=[],
                total_steps=1
            )
        
        # Build initial prompt
        tools_description = self._format_tools_for_prompt()
        
        for iteration in range(self.max_iterations):
            # Build prompt with current context
            prompt = self.AGENT_PROMPT.format(
                tools=tools_description,
                query=query,
                context=context
            )
            
            # Get agent response
            response = self.llm.generate(prompt, temperature=0.1)
            
            # Parse response
            thought, action, action_input, final_answer = self._parse_agent_response(
                response.answer
            )
            
            # Create step record
            step = AgentStep(
                step_number=iteration + 1,
                thought=thought or "No explicit thought",
                action=action,
                action_input=action_input
            )
            
            # Check for final answer
            if final_answer:
                step.is_final = True
                steps.append(step)
                
                return AgentResponse(
                    answer=final_answer,
                    steps=steps,
                    retrieval_used=retrieval_used,
                    sources=list(set(sources)),
                    total_steps=len(steps)
                )
            
            # Execute action if specified
            if action and action in self.tools:
                tool = self.tools[action]
                
                try:
                    observation = tool.function(**(action_input or {}))
                    step.observation = observation
                    
                    # Track if retrieval was used
                    if action in ["search_documents", "get_document_section", "compare_sources"]:
                        retrieval_used = True
                        # Extract sources from observation
                        source_matches = re.findall(r'\[Source: ([^\]]+)\]', observation)
                        sources.extend(source_matches)
                    
                    # Add to context for next iteration
                    context += f"\n\nStep {iteration + 1}:\n"
                    context += f"Thought: {thought}\n"
                    context += f"Action: {action}\n"
                    context += f"Observation: {observation}\n"
                    
                except Exception as e:
                    step.observation = f"Error executing action: {str(e)}"
                    logger.error(f"Tool execution error: {e}")
            
            steps.append(step)
            
            logger.debug(f"Agent step {iteration + 1}: {action or 'reasoning'}")
        
        # Max iterations reached without final answer
        # Synthesize best answer from context
        final_prompt = f"""Based on your investigation, provide a final answer.

Context gathered:
{context}

Original query: {query}

Synthesize a final answer:"""
        
        final_response = self.llm.generate(final_prompt)
        
        return AgentResponse(
            answer=final_response.answer,
            steps=steps,
            retrieval_used=retrieval_used,
            sources=list(set(sources)),
            total_steps=len(steps),
            metadata={"max_iterations_reached": True}
        )


class QueryRouter:
    """
    Routes queries to appropriate handling strategy.
    
    Decides between:
    - Simple RAG (single retrieval + generation)
    - Agent mode (multi-step reasoning)
    - Direct response (no retrieval needed)
    """
    
    def __init__(
        self,
        llm: Optional[OpenAILLM] = None,
        complexity_threshold: float = 0.7
    ):
        """
        Initialize query router.
        
        Args:
            llm: LLM for query analysis
            complexity_threshold: Threshold for agent mode
        """
        self.llm = llm or OpenAILLM()
        self.complexity_threshold = complexity_threshold
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to determine complexity and required approach.
        
        Returns analysis with:
        - complexity_score
        - requires_retrieval
        - suggested_mode: "simple", "agent", or "direct"
        - sub_queries: If complex, break into sub-queries
        """
        # Simple heuristics first
        query_lower = query.lower()
        
        # Complex indicators
        complexity_indicators = [
            "compare", "difference between", "versus", "vs",
            "multiple", "all the", "list all",
            "and also", "additionally", "furthermore",
            "why", "how does", "explain the reasoning"
        ]
        
        complexity_score = 0.3  # Base
        
        for indicator in complexity_indicators:
            if indicator in query_lower:
                complexity_score += 0.15
        
        # Length also indicates complexity
        if len(query.split()) > 20:
            complexity_score += 0.1
        
        complexity_score = min(complexity_score, 1.0)
        
        # Determine mode
        if complexity_score >= self.complexity_threshold:
            mode = "agent"
        elif "hello" in query_lower or "thanks" in query_lower:
            mode = "direct"
        else:
            mode = "simple"
        
        return {
            "complexity_score": complexity_score,
            "requires_retrieval": mode != "direct",
            "suggested_mode": mode,
            "query": query
        }
    
    def route(self, query: str) -> str:
        """
        Route query to appropriate mode.
        
        Returns: "simple", "agent", or "direct"
        """
        analysis = self.analyze_query(query)
        return analysis["suggested_mode"]


# Factory function
def create_agent(
    agent_type: str = "construction_rag",
    **kwargs
) -> BaseAgent:
    """Create an agent instance."""
    agents = {
        "construction_rag": ConstructionRAGAgent
    }
    
    if agent_type not in agents:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    return agents[agent_type](**kwargs)


if __name__ == "__main__":
    print("=" * 60)
    print("Agentic RAG Demo")
    print("=" * 60)
    
    print("\nThis module implements a REACT-style agent for construction RAG.")
    print("\nThe agent can:")
    print("  1. Search construction documents")
    print("  2. Compare information across sources")
    print("  3. Perform calculations")
    print("  4. Break complex queries into sub-queries")
    print("  5. Reason through multi-step problems")
    
    print("\nExample query that would trigger agent mode:")
    print('  "Compare the PPE requirements in the safety manual')
    print('   with what\'s specified in the subcontractor agreement"')
    
    # Demo query routing
    router = QueryRouter()
    
    queries = [
        "What is fall protection?",
        "Compare PPE requirements across all documents",
        "Hello, how are you?",
        "List all safety requirements for electrical work and explain why each is important"
    ]
    
    print("\nQuery Routing Demo:")
    for query in queries:
        result = router.analyze_query(query)
        print(f"\n  Query: {query[:50]}...")
        print(f"  Complexity: {result['complexity_score']:.2f}")
        print(f"  Mode: {result['suggested_mode']}")
