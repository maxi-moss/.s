# Opper Answer Relevance Code Examples

This document contains comprehensive code examples for implementing Answer Relevance evaluation using the Opper SDK.

## Core Implementation Patterns

### 1. @evaluator Decorator Pattern
```python
from opperai import AsyncOpper, evaluator
from opperai.types import Metric
from typing import List

@evaluator
async def answer_relevance_evaluator(question: str, answer: str) -> List[Metric]:
    """Evaluate if answer addresses the original question"""
    return [
        Metric(
            dimension="relevance", 
            value=relevance_score,  # 0.0 to 1.0
            comment=f"Relevance based on {n_synthetic_questions} synthetic questions"
        )
    ]
```

### 2. Span-Based Tracing Integration
```python
# Links relevance evaluation to model call in Opper dashboard
await opper.evaluate(
    span_id=response.span_id,  # Automatic dashboard integration
    evaluators=[answer_relevance_evaluator(question=question, answer=answer)]
)
```

### 3. RAG System Integration
```python
from pydantic import BaseModel, Field
from typing import List

class RAGOutput(BaseModel):
    answer: str = Field(..., description="The answer to the question")
    context_used: List[str] = Field(..., description="Context used for answering")

async def rag_with_relevance_eval(question: str, context: List[str]):
    # Generate RAG response
    answer, response = await opper.call(
        name="rag",
        model="openai/gpt-4o",
        instructions="Answer based on provided context",
        input={"question": question, "context": context},
        output_type=RAGOutput,
    )
    
    # Evaluate relevance
    await opper.evaluate(
        span_id=response.span_id,
        evaluators=[
            answer_relevance_evaluator(question=question, answer=answer.answer)
        ],
    )
    
    return answer.answer
```

### 4. Asynchronous Evaluation for Performance
```python
import asyncio

# Non-blocking evaluation for production systems
async def fast_rag_with_async_eval(question: str, context: List[str]):
    answer, response = await opper.call(
        name="rag",
        instructions="Answer the question using provided context",
        input={"question": question, "context": context},
        output_type=str,
    )
    
    # Evaluate asynchronously to avoid blocking response
    asyncio.create_task(opper.evaluate(
        span_id=response.span_id,
        evaluators=[answer_relevance_evaluator(question, answer)]
    ))
    
    return answer
```

### 5. Multi-Dimensional Assessment
```python
# Combine relevance with other evaluators for comprehensive assessment
await opper.evaluate(
    span_id=response.span_id,
    evaluators=[
        answer_relevance_evaluator(question=question, answer=answer),
        answer_groundedness_evaluator(answer=answer, context=context),
        faithfulness_evaluator(answer=answer, context=context),
    ]
)
```

## Practical Examples

### Example 1: Simple Q&A Evaluation
```python
import asyncio
from opperai import AsyncOpper

async def evaluate_qa_relevance():
    opper = AsyncOpper()
    
    question = "What is the capital of France?"
    answer = "Paris is the capital and largest city of France."
    
    # Direct evaluation call
    result, response = await opper.call(
        name="qa_task",
        instructions="Answer the question directly",
        input=question,
        output_type=str,
    )
    
    # Evaluate relevance
    await opper.evaluate(
        span_id=response.span_id,
        evaluators=[answer_relevance_evaluator(question=question, answer=result)]
    )
    
    return result

# Run the evaluation
asyncio.run(evaluate_qa_relevance())
```

### Example 2: Batch Evaluation for Model Comparison
```python
async def compare_models_relevance():
    questions = [
        "How does photosynthesis work?",
        "What causes climate change?",
        "Explain quantum computing"
    ]
    
    models = ["openai/gpt-4o", "anthropic/claude-3.7-sonnet", "google/gemini-pro"]
    
    for model in models:
        total_relevance = 0
        
        for question in questions:
            answer, response = await opper.call(
                name=f"qa_comparison_{model.replace('/', '_')}",
                model=model,
                instructions="Provide a clear, direct answer",
                input=question,
                output_type=str,
            )
            
            # Evaluate each answer
            await opper.evaluate(
                span_id=response.span_id,
                evaluators=[answer_relevance_evaluator(question, answer)]
            )
        
        print(f"Model {model} average relevance: {total_relevance/len(questions)}")
```

### Example 3: Real-time Chat Evaluation
```python
class ChatMessage(BaseModel):
    user_question: str = Field(..., description="User's question")
    bot_response: str = Field(..., description="Bot's response")
    
async def chat_with_evaluation(user_input: str, conversation_history: List[str]):
    # Generate response
    response, call_obj = await opper.call(
        name="chat_bot",
        instructions="You are a helpful assistant. Answer questions clearly and directly.",
        input={
            "current_question": user_input,
            "history": conversation_history
        },
        output_type=str,
    )
    
    # Real-time relevance evaluation
    asyncio.create_task(opper.evaluate(
        span_id=call_obj.span_id,
        evaluators=[
            answer_relevance_evaluator(question=user_input, answer=response)
        ]
    ))
    
    return response
```

## Optimization and Configuration Examples

### Performance Tuning

#### Number of Synthetic Questions Configuration
```python
# Configuration for synthetic question generation
SYNTHETIC_QUESTIONS_CONFIG = {
    "development": 3,  # Fast iteration
    "staging": 4,      # Balanced accuracy/cost
    "production": 5    # High accuracy
}

@evaluator
async def configurable_relevance_evaluator(
    question: str, 
    answer: str, 
    num_questions: int = 3
) -> List[Metric]:
    """Configurable relevance evaluator"""
    return [
        Metric(
            dimension="relevance",
            value=relevance_score,
            comment=f"Relevance score using {num_questions} synthetic questions"
        )
    ]
```

#### Embedding Model Selection
```python
# High accuracy configuration (recommended for evaluation)
EMBEDDING_CONFIG_HIGH_ACCURACY = {
    "model": "text-embedding-3-large",
    "dimensions": 3072
}

# Cost-optimized configuration (for high-volume scenarios)
EMBEDDING_CONFIG_COST_OPTIMIZED = {
    "model": "text-embedding-3-small", 
    "dimensions": 1536
}

async def relevance_with_config(question: str, answer: str, config: dict):
    """Relevance evaluation with configurable embedding model"""
    # Implementation would use the specified embedding configuration
    pass
```

#### Domain Customization with Few-Shot
```python
# Use Opper datasets for domain-specific improvements
async def setup_domain_examples():
    dataset_id = "question_generation_examples"

    # Add few-shot examples to improve synthetic question quality
    examples = [
        {
            "input": "What is machine learning?",
            "output": "Machine learning is a subset of AI that enables computers to learn from data.",
            "metadata": {"domain": "technology", "question_type": "definition"}
        },
        {
            "input": "How does blockchain work?",
            "output": "Blockchain is a distributed ledger technology that maintains a continuously growing list of records.",
            "metadata": {"domain": "technology", "question_type": "process"}
        },
        {
            "input": "What causes inflation?",
            "output": "Inflation is caused by increased money supply, demand-pull factors, and cost-push factors.",
            "metadata": {"domain": "economics", "question_type": "causation"}
        }
    ]
    
    for example in examples:
        await opper.datasets.add_example(
            dataset_id=dataset_id,
            input=example["input"],
            output=example["output"],
            metadata=example["metadata"]
        )
```

### Monitoring and Alerting

#### Set Relevance Thresholds
```python
async def evaluate_with_threshold(question: str, answer: str, min_relevance: float = 0.7):
    """Evaluate with quality threshold for production monitoring"""
    
    # Perform evaluation (implementation details in Opper SDK)
    relevance_score = await calculate_relevance(question, answer)
    
    if relevance_score < min_relevance:
        # Log low relevance for monitoring
        print(f"LOW RELEVANCE ALERT: Score {relevance_score} for question: {question}")
        
        # Could integrate with monitoring systems
        # send_alert_to_slack(f"Low relevance detected: {relevance_score}")
        # log_to_monitoring_system("relevance_threshold_breach", relevance_score)
        
    return relevance_score
```

#### Production Monitoring Integration
```python
import logging
from datetime import datetime

class RelevanceMonitor:
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.logger = logging.getLogger("relevance_monitor")
        
    async def monitor_relevance(self, question: str, answer: str, span_id: str):
        """Monitor relevance with logging and alerting"""
        
        # Evaluate relevance
        await opper.evaluate(
            span_id=span_id,
            evaluators=[answer_relevance_evaluator(question, answer)]
        )
        
        # Could extract score from evaluation result for monitoring
        # relevance_score = extract_relevance_score(evaluation_result)
        
        # Log for analytics
        self.logger.info({
            "timestamp": datetime.utcnow().isoformat(),
            "question": question,
            "answer_length": len(answer),
            "span_id": span_id,
            # "relevance_score": relevance_score
        })

# Usage in production
monitor = RelevanceMonitor(threshold=0.75)

async def production_rag_with_monitoring(question: str, context: List[str]):
    answer, response = await opper.call(
        name="production_rag",
        instructions="Answer based on context",
        input={"question": question, "context": context},
        output_type=str,
    )
    
    # Monitor relevance
    await monitor.monitor_relevance(question, answer, response.span_id)
    
    return answer
```

## Advanced Use Cases

### A/B Testing for Prompt Optimization
```python
async def ab_test_prompts(question: str, context: List[str]):
    """Compare different prompts using relevance scores"""
    
    prompts = {
        "direct": "Answer the question directly based on the context.",
        "detailed": "Provide a comprehensive answer using the context. Include specific details and examples.",
        "concise": "Give a brief, focused answer based on the context."
    }
    
    results = {}
    
    for prompt_name, prompt_text in prompts.items():
        answer, response = await opper.call(
            name=f"ab_test_{prompt_name}",
            instructions=prompt_text,
            input={"question": question, "context": context},
            output_type=str,
        )
        
        # Evaluate relevance for each prompt
        await opper.evaluate(
            span_id=response.span_id,
            evaluators=[answer_relevance_evaluator(question, answer)]
        )
        
        results[prompt_name] = {
            "answer": answer,
            "span_id": response.span_id
        }
    
    return results
```

### Multi-Language Relevance Evaluation
```python
async def multilingual_relevance_eval(question: str, answer: str, language: str):
    """Evaluate relevance for non-English languages"""
    
    # Could include language-specific processing
    language_config = {
        "spanish": {"embedding_model": "text-embedding-3-large"},
        "french": {"embedding_model": "text-embedding-3-large"},
        "german": {"embedding_model": "text-embedding-3-large"}
    }
    
    config = language_config.get(language, {"embedding_model": "text-embedding-3-large"})
    
    # Evaluate with language-aware configuration
    return await answer_relevance_evaluator(question, answer)
```

This comprehensive collection of code examples demonstrates practical implementation patterns for Answer Relevance evaluation using the Opper SDK, covering basic usage, optimization, monitoring, and advanced use cases.