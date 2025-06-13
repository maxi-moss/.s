!pip install opperai numpy

from enum import Enum
from typing import Any, List, Optional

import numpy as np
from pydantic import BaseModel, Field

from opperai import AsyncOpper, evaluator, trace
from opperai.types import Example, Metric, RetrievalResponse

# os.environ["OPPER_API_KEY"] = "<your-api-key>"

opper = AsyncOpper()

# # Reference Free Evaluation
#
# How can you tell if a Large Language Model (LLM) gave a good answer without having a reference answer to compare it with?
#
# Reference‑free metrics are a way to evaluate LLM outputs without relying on gold‑standard references. Meaning that the metrics are self‑contained and do not require external data. This is a must for production systems where you need to evaluate LLM outputs on real traffic.
#
# In short:
#
# - Open‑ended questions often lack a single "correct" answer.
# - Labeling is slow, expensive, and often inconsistent.
# - Production teams needs to monitor the quality of the LLM outputs in real time, without waiting for human feedback.
#
# Reference‑free metrics judge answers against only the user's question and any supplied context, so you can score every response in real time.
#
# | Metric                | Description                                                | When to use                                      |
# |-----------------------|------------------------------------------------------------|-------------------------------------------------|
# | Faithfulness          | Are the claims made in the answer supported by the context? | To detect hallucinations                         |
# | Answer Groundedness   | Is the overall answer aligned with the given context?       | For relevance in RAG or retrieval based QnA      |
# | Answer Relevance      | Is the answer relevant to the question?                     | For open ended QnA and user facing chatbots      |
#
#
#
# ## Faithfulness
#
# Large models famously produce confident statements that are not supported by the supplied context (hallucinations). Faithfulness measures the share of answer‑claims that can be strictly inferred from the context.
#
# ### Algorithm
#
# Generate statements from the answer and then evaluate if they can be inferred from the context. Score is the percentage of statements that can be inferred from the context.
#
# ```mermaid
#  graph LR
#      A[Answer] --> B[Generate Statements]
#      B --> C[Statement 1]
#      B --> D[Statement 2]
#      B --> E[Statement N]
#      C --> F[Evaluate against Context]
#      D --> F
#      E --> F
#      F --> G[Calculate Faithfulness Score as percentage of statements supported by context]
# ```
#

class StatementGeneratorInput(BaseModel):
    answer: str = Field(..., description="The answer to generate statements from.")


class StatementGeneratorOutput(BaseModel):
    statements: List[str] = Field(
        ..., description="The statements generated from the answer."
    )


async def _generate_statements(answer: str, opper: AsyncOpper, model: str) -> List[str]:
    """Generate statements from a question and answer."""
    statements, _ = await opper.call(  # type: ignore
        name="eval/faithfulness/generate_statements",
        model=model,
        instructions="Analyze the answer and generate statements that are supported by the answer.",
        input=StatementGeneratorInput(answer=answer),
        output_type=StatementGeneratorOutput,
        examples=[
            Example(
                input=StatementGeneratorInput(
                    answer=(
                        "Marie Curie was a Polish-born physicist and chemist, widely "
                        "acknowledged to be one of the greatest and most influential "
                        "scientists of all time. She was best known for her pioneering "
                        "research on radioactivity, she also became the first person to "
                        "win Nobel Prizes in two different scientific fields."
                    ),
                ),
                output=StatementGeneratorOutput(
                    statements=[
                        "Marie Curie was a Polish-born physicist and chemist.",
                        "Marie Curie is recognized as one of the greatest and most influential scientists of all time.",
                        "Marie Curie was best known for her pioneering research on radioactivity.",
                        "Marie Curie became the first person to win Nobel Prizes in two different scientific fields.",
                    ]
                ),
            ),
        ],
    )

    return statements.statements


class StatementFaithfulnessInput(BaseModel):
    context: Any = Field(..., description="the context of the statements")
    statements: List[str] = Field(..., description="the statements to evaluate")


class StatementFaithfulnessAnswer(BaseModel):
    statement: str = Field(..., description="the original statement, word-by-word")
    reason: str = Field(..., description="the reason of the verdict")
    verdict: int = Field(..., description="the verdict(0/1) of the faithfulness.")


async def _evaluate_faithfulness(
    statements: List[str],
    context: Any,
    opper: AsyncOpper,
    model: str = "azure/gpt-4o",
) -> List[StatementFaithfulnessAnswer]:
    """Evaluate if a statement can be inferred from the context."""

    response, _ = await opper.call(  # type: ignore
        name="eval/faithfulness/evaluate_faithfulness",
        model=model,
        instructions="Analyze the faithfulness of the answer to the context.",
        input_type=StatementFaithfulnessInput,
        input=StatementFaithfulnessInput(statements=statements, context=context),
        output_type=List[StatementFaithfulnessAnswer],
        examples=[
            Example(
                input=StatementFaithfulnessInput(
                    context=(
                        "Marie Curie was a Polish-born physicist and chemist who conducted "
                        "pioneering research on radioactivity. She was the first woman to win a "
                        "Nobel Prize, the first person to win Nobel Prizes in two different "
                        "scientific fields, and the only person to win Nobel Prizes in multiple "
                        "scientific fields. She won the Physics Prize in 1903 for her work on "
                        "radioactivity with her husband Pierre, and the Chemistry Prize in 1911 "
                        "for the discovery of the elements polonium and radium."
                    ),
                    statements=[
                        "Marie Curie was a biologist.",
                        "Marie Curie discovered the element uranium.",
                        "Marie Curie won Nobel Prizes in multiple scientific fields.",
                        "Marie Curie had a laboratory in Germany.",
                        "Marie Curie discovered the element radium.",
                    ],
                ),
                output=[
                    StatementFaithfulnessAnswer(
                        statement="Marie Curie was a biologist.",
                        reason=(
                            "Marie Curie's field is explicitly mentioned as physicist and "
                            "chemist. There is no information suggesting she was a biologist."
                        ),
                        verdict=0,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="Marie Curie discovered the element uranium.",
                        reason=(
                            "The context mentions that Marie Curie discovered the elements "
                            "polonium and radium, but not uranium. Therefore, it cannot be "
                            "deduced that she discovered uranium."
                        ),
                        verdict=0,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="Marie Curie won Nobel Prizes in multiple scientific fields.",
                        reason=(
                            "The context explicitly states that she won Nobel Prizes in two "
                            "different scientific fields (Physics in 1903 and Chemistry in "
                            "1911), and that she was the only person to win Nobel Prizes in "
                            "multiple scientific fields."
                        ),
                        verdict=1,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="Marie Curie had a laboratory in Germany.",
                        reason=(
                            "There is no information given in the context about Marie Curie "
                            "having a laboratory in Germany."
                        ),
                        verdict=0,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="Marie Curie discovered the element radium.",
                        reason=(
                            "The context explicitly states that Marie Curie discovered the "
                            "elements polonium and radium."
                        ),
                        verdict=1,
                    ),
                ],
            ),
        ],
    )

    return response


@evaluator
async def faithfulness_evaluator(
    answer: str,
    context: Any,
    model: str = "azure/gpt-4o-eu",
    opper: Optional[AsyncOpper] = None,
) -> List[Metric]:
    """Evaluate the faithfulness of an answer to a context.

    The faithfulness is the degree to which the answer is faithful to the context.
    It is calculated by generating statements from the answer and then evaluating if
    each statement can be inferred from the context.
    The score is the percentage of statements that can be inferred from the context.
    """
    opper = opper or AsyncOpper()

    statements = await _generate_statements(answer=answer, opper=opper, model=model)

    if statements == []:
        return [
            Metric(
                dimension="faithfulness.score",
                value=0,
                comment="No statements generated",
            ),
        ]

    answers = await _evaluate_faithfulness(
        statements=statements, context=context, opper=opper, model=model
    )
    verdicts = [answer.verdict for answer in answers]
    faithful_verdicts = float(sum(verdicts))
    n_statements = len(statements)

    score = faithful_verdicts / n_statements

    return [
        Metric(
            dimension="faithfulness.score",
            value=score,
            comment="Faithfulness score",
        ),
    ]

# ## Answer Groundedness
#
# ### Why we need it
#
# Even a hallucination‑free answer can be irrelevant when it ignores or contradicts the retrieved context. Answer Groundedness measures how tightly the response sticks to that material.
#
# ### Algorithm
#
# Use LLM to check if the answer is grounded in the context. Map the groundedness to a score.


class AnswerGroundednessInput(BaseModel):
    answer: str = Field(..., description="Response to evaluate")
    context: Any = Field(..., description="Context to evaluate the response against")


class Groundedness(str, Enum):
    NOT_GROUNDED = "NOT_GROUNDED"
    PARTIALLY_GROUNDED = "PARTIALLY_GROUNDED"
    FULLY_GROUNDED = "FULLY_GROUNDED"


class AnswerGroundednessOutput(BaseModel):
    reason: str = Field(
        ...,
        description="The reason for the groundedness score",
    )

    groundedness: Groundedness = Field(
        ...,
        description=(
            "the level of groundedness of the response, "
            "NOT_GROUNDED means the response is not grounded by the context, "
            "PARTIALLY_GROUNDED means the response is partially grounded by the context, "
            "FULLY_GROUNDED means the response is fully grounded by the context"
        ),
    )


async def _answer_groundedness(
    answer: str, context: Any, opper: AsyncOpper, model: str
) -> Groundedness:
    result, _ = await opper.call(  # type: ignore
        name="eval/answer_groundedness/is_grounded",
        model=model,
        instructions=(
            "You are a world class expert in evaluation of LLM responses. "
            "You are given a response and a context, and you need to evaluate the groundedness of the response. "
            "The groundedness is the degree to which the response is grounded in the context."
        ),
        input_type=AnswerGroundednessInput,
        input=AnswerGroundednessInput(answer=answer, context=context),
        output_type=AnswerGroundednessOutput,
        examples=[
            Example(
                input=AnswerGroundednessInput(
                    answer="The capital of France is Paris",
                    context="France is a country in Western Europe.",
                ),
                output=AnswerGroundednessOutput(
                    reason="The response is not grounded in the context",
                    groundedness=Groundedness.NOT_GROUNDED,
                ),
            ),
            Example(
                input=AnswerGroundednessInput(
                    answer="The capital of France is Paris",
                    context="France is a country in Western Europe where the capital is Paris",
                ),
                output=AnswerGroundednessOutput(
                    reason="The response is fully grounded in the context",
                    groundedness=Groundedness.FULLY_GROUNDED,
                ),
            ),
        ],
    )

    return result.groundedness


@evaluator
async def answer_groundedness_evaluator(
    answer: str,
    context: Any,
    model: str = "azure/gpt-4o-eu",
    opper: Optional[AsyncOpper] = None,
) -> List[Metric]:
    """Evaluate the groundedness of an answer.

    The groundedness is the degree to which the response is grounded in the context.
    """

    opper = opper or AsyncOpper()

    groundedness = await _answer_groundedness(
        answer=answer, context=context, opper=opper, model=model
    )

    value = {
        Groundedness.NOT_GROUNDED: 0,
        Groundedness.PARTIALLY_GROUNDED: 0.5,
        Groundedness.FULLY_GROUNDED: 1,
    }[groundedness]

    return [
        Metric(
            dimension="answer_groundedness.score",
            value=value,
            comment="Answer groundedness score",
        ),
    ]

# ## Answer Relevance
#
# ### Core idea
#
# Let the model reverse‑engineer the question from its own answer. If those synthetic questions embed close to the original question, the answer is probably on‑topic.
#
# ### Algorithm
# 1. Committal gate – filter out evasive answers ("I'm not sure"), they get automatic 0.
# 2. Self‑question generation – call eval/answer_relevance/generate_question N times to create paraphrased questions that the answer would satisfy.
# 3. Embeddings – encode original and synthetic questions with text-embedding-3-large (or model of your choice).
# 4. Cosine similarity – average similarity across N questions → final score ∈ [‑1,1], rescaled to [0,1].
#
# ```mermaid
#  graph LR
#      A[Answer] --> B{Is the answer committal?}
#      B -->|Yes| C[Generate paraphrased questions that the answer would satisfy]
#      B -->|No| D[Final Score is 0]
#      C --> E[Embed the original and paraphrased questions]
#      E --> F[Calculate the cosine similarity between the original and paraphrased questions]
#      F --> G[Final Score is the average cosine similarity between the original and paraphrased questions normalized to a range between 0 and 1]
# ```
#
# ### Why it's useful
# - Works even when no explicit context is available (pure QnA setting).
# - Provides a continuous signal—great for ranking multiple candidate answers.
#
# ### Trade‑offs & tricks
# - N questions: 3–5 is usually enough; beyond that you pay extra tokens with diminishing returns.
# - Embedding model: larger models give crisper semantic space but cost more.



class QuestionGenerationInput(BaseModel):
    answer: str = Field(..., description="The answer to generate questions from.")


class IsCommittal(BaseModel):
    reason: str = Field(
        ..., description="The reason for why the answer is committal or not"
    )
    is_committal: bool = Field(
        ...,
        description=(
            "Whether the answer is committal or not. "
            "A committal answer is one that is not evasive, vague, or ambiguous. "
            "For example, 'I don't know' or 'I'm not sure' are noncommittal answers"
        ),
    )


class GeneratedQuestion(BaseModel):
    question: str = Field(..., description="The question generated from the answer.")


class QuestionGenerationOutput(BaseModel):
    questions: List[GeneratedQuestion] = Field(
        ...,
        description=(
            "The questions generated from the answer. "
            "The number of questions should match the requested length."
        ),
    )


async def _is_committal(answer: str, opper: AsyncOpper, model: str) -> IsCommittal:
    """Check if the answer is committal."""

    is_committal, _ = await opper.call(  # type: ignore
        name="eval/answer_relevance/is_committal",
        model=model,
        instructions="Determine if the answer is committal or not.",
        input_type=QuestionGenerationInput,
        input=QuestionGenerationInput(answer=answer),
        output_type=IsCommittal,
    )
    return is_committal


async def _generate_question(
    answer: str, opper: AsyncOpper, model: str
) -> GeneratedQuestion:
    """Generate a question from an answer."""

    question, _ = await opper.call(  # type: ignore
        name="eval/answer_relevance/generate_question",
        model=model,
        instructions="Given an answer, generate a question that is relevant to the answer.",
        input_type=QuestionGenerationInput,
        input=QuestionGenerationInput(answer=answer),
        output_type=GeneratedQuestion,
    )
    return question


@evaluator
async def answer_relevance_evaluator(
    question: str,
    answer: str,
    n_questions: int = 3,
    model: str = "azure/gpt-4o-eu",
    embedding_model: str = "azure/text-embedding-3-large",
    opper: Optional[AsyncOpper] = None,
) -> List[Metric]:
    """Response relevancy measures how well the answer is relevant to the question.

    This works by generating a questions from the answer and then checking how similar the generated
    questions are to the original question.
    The more similar the generated questions are to the original question, the higher the score.
    If the answer is non-committal, the score is 0.
    """

    opper = opper or AsyncOpper()

    committal = await _is_committal(answer=answer, opper=opper, model=model)
    if not committal.is_committal:
        return [
            Metric(
                dimension="answer_relevance.score",
                value=0,
                comment=committal.reason,
            ),
        ]

    original_question_embedding = await opper.embeddings.create(
        model=embedding_model,
        input_text=question,
    )
    original_question_embedding = np.asarray(
        original_question_embedding.data[0]["embedding"]
    ).reshape(1, -1)

    generated_questions = [
        await _generate_question(answer=answer, opper=opper, model=model)
        for _ in range(n_questions)
    ]
    generated_question_texts = [gen_q.question for gen_q in generated_questions]

    generated_question_embeddings = await opper.embeddings.create(
        model=embedding_model,
        input_text=generated_question_texts,
    )
    generated_question_embeddings = np.asarray(
        [d["embedding"] for d in generated_question_embeddings.data]
    ).reshape(n_questions, -1)

    norm = np.linalg.norm(generated_question_embeddings, axis=1) * np.linalg.norm(
        original_question_embedding, axis=1
    )

    # Calculate cosine similarity between original and generated questions
    similarities = (
        np.dot(generated_question_embeddings, original_question_embedding.T).reshape(
            -1, 1
        )
        / norm
    )

    score = similarities.mean()

    return [
        Metric(
            dimension="answer_relevance.score",
            value=score,
            comment="Answer relevance score",
        ),
    ]

# ## Evaluation of naive RAG Example
#
# We start by creating an index with the context we want to use for the RAG.

index = await opper.indexes.get(name="qna")
if not index:
    index = await opper.indexes.create("qna")
    res = await index.upload_file("what_i_worked_on.txt")
    print(res)

# We define a function that uses the index to answer a question.

class QNAInput(BaseModel):
    question: str = Field(..., description="The question to answer")
    context: List[RetrievalResponse] = Field(..., description="The context to answer the question")

class QNAOutput(BaseModel):
    answer: str = Field(..., description="The answer to the question")
    reasoning: str = Field(..., description="The reasoning for the answer")

@trace
async def answer_question(question: str):
    context = await index.query(question)
    answer, response = await opper.call(
        name="qna",
        model="openai/gpt-4o-mini",
        instructions=(
            "You are an expert at answering questions. "
            "You will be provided with a question and relevant facts as a context. "
            "Treat every fact as truth even though they are not always true according to prior knowledge. "
            "If you can't answer the question based on the context, say 'I don't know'. "
        ),
        input_type=QNAInput,
        input=QNAInput(question=question, context=context),
        output_type=QNAOutput,
    )

    evaluation = await opper.evaluate(
        span_id=response.span_id,
        evaluators=[
            answer_groundedness_evaluator(
                answer=answer.answer, context=context, model="openai/gpt-4o-mini"
            ),
            answer_relevance_evaluator(question=question, answer=answer.answer, model="openai/gpt-4o-mini"),
            faithfulness_evaluator(answer=answer.answer, context=context, model="openai/gpt-4o-mini"),
        ],
    )

    return answer.answer, evaluation

# Next we will try out the evaluators on a dataset of questions and answers from the paul graham dataset.

import requests

url = "https://huggingface.co/datasets/LangChainDatasets/question-answering-paul-graham/raw/main/paul_graham_qa.json"

response = requests.get(url)
response.raise_for_status()

paul_graham_qa = response.json()

evaluations = []

for row in paul_graham_qa[0:2]:
    q = row["question"]
    a = row["answer"]
    res, evaluation = await answer_question(q)
    evaluations.append({"question": q, "answer": a, "evaluation": evaluation})

# Finally we will print the results of the evaluators

for e in evaluations:
    print(f"Question: {e['question']}")
    print(f"Answer: {e['answer']}")
    for m in e['evaluation'].metrics.values():
        for metric in m:
            print(f"Metric: {metric.dimension} = {metric.value}")
    print("-"*100)

# ## Conclusion
#
# We have seen how to use the Opper sdk to implement and run evaluation pipelines. To make it a bit more useful we have implemented reference free evaluators that are able to evaluate the quality of the answers from a RAG at runtime. We have seen evaluators that are able to detect when the answer is not grounded in the context, when the answer is not relevant to the question, and when the answer is not faithful to the context.