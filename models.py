"""
Pydantic models for structured outputs in the Medical Q&A Chatbot.

This module demonstrates LangChain's Structured Outputs capability,
ensuring reliable, typed JSON responses from the LLM.
"""

from pydantic import BaseModel, Field
from typing import List, Literal


class Source(BaseModel):
    """Represents a source document citation."""
    file_name: str = Field(description="Name of the PDF source file")
    page_labels: List[str] = Field(description="List of page numbers referenced")


class MedicalResponse(BaseModel):
    """
    Structured response from the medical Q&A chatbot.

    This model ensures the LLM returns consistent, parseable responses
    with proper citations and confidence levels.
    """
    answer: str = Field(
        description="The comprehensive answer to the medical question"
    )
    sources: List[Source] = Field(
        description="List of sources used to generate the answer"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level based on context relevance"
    )


class QueryInput(BaseModel):
    """Input model for the RAG query tool."""
    query: str = Field(description="The medical question to answer")


class SummaryInput(BaseModel):
    """Input model for the summarization tool."""
    text: str = Field(description="The text to summarize")
    max_sentences: int = Field(default=3, description="Maximum sentences in summary")


class ConversationTurn(BaseModel):
    """Represents a single turn in the conversation."""
    role: Literal["user", "assistant"] = Field(description="The speaker role")
    content: str = Field(description="The message content")
