# Exercise 1: Understanding Application - LangChain Integration Report

## Project Overview

### Application Summary

The **Medical Q&A Chatbot** is a Retrieval-Augmented Generation (RAG) application that answers medical questions using a
knowledge base of PDF documents. The application uses:

- **Frontend**: Streamlit web interface
- **Vector Store**: FAISS for efficient similarity search
- **Embeddings**: HuggingFace medical embeddings (MedEmbed-large-v0.1)
- **LLM**: GPT-4o-mini via Azure endpoint
- **Framework**: LangChain

### Key Features

1. PDF document ingestion and indexing
2. Semantic search over medical documents
3. Context-aware question answering
4. Source citation and document preview

### Areas for Enhancement

The following areas were identified for LangChain integration:

- **Structured Outputs**: Ensure consistent, parseable responses
- **LCEL**: Modern chain composition for better code maintainability
- **Tools & Agents**: Automated task orchestration
- **Chat Memory**: Multi-turn conversation support

---

## Integration of LangChain Concepts

### 1. Chat Models

#### Implementation

Chat Models are the core of conversational AI.
We enhanced the chatbot with conversation memory to support multi-turn interactions.

**File**: `rag.py`

```python
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder

def get_conversational_chain():
    """Create a chain that maintains conversation history."""
    llm = get_llm()
    retriever = get_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a medical assistant..."),
        MessagesPlaceholder(variable_name="chat_history"),  # Dynamic history injection
        ("human", "{input}")
    ])

    chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["input"])),
            "chat_history": lambda x: format_chat_history(x.get("chat_history", [])),
            "input": lambda x: x["input"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
```

#### Benefits

- **Context Awareness**: The model remembers previous questions and can handle follow-up queries
- **Natural Conversation**: Users can ask "What about the dosage?" without restating the drug name
- **Improved UX**: More natural, human-like interaction

---

### 2. Tools & Tool Calling

#### Implementation

Custom tools allow the LLM to perform specific actions beyond text generation.
We created four specialized tools:

**File**: `tools.py`

```python
from langchain_core.tools import tool

@tool
def search_medical_knowledge(query: str) -> str:
    """
    Search the medical knowledge base for information.
    Use this tool when you need to find medical information about
    drugs, conditions, treatments, or mechanisms of action.
    """
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)
    # Format and return results...

@tool
def summarize_context(text: str, max_sentences: int = 3) -> str:
    """Summarize long texts into key points."""
    # LLM-based summarization...

@tool
def list_available_sources() -> str:
    """List all available PDF documents in the knowledge base."""
    # File system inspection...

@tool
def get_drug_interactions(drug_name: str) -> str:
    """Search for drug interaction information."""
    # Targeted search for interaction data...
```

#### Agent Orchestration

The Agent decides which tool to use based on the user's question:

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor

def get_medical_agent() -> AgentExecutor:
    llm = get_llm()
    tools = [search_medical_knowledge, summarize_context,
             list_available_sources, get_drug_interactions]

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)
```

#### Benefits

- **Specialized Actions**: Each tool handles a specific task efficiently
- **Autonomous Decision Making**: LLM chooses the right tool for each query
- **Extensibility**: Easy to add new tools (e.g., drug calculators, reference lookup)

---

### 3. Structured Outputs

#### Implementation

Structured outputs ensure the LLM returns consistent, typed JSON responses using Pydantic models.

**File**: `models.py`

```python
from pydantic import BaseModel, Field
from typing import List, Literal

class Source(BaseModel):
    """Represents a source document citation."""
    file_name: str = Field(description="Name of the PDF source file")
    page_labels: List[str] = Field(description="Page numbers referenced")

class MedicalResponse(BaseModel):
    """Structured response from the medical Q&A chatbot."""
    answer: str = Field(description="The comprehensive answer")
    sources: List[Source] = Field(description="Sources used")
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level based on context relevance"
    )
```

**Usage in Chain** (`rag.py`):

```python
def get_structured_chain():
    chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm.with_structured_output(MedicalResponse)  # Guaranteed structured output
    )
    return chain
```

#### Benefits

- **Type Safety**: Guaranteed response structure with validation
- **Reliable Parsing**: No need for regex or string parsing
- **Confidence Levels**: Built-in quality assessment for answers
- **Automatic Citations**: Structured source tracking

---

### 4. LCEL (LangChain Expression Language)

#### Implementation

LCEL provides a declarative way to compose chains using the pipe (`|`) operator.

**Before (Traditional)**:

```python
# Old approach with create_stuff_documents_chain
qa_chain = create_stuff_documents_chain(llm, prompt, document_prompt=document_prompt)
rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=qa_chain)
```

**After (LCEL)**:

```python
# Modern LCEL approach
chain = (
    {
        "context": retriever | format_docs,  # Parallel retrieval + formatting
        "input": RunnablePassthrough()
    }
    | prompt                                   # Apply prompt template
    | llm.with_structured_output(MedicalResponse)  # LLM with structured output
)
```

#### Key LCEL Features Used

| Feature | Usage | Benefit |
| --------- | ------- | --------- |
| Pipe Operator (`\|`) | Chain composition | Readable, linear flow |
| `RunnablePassthrough()` | Pass input through | Simple data forwarding |
| `RunnableLambda` | Custom functions | Flexible transformations |
| Dictionary syntax | Parallel execution | Run retriever and passthrough simultaneously |

#### Benefits

- **Readability**: Linear, intuitive chain composition
- **Composability**: Easy to add/remove/modify steps
- **Streaming Support**: Built-in support for streaming responses
- **Debugging**: Clear execution flow for troubleshooting

---

## Implementation with Generative AI

### Scenario 1: Enhanced Medical Q&A with Confidence Scoring

**Use Case**: Medical professionals need to know how reliable an answer is based on available evidence.

```python
# Query the chatbot
response = query_with_structured_output("What is the mechanism of action of azithromycin?")

# Response structure:
# {
#   "answer": "Azithromycin is a macrolide antibiotic that works by...",
#   "sources": [{"file_name": "azithromycin.pdf", "page_labels": ["3", "4"]}],
#   "confidence": "high"
# }

# Application can display confidence indicator:
if response.confidence == "high":
    display_with_green_badge(response.answer)
elif response.confidence == "low":
    display_with_warning("This answer may be incomplete...")
```

**Benefit**: Users can assess answer reliability before making decisions.

---

### Scenario 2: Multi-Turn Medical Consultation

**Use Case**: Patients asking follow-up questions about medications.

```python
# First question
history = []
answer1 = query_with_history("What are the side effects of azithromycin?", history)
history.append({"role": "user", "content": "What are the side effects..."})
history.append({"role": "assistant", "content": answer1})

# Follow-up question (model remembers context)
answer2 = query_with_history("What about drug interactions?", history)
# Model understands "drug interactions" refers to azithromycin
```

**Benefit**: Natural conversation flow without repeating context.

---

### Scenario 3: Autonomous Information Gathering with Agents

**Use Case**: Complex queries requiring multiple search strategies.

```python
# User asks a complex question
response = query_with_agent(
    "Compare the side effects of azithromycin with other antibiotics "
    "and summarize the key differences"
)

# Agent automatically:
# 1. Uses search_medical_knowledge to find azithromycin info
# 2. Searches for other antibiotic side effects
# 3. Uses summarize_context to condense findings
# 4. Returns comprehensive comparison
```

**Benefit**: Agent handles complex queries autonomously, combining multiple tools.

---

## Project Structure

```bash
Q-A-Chatbot/
├── app.py              # Streamlit UI with mode selection
├── rag.py              # RAG chains (Standard, LCEL, Conversational)
├── tools.py            # Custom tools and Agent setup
├── models.py           # Pydantic models for structured outputs
├── ingest.py           # Document ingestion and vectorstore
├── utils.py            # Utility functions
├── requirements.txt    # Dependencies
└── data/               # PDF documents
```

---

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables (create .env file)
OPENAI_API_KEY=your_key
BASE_URL=https://models.inference.ai.azure.com
MODEL_NAME=gpt-4o-mini

# Run the application
streamlit run app.py
```

### Mode Selection

The sidebar allows selecting different chain implementations:

1. **Standard RAG**: Original retrieval chain
2. **LCEL + Structured Output**: Modern chain with Pydantic models
3. **Conversational**: Chain with memory for multi-turn chat
4. **Agent with Tools**: Autonomous agent with custom tools

---

## Summary of Benefits

| LangChain Concept | Implementation | Business Value |
| ------------------- | ---------------- | ---------------- |
| **Chat Models** | Conversation memory | Natural multi-turn interactions |
| **Tools & Agents** | 4 custom tools + agent | Autonomous task completion |
| **Structured Outputs** | Pydantic models | Reliable, typed responses |
| **LCEL** | Pipe-based chains | Maintainable, composable code |

---

## Future Enhancements

1. **Streaming Responses**: Leverage LCEL's streaming for real-time output
2. **More Tools**: Add drug dosage calculator, interaction checker API
3. **Evaluation**: Implement LangSmith for chain evaluation and monitoring
4. **Memory Persistence**: Store conversation history in a database
5. **RAG Improvements**: Add re-ranking, hybrid search

---

## Conclusion

This exercise demonstrates how LangChain's core concepts can enhance a medical Q&A chatbot:

- **Structured Outputs** ensure reliable, parseable responses with confidence scoring
- **LCEL** provides clean, maintainable chain composition
- **Tools & Agents** enable autonomous information gathering and task orchestration
- **Chat Models with Memory** support natural multi-turn conversations

These enhancements improve both the developer experience (cleaner code, easier debugging) and user experience (better answers,
natural conversations, reliability indicators).
