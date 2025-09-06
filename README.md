<div align="center">
  <h1>CorpusAI: A RAG-Powered Knowledge Engine</h1>
  <p>
    An intelligent engine for unifying and querying your internal knowledge corpus—support tickets, knowledge bases, and other documents—using the power of Retrieval-Augmented Generation.
  </p>
  <p>
    <a href="#-key-features">Key Features</a> •
    <a href="#-architecture-how-it-works">Architecture</a> •
    <a href="#-getting-started">Getting Started</a> •
    <a href="#-technology-stack">Technology Stack</a> •
    <a href="#-contributing">Contributing</a>
  </p>

  [![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io)
  [![Orchestration](https://img.shields.io/badge/Orchestration-LangChain-green.svg)](https://www.langchain.com/)
  [![LLM](https://img.shields.io/badge/LLM-Ollama-lightgrey.svg)](https://ollama.com/)

</div>

---

## 📖 Overview

**CorpusAI** transforms your scattered internal documents into a centralized, intelligent knowledge base. It moves beyond simple keyword search by leveraging a **Retrieval-Augmented Generation (RAG)** pipeline to understand the context and intent of natural language queries.

Whether you're resolving support tickets, onboarding new team members, or searching for specific information in a vast sea of documents, CorpusAI provides precise, context-aware answers by consulting the original source material.

## ✨ Key Features

*   **Unified Knowledge Access**: Ingest and search across diverse document types (PDFs, text files, etc.) in a single, unified space.
*   **Natural Language Querying**: Ask questions in plain English, just as you would ask a human expert.
*   **Context-Aware Answers**: Get direct answers synthesized by an LLM, grounded in the specific information found within your documents.
*   **Source Verification**: Responses are based directly on your provided documents, minimizing hallucinations and ensuring accuracy.
*   **Local & Private**: Powered by **Ollama**, the entire pipeline can run locally on your machine, ensuring your sensitive data remains secure.
*   **Performance Monitoring**: Built-in logging and performance utilities to track and optimize query times and resource usage.
*   **Intuitive Web Interface**: A clean and interactive UI built with Streamlit for easy document management and querying.

---

## 🏗️ Architecture: How It Works

CorpusAI operates on a robust RAG pipeline, which can be broken down into two core phases:

#### **Phase 1: Ingestion (Indexing the Corpus)**

1.  **Document Loading**: Your documents (support tickets, knowledge bases, etc.) are loaded into the system.
2.  **Intelligent Chunking**: The documents are segmented into smaller, semantically meaningful chunks.
3.  **Embedding & Storage**: Each chunk is converted into a vector embedding and stored in a **ChromaDB** vector database alongside its source metadata. This creates a searchable index of your knowledge corpus.



#### **Phase 2: Retrieval & Generation (Answering a Query)**

1.  **User Query**: A user asks a question in the Streamlit UI.
2.  **Semantic Retrieval**: The query is embedded, and a similarity search is performed against the ChromaDB index to retrieve the most relevant document chunks.
3.  **Context Augmentation**: The retrieved chunks are injected as context into a prompt template.
4.  **LLM Generation**: The augmented prompt is sent to a local LLM (via Ollama), which synthesizes a coherent, human-readable answer based *only* on the provided context.
5.  **Response Display**: The final answer is displayed in the UI.



---

## 🛠️ Technology Stack

*   **Backend & UI**: [Streamlit](https://streamlit.io/)
*   **RAG Orchestration**: [LangChain](https://www.langchain.com/)
*   **Vector Database**: [ChromaDB](https://www.trychroma.com/)
*   **Local LLM Server**: [Ollama](https://ollama.com/) (for running models like Llama 3, Mistral)
*   **Embeddings Model**: [Hugging Face Sentence Transformers](https://huggingface.co/sentence-transformers) (`all-MiniLM-L6-v2`)

---

## 🚀 Getting Started

Follow these steps to set up and run CorpusAI on your local machine.

#### **Prerequisites**

*   Python 3.9+
*   [Ollama](https://ollama.com/) installed and running.

#### **Installation**

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/CorpusAI.git
    cd CorpusAI
    ```

2.  **Create a Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Your Local LLM with Ollama**:
    *   First, ensure the Ollama application is running in the background.
    *   Pull a model from the command line. We recommend starting with **Llama 3**.
    ```bash
    ollama pull llama3
    ```

#### **Running the Application**

1.  Launch the Streamlit app:
    ```bash
    streamlit run app.py
    ```
2.  Open your web browser and navigate to the local URL provided (usually `http://localhost:8501`).

---

## 💡 How to Use

1.  **Ingest Documents**: Use the interface to upload your PDF files or connect to a data source. CorpusAI will automatically process and index them.
2.  **Ask Questions**: Once a vector database is loaded, use the chat input to ask questions about the content of your documents.
3.  **Receive Answers**: Get concise, AI-generated answers directly in the chat interface.

---

## 🤝 Contributing

Contributions are welcome! If you have ideas for new features, bug fixes, or improvements, please feel free to open an issue or submit a pull request.

1.  **Fork** the repository.
2.  Create a new **branch** (`git checkout -b feature/your-feature-name`).
3.  **Commit** your changes (`git commit -m 'Add some feature'`).
4.  **Push** to the branch (`git push origin feature/your-feature-name`).
5.  Open a **Pull Request**.

---

## 📜 License


This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
