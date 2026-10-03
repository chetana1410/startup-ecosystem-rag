# 🚀 EcoAI: Where Data Meets Dialogue

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Framework: Streamlit](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io)
[![VectorDB: Chroma](https://img.shields.io/badge/VectorDB-Chroma-blueviolet.svg)](https://www.trychroma.com/)

**An advanced chatbot that harnesses the power of local LLMs to unlock hyper-precise answers from your document repository.**

EcoAI is an innovative, cutting-edge chatbot designed to facilitate seamless interactions between users and their data. By leveraging local Large Language Models (LLMs) and a sophisticated retrieval pipeline, EcoAI enables users to engage in more productive, informative, and personalized conversations.

---

## 🌟 Key Feature: Two-Level Retrieval System

Standard retrieval systems often struggle to find the precise "nugget" of information within a large document chunk. Our system overcomes this with a novel two-level retrieval architecture that ensures the answers are not just relevant, but hyper-focused and accurate.

### How It Works

1.  **Document Processing & Propositioning**:
    * Instead of just chunking documents, the system first processes them into **propositions**. A proposition is a single, atomic fact or statement extracted from the text. This creates a highly granular and semantically rich knowledge base.

2.  **Level 1: Proposition-Based Retrieval**:
    * When a user asks a query, the system first searches against the database of propositions. This allows it to find the *exact sentences* or facts that are semantically closest to the user's question.

3.  **Level 2: Agentic Re-ranking & Contextualization**:
    * The retrieved propositions and their source documents are passed to a **Re-ranking Agent**. This agent, powered by an Ollama LLM, analyzes the propositions in their original context and intelligently re-ranks the source documents to select the best one for formulating a complete answer.

---

## ⚙️ Installation

Follow these steps to set up and run the project on your local machine.

### 1. Prerequisites

* **Python 3.11 or higher.**
* **Ollama installed and running locally.** You can download it from [ollama.com](https://ollama.com).

### 2. Clone the Repository

First, clone the project repository to your local machine:
```bash
git clone https://github.com/chetana1410/startup-ecosystem-rag.git
cd startup-ecosystem-rag
```

### 3. Download the Local LLM

After installing Ollama, you need to pull the language model. Run the following command in your terminal:
```bash
ollama pull llama3.1:8b-instruct-fp16
```

**Note:** You can use any model supported by Ollama. To change the model, simply update the ``LLM_MODEL_NAME`` variable in the ``source/config/llm_config.py`` file.

### 4. Install Poetry

This project uses Poetry for dependency management. If you don't have it, run the appropriate command for your OS.

**Linux, macOS, Windows (WSL):**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**Windows (PowerShell):**
```bash
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```
Verify the installation by running ``poetry --version``.

### 5. Install Project Dependencies

Navigate to the project root (where ``pyproject.toml`` is located) and run:
```bash
poetry install
```
This will create a virtual environment and install all required packages.

### 6. Activate the virtual environment by running:

```bash

eval $(poetry env activate)

```

## ▶️ Usage

**1. Prepare Your Data**

Sample documents are already in ``data/``. To use your own, replace those files with ``.pdf``, ``.txt``, or ``.md`` files. The first ingest writes a local ``vector_data/`` index, which is gitignored.

**2. Ingest Your Documents (First-Time Setup)**

```bash
poetry shell
LOAD_DOCS=1 streamlit run app.py
```

Stop the app after the progress bars finish. Later runs should omit ``LOAD_DOCS`` so the index is reused.

**3. Launch the Application**

First, activate the virtual environment using Poetry:

```bash
poetry shell
```

Then, run the Streamlit app:

```bash
streamlit run app.py
```

This will start a local web server and open the EcoAI interface in your browser.

**IMPORTANT NOTE:** The first time you run the application and submit a query, it may take longer to respond. This is because the embedding and re-ranker models are being downloaded and cached in the background. Subsequent runs will be much faster.


## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

1. Fork the Project

2. Create your Feature Branch (git checkout -b feature/AmazingFeature)

3. Commit your Changes (git commit -m 'Add some AmazingFeature')

4. Push to the Branch (git push origin feature/AmazingFeature)

5. Open a Pull Request


## 📄 License

This project is distributed under the MIT License.