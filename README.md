# LLM Message Classifier

This project provides a framework for classifying text messages into predefined categories using a local Large Language Model (LLM). It's designed to let you experiment with different prompting strategies (zero-shot, one-shot, and few-shot) to find the most effective approach for your needs.

This setup uses a local LLM server (like LM Studio, Ollama, or others) and does **not** require an OpenAI API key.

## Features

- **Local LLM Powered**: Runs classification locally, ensuring data privacy and no API costs.
- **Prompt Engineering**: Easily test different prompt strategies:
  - Zero-Shot
  - One-Shot
  - Few-Shot
- **Customizable**: Define your own categories and easily adapt the prompts.

## Project Structure

```
.
├── .gitignore
├── README.md
├── examples/
│   └── 1/
│       ├── messages.json
│       ├── results.json
│       ├── results/
│       │   ├── few_shot_results.json
│       │   ├── one_shot_results.json
│       │   └── zero_shot_results.json
│       └── shot_examples.json
├── main.py
├── prompts/
│   ├── few_shot/
│   │   ├── system.txt
│   │   └── user.txt
│   ├── one_shot/
│   │   ├── system.txt
│   │   └── user.txt
│   └── zero_shot/
│       ├── system.txt
│       └── user.txt
├── pytest.ini
├── requirements.txt
├── src/
│   ├── classifier.py
│   └── file_handler.py
└── tests/
    └── test_classifier.py
```

## Getting Started

### Prerequisites

1.  **Python 3.8+**
2.  **A running local LLM server**. The server must provide an OpenAI-compatible API endpoint. Popular options include:
    - [LM Studio](https://lmstudio.ai/)
    - [Ollama](https://ollama.ai/) (with a proxy like `litellm`)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd test_250801
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    # On Windows, use: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

Before running the application, you need to configure the `main.py` script to point to your local LLM server's API endpoint. 

Open `main.py` and modify the `base_url` parameter in the `OpenAI()` client initialization:

```python
# Example for LM Studio
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
```

### Usage

With your local LLM server running, execute the main script:

```bash
python main.py
```

You can modify `main.py` to change the message content, categories, and the prompting strategy you wish to test.
