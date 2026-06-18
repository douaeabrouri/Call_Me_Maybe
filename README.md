# *This project has been created as part of the 42 curriculum by doabrour.*

# 📞 Call Me Maybe

A lightweight function-calling system powered by **Qwen3-0.6B** and **constrained decoding**.

---

## Description

Call Me Maybe is a miniature function-calling framework that enables a Small Language Model (LLM) to select and invoke predefined functions based on natural language requests.

The goal of the project is to transform user prompts such as:

> "What is the sum of 2 and 3?"

into structured function calls:

```json
{
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2,
    "b": 3
  }
}
```

The project demonstrates how constrained decoding techniques can improve the reliability of small language models by forcing outputs to follow a valid JSON structure.

---

##  Features

* Function selection using Qwen3-0.6B
* Parameter extraction from natural language
* JSON-constrained decoding
* Type validation and casting
* Error handling for malformed inputs
* Function execution pipeline
* Automatic result generation

---

##  Instructions

### Installation

Clone the repository:

```bash
git clone <repository_url>
cd call_me_maybe
```

Install dependencies:

```bash
uv sync
```

---

### Running the Project

Execute:

```bash
make run
```

or

```bash
uv run python -m src
```

Generated results will be written to:

```text
data/output/function_calling_results.json
```

---

### Debug Mode

```bash
make debug
```

---

##  Algorithm Explanation

### Function Selection

The user prompt is provided to the Qwen3-0.6B model together with the list of available functions.

Example:

```text
Request: What is the sum of 2 and 3?

Available functions:
fn_add_numbers
fn_greet
fn_reverse_string
fn_get_square_root
fn_substitute_string_with_regex
```

The model generates the most appropriate function name.

---

### Parameter Extraction

Once a function is selected, a second prompt asks the model to extract only the required parameters.

Example:

```json
{
  "a": 2,
  "b": 3
}
```

---

### Constrained Decoding

To improve reliability, generated tokens are filtered during decoding.

For every generated token:

1. Candidate tokens are evaluated.
2. Invalid JSON continuations are rejected.
3. Their logits are replaced with:

```python
float("-inf")
```

which prevents the model from selecting them.

Only tokens that keep the output syntactically valid are allowed.

This significantly reduces malformed JSON outputs.

---

### Validation Layer

After generation:

* Required parameters are verified.
* Parameter types are checked.
* Values are automatically cast when possible.
* Invalid outputs are rejected.

---

##  Design Decisions

### Why Qwen3-0.6B?

The project specification requires compatibility with Qwen3-0.6B.

The model is lightweight, fast, and capable of performing function selection and parameter extraction despite its relatively small size.

---

### Why Constrained Decoding?

Small models frequently generate:

* Invalid JSON
* Missing keys
* Incorrect formatting

Constrained decoding improves output consistency without retraining the model.

---

### Why Validation After Generation?

Even valid JSON can contain incorrect parameter types.

A validation layer provides an additional safety net before function execution.

---

##  Performance Analysis

### Accuracy

The system performs well on:

* Arithmetic requests
* Greeting generation
* String manipulation
* Square root calculations

Regex-related tasks are more challenging because they require semantic understanding rather than direct extraction.

---

### Speed

The largest computational cost comes from repeated model inference during token generation.

Several optimizations were implemented:

* Limited generation length
* Early stopping
* Candidate token filtering

---

### Reliability

Reliability is increased through:

* JSON-constrained decoding
* Parameter validation
* Type casting
* Exception handling

These mechanisms reduce crashes and malformed outputs.

---

##  Challenges Faced

### 1. Invalid JSON Generation

Problem:

The model often generated incomplete or malformed JSON.

Solution:

Implemented constrained decoding using JSON prefix validation.

---

### 2. Regex Parameter Extraction

Problem:

The model frequently copied literal values instead of generating meaningful regex patterns.

Example:

```json
{
  "regex": "34"
}
```

instead of:

```json
{
  "regex": "[0-9]+"
}
```

Solution:

Additional prompting and post-processing logic were introduced.

---

### 3. Function Misclassification

Problem:

The model occasionally selected incorrect functions.

Solution:

Improved function descriptions and prompt structure to provide stronger semantic guidance.

---

##  Testing Strategy

Testing was performed using:

### Public Test Cases

Provided prompts from the project.

### Edge Cases

Examples:

```text
Unknown functions
Invalid parameters
Missing values
Malformed JSON
```

### Manual Validation

Generated outputs were manually inspected and compared against expected results.

---

##  Example Usage

### Input

```text
What is the sum of 265 and 345?
```

### Generated Function Call

```json
{
  "name": "fn_add_numbers",
  "parameters": {
    "a": 265,
    "b": 345
  }
}
```

---

### Input

```text
Reverse the string 'hello'
```

### Generated Function Call

```json
{
  "name": "fn_reverse_string",
  "parameters": {
    "s": "hello"
  }
}
```

---

##  Resources

### Documentation

* Qwen Documentation
* Hugging Face Transformers Documentation
* PyTorch Documentation
* Python json Documentation
* Python re Documentation

### Articles

* OpenAI Function Calling Concepts
* Constrained Decoding Techniques for Language Models
* Structured Generation with LLMs

### AI Usage

AI tools were used during development for:

* Understanding constrained decoding concepts
* Debugging Python issues
* Reviewing prompt engineering strategies
* Explaining model behavior
* Improving documentation quality

All design decisions, implementation, debugging, testing, and final code integration were performed manually.
