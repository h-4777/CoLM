# **Co**llaboration in **L**arge-**M**odels(**CoLM**)

[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

![CoLM](C:\Users\sqqq\Downloads\CoLM.jpg)

<p align="center">
  <a href="#colM-overview"><strong>Overview</strong></a> ·
  <a href="#quick-start"><strong>Quick Start</strong></a> ·
  <a href="#evaluation"><strong>Evaluation</strong></a>
</p>

## Overview

**CoLM** (**Co**llaboration in **L**arge-**M**odels) is a novel framework designed for realistic **client-server deployment** scenarios, where multiple clients share access to a few powerful LLM servers.

Unlike ensemble or multi-agent LLM setups (server-to-server coordination), CoLM proposes a **client-to-server** collaboration strategy. Server models generate responses that **guide a lightweight client model**, enabling iterative refinement with significantly reduced inference cost.

This paradigm has been extended to both **language** and **vision-language** tasks, and shows consistent improvements on difficult or failed queries.

---

## Quick Start

### 1. Environment Setup

Export your API keys as environment variables:

```bash
export OPENAI_API_KEY=your-openai-key
export DEEPSEEK_API_KEY=your-deepseek-key
export QWEN_API_KEY=your-qwen-key
```

### 2.Run the script

`python CoLM.py`

### 3. Customize your question

You can modify the `question` in the `__main__` block:

```
python
question = "What are the social and technical challenges of AI alignment?"
```

### ⚙️ Parameters

The `main()` function supports the following arguments:

- `question`: Your input question.
- `use_selection`: Whether to use a big model (`gpt-4o`) to select the top-K relevant models.
- `iterations`: Number of refinement rounds.
- `top_k`: Number of models to select (if `use_selection=True`).

### 📌 Example Models

The following specialized models are simulated in this project:

| Name          | Description                                      |
| ------------- | ------------------------------------------------ |
| `qwen-math`   | A math-focused assistant                         |
| `gpt-conv`    | A conversational and emotionally aware assistant |
| `qwen-coder`  | A helpful code assistant                         |
| `ds-creative` | A creative writing-focused assistant             |

These models can be replaced or extended with other endpoints and prompts.

### Example Output

```
>>> Selected Models: ['gpt-conv', 'ds-creative']

>>> gpt-conv initial response: The Alignment Problem by Brian Christian explores...
...

=== Iteration 1: Refinement ===
...

=== Final Model Outputs ===
[gpt-conv]
The book presents real-world risks in AI development...
```

## Evaluation

We provide scripts to quickly reproduce some of the results presented in our paper
For convenience, we have included the code from [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval),
[MT-Bench](https://github.com/lm-sys/FastChat), and [arena-hard](https://https://github.com/lmarena/arena-hard-auto), with necessary modifications.
We extend our gratitude to these projects for creating the benchmarks.

### Preparation

```bash
# install requirements
pip install -r requirements.txt
cd alpaca_eval
pip install -e .
cd FastChat
pip install -e ".[model_worker,llm_judge]"
cd ..

# setup api keys
export TOGETHER_API_KEY=<TOGETHER_API_KEY>
export OPENAI_API_KEY=<OPENAI_API_KEY>
```

### Run AlpacaEval 2

To run AlpacaEval 2, execute the following scripts:

```
bash run_eval_alpaca_eval.sh
```

### Run MT-Bench

For a minimal example of MT-Bench evaluation, run:

```
bash run_eval_mt_bench.sh
```

### Run FLASK

For a minimal example of FLASK evaluation, run:

```
bash run_eval_flask.sh
```

## 

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
