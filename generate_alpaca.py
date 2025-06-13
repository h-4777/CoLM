import os
import random
import time
import json
from openai import OpenAI
from filelock import FileLock
from concurrent.futures import ThreadPoolExecutor
from datasets import load_from_disk

def call_model(model_source, messages, clients):
    time.sleep(random.uniform(0.3, 0.8))
    try:
        client = clients[model_source]
        response = client.chat.completions.create(
            model=model_source,
            messages=messages,
            max_tokens=1000,
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Model call failed for {model_source}: {e}")
        return ""

def select_models(client, model_id, selection_prompt, question, specializations, top_k):
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": selection_prompt},
                {"role": "user", "content": f"Given the question: '{question}', select the {top_k} most relevant specializations from the following:\n\n" +
                                                "\n".join(specializations.keys()) + "\n\nReturn only the names of the most relevant models, separated by commas."}
            ]
        )
        selected = response.choices[0].message.content.strip().split(", ")
        print(f"\n>>> Selected Models: {selected}")
        return selected
    except Exception as e:
        print(f"Model selection failed: {e}")
        return []

def get_model_responses(selected, model_prompts, model_sources, question, clients):
    responses = {}
    for model_name in selected:
        model_id = model_sources.get(model_name, "gpt-4o")
        prompt = model_prompts[model_name]
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ]
        response = call_model(model_id, messages, clients)
        responses[model_name] = response
        print(f"\n>>> {model_name} initial response: {response[:100]}...")
    return responses

def summarize_all(client, model_id, responses, summary_prompt):
    combined = "\n\n".join([f"{name}:\n{resp}" for name, resp in responses.items()])
    messages = [
        {"role": "system", "content": summary_prompt},
        {"role": "user", "content": f"Here are multiple responses:\n\n{combined}"}
    ]
    response = client.chat.completions.create(
        model=model_id,
        messages=messages
    )
    return response.choices[0].message.content.strip()

def refine_responses(
    clients, selected_models, model_prompts, model_sources,
    initial_responses, summary_prompt, question, iterations
):
    current_responses = initial_responses.copy()
    for i in range(iterations):
        print(f"\n=== Iteration {i + 1}: Refinement ===")
        summary = summarize_all(
            client=clients["gpt-4o"],
            model_id="gpt-4o",
            responses=current_responses,
            summary_prompt=summary_prompt
        )

        updated_responses = {}
        for model_name, prompt in model_prompts.items():
            model_id = model_sources.get(model_name, "gpt-4o")
            messages = [
                {"role": "system", "content": prompt},
                {"role": "system", "content": (
                    f"Here is a refined summary of previous answers:\n\n{summary}\n\n"
                    f"{'Your perspective is relevant.' if model_name in selected_models else 'Your perspective is less relevant.'}\n"
                    "Please revise your previous answer based on this summary and the question."
                )},
                {"role": "user", "content": question}
            ]
            response = call_model(model_id, messages, clients)
            updated_responses[model_name] = response
            print(f"\n>>> {model_name} refined response (iteration {i + 1}): {response[:100]}...")

        current_responses = updated_responses
    return current_responses

MODEL_PROMPTS = {
    "qwen-math": "You are a helpful math assistant.",
    "gpt-conv": "You are a conversational assistant focused on natural, fluent communication.",
    "qwen-coder": "You are a helpful code assistant.",
    "ds-creative": "You are a creative assistant who writes imaginatively and artistically."
}

MODEL_SOURCES = {
    "qwen-math": "qwen-math-plus",
    "gpt-conv": "gpt-4o",
    "qwen-coder": "qwen-coder-plus",
    "ds-creative": "deepseek-chat"
}

SUMMARY_PROMPT = (
    "Please summarize the following answers by:\n"
    "- Removing redundancy\n"
    "- Keeping important insights\n"
    "- Improving clarity and coherence\n"
    "- Returning a concise but complete synthesis"
)

CLIENTS = {
    "gpt-4o": OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
    "deepseek-chat": OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"]),
    "qwen-math-plus": OpenAI(api_key=os.environ["QWEN_API_KEY"]),
    "qwen-coder-plus": OpenAI(api_key=os.environ["QWEN_API_KEY"])
}


def process_instruction(item, output_path):
    instr = item["instruction"]
    dataset_name = item.get("dataset", "alpaca_eval")
    selection_prompt = (
    "You are an AI assistant that selects the most relevant model specializations for a given question. "
    "Only return model names, separated by commas."
    )

    selected_models = select_models(
        client=CLIENTS["gpt-4o"],
        model_id="gpt-4o",
        selection_prompt=selection_prompt,
        question=instr,
        specializations=MODEL_PROMPTS,
        top_k=2  
    )

    initial_responses = get_model_responses(
        selected_models, MODEL_PROMPTS, MODEL_SOURCES, instr, CLIENTS
    )
    final_responses = refine_responses(
        CLIENTS, selected_models, MODEL_PROMPTS, MODEL_SOURCES,
        initial_responses, SUMMARY_PROMPT, instr, iterations=2
    )

    for model_name in selected_models:
        model_output_file = os.path.join(output_path, f"{model_name}.json")
        lock_file = model_output_file + ".lock"
        os.makedirs(output_path, exist_ok=True)

        with FileLock(lock_file):
            if os.path.exists(model_output_file):
                with open(model_output_file, "r", encoding="utf-8") as f:
                    try:
                        existing_data = json.load(f)
                    except json.JSONDecodeError:
                        existing_data = []
            else:
                existing_data = []

            existing_data.append({
                "dataset": dataset_name,
                "generator": model_name,
                "instruction": instr,
                "output": final_responses[model_name]
            })

            with open(model_output_file, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)


def evaluate_and_save(output_path):
    eval_set = load_from_disk("./alpaca_eval")["eval"]
    eval_set = eval_set.remove_columns(["output", "generator"])
    print(f"Loaded {len(eval_set)} items from alpaca_eval.")

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(process_instruction, item, output_path) for item in eval_set]
        for future in futures:
            future.result()  

if __name__ == "__main__":
    output_path = "outputs_eval"
    evaluate_and_save(output_path)