import os
import random
import time
from openai import OpenAI


# -----------------------------
# Call model
# -----------------------------
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


# -----------------------------
# Step 1: Model Selection (optional)
# -----------------------------
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


# -----------------------------
# Step 2: Get initial responses
# -----------------------------
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


# -----------------------------
# Step 3: Summarize all responses
# -----------------------------
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


# -----------------------------
# Step 4: Refinement
# -----------------------------
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

        current_responses = updated_responses  # update for next round

    return current_responses


# -----------------------------
# Main 
# -----------------------------
def main(
    question,
    use_selection=True,
    iterations=2,
    top_k=2
):
    model_prompts = {
        "qwen-math": "You are a helpful math assistant.",
        "gpt-conv": "You are a conversational assistant focused on natural, fluent communication.",
        "qwen-coder": "You are a helpful code assistant.",
        "ds-creative": "You are a creative assistant who writes imaginatively and artistically."
    }

    model_sources = {
        "qwen-math": "qwen-math-plus",
        "gpt-conv": "gpt-4o",
        "qwen-coder": "qwen-coder-plus",
        "ds-creative": "deepseek-chat"
    }

    summary_prompt = (
        "Please summarize the following answers by:\n"
        "- Removing redundancy\n"
        "- Keeping important insights\n"
        "- Improving clarity and coherence\n"
        "- Returning a concise but complete synthesis"
    )

    clients = {
        "gpt-4o": OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
        "deepseek-chat": OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"]),
        "qwen-math-plus": OpenAI(api_key=os.environ["QWEN_API_KEY"]),
        "qwen-coder-plus": OpenAI(api_key=os.environ["QWEN_API_KEY"])
    }

    if use_selection:
        selected = select_models(
            client=clients["gpt-4o"],
            model_id="gpt-4o",
            selection_prompt=summary_prompt,
            question=question,
            specializations=model_prompts,
            top_k=top_k
        )
    else:
        selected = list(model_prompts.keys())

    initial_responses = get_model_responses(
        selected=selected,
        model_prompts=model_prompts,
        model_sources=model_sources,
        question=question,
        clients=clients
    )

    final_responses = refine_responses(
        clients=clients,
        selected_models=selected,
        model_prompts=model_prompts,
        model_sources=model_sources,
        initial_responses=initial_responses,
        summary_prompt=summary_prompt,
        question=question,
        iterations=iterations
    )

    print("\n=== Final Model Outputs ===")
    for model, response in final_responses.items():
        print(f"\n[{model}]\n{response}\n")


if __name__ == "__main__":
    sample_question = (
        "Explain the book 'The Alignment Problem' by Brian Christian. "
        "Provide a synopsis of themes and analysis. "
        "Recommend a bibliography of related reading."
    )
    main(
        question=sample_question,
        use_selection=True,  # Set to False to let all models participate
        iterations=2,
        top_k=2
    )