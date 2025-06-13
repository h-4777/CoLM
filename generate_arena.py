import os
import json
import time
import random
import shortuuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from loguru import logger
from openai import OpenAI
from tqdm import tqdm

def load_existing_instructions(answer_file):
    existing_instructions = set()
    if os.path.exists(answer_file):
        with open(answer_file, "r", encoding="utf-8") as f:
            for line in f:
                result = json.loads(line)
                existing_instructions.add(result["question_id"])
    return existing_instructions

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

def get_model_responses(selected, model_prompts, model_sources, questions, clients):
    responses = {}
    for model_name in selected:
        model_id = model_sources.get(model_name, "gpt-4o")
        prompt = model_prompts[model_name]
        messages = [{"role": "system", "content": prompt}]

        for turn in questions["turns"]:
            messages.append({"role": "user", "content": turn})
            response = call_model(model_id, messages, clients)
            responses[model_name].append(response)
            messages.append({"role": "assistant", "content": response})
        print(f"\n>>> {model_name} initial response: {response[:100]}...")
    return responses

file_lock = Lock()


def summarize_responses(client, responses, big_model_prompt, questions):
    summarized_turns = []
    for turn_idx, turn in enumerate(questions["turns"]):
        turn_responses = [f"{m}: {r[turn_idx]}" for m, r in responses.items() if turn_idx < len(r)]
        if not turn_responses:
            continue

        summary_prompt = f"""
        Round {turn_idx + 1} question: {turn}
        Here are responses from different models:
        {'\n\n'.join(turn_responses)}
        Please synthesize and refine...
        """

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": big_model_prompt},
                {"role": "user", "content": summary_prompt}
            ]
        )
        summarized_turns.append(completion.choices[0].message.content)
    return summarized_turns

def generate_final_responses(clients, model_sources, small_model_prompts, summary_responses, item, output_path="./output"):
    
    final_responses = {model_name: [] for model_name in small_model_prompts}
    os.makedirs(output_path, exist_ok=True)  

    for model_name, prompt in small_model_prompts.items():
        model_source = model_sources.get(model_name, "gpt-4o") 
        messages = [{"role": "system", "content": prompt}]
        messages.append({
            "role": "system",
            "content": (
                "You are an expert assistant responsible for refining your own previous response using feedback from other model outputs.\n\n"
                "Your goals:\n"
                "- Carefully analyze the provided summary of strong model responses.\n"
                "- Identify concrete ways to improve factual accuracy, clarity, and reasoning depth.\n"
                "- Revise your answer to better meet the user's original request, while preserving your unique style and domain expertise.\n\n"
                "Requirements:\n"
                "- Your updated response must fully address the user's original question.\n"
                "- Maintain logical flow and coherence across turns.\n"
                "- Be specific, accurate, and helpful.\n"
                "- Output only the final improved answer — do not include any explanations, commentary, or headings.\n\n"
                "Only return the refined answer text."
            )
})

        for turn_idx, turn in enumerate(item["turns"]):
            summary_response = summary_responses[turn_idx] if turn_idx < len(summary_responses) else ""

            messages.append({
                "role": "user",
                "content": f"""
                    ### Round {turn_idx + 1}:
                    #### Question:
                    {turn}

                    #### Summary of the best response:
                    {summary_response}

                    Using the above summary, revise your original answer.  
                    Ensure the updated response:
                    - Maintains dialogue flow and consistency with previous turns  
                    - Preserves factual accuracy  
                    - Aligns in tone and reasoning style with the summary  

                    **Output only the final improved answer. Do not repeat the question or summary.**
                    """
            })

            try:
                response = call_model(
                model_source=model_source,
                messages=messages,
                clients=clients
            )
                final_responses[model_name].append(response)
                messages.append({"role": "assistant", "content": response})


            except Exception as e:
                print(f"{model_name} Model call failed for ( turn {turn_idx + 1}）:", str(e))
        print(final_responses[model_name])

        model_output_file = os.path.join(output_path, f"{model_name}.jsonl")

        with open(model_output_file, "a", encoding="utf-8") as f:
            json_line = json.dumps({
                "question_id": item["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_name,
                "choices": [{"index": 0, "turns": final_responses[model_name]}],
                "tstamp": time.time(),
            }, ensure_ascii=False)
            f.write(json_line + "\n")
    return final_responses


TOP_K = 2
MAX_WORKERS = 32

def process_single_question(question, clients, prompts, sources):
    big_prompt = "You are a helpful assistant."
    try:
        selected = select_models(clients["gpt-4o"], "gpt-4o", big_prompt, question, prompts, TOP_K)
        responses = get_model_responses(selected, prompts, sources, question, clients)
        summary = summarize_responses(clients["gpt-4o"], responses, big_prompt, question)
        generate_final_responses(clients, sources, prompts, summary, question, "./output")
        return question["question_id"], "success"
    except Exception as e:
        return question["question_id"], f"failed: {e}"

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


CLIENTS = {
    "gpt-4o": OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
    "deepseek-chat": OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"]),
    "qwen-math-plus": OpenAI(api_key=os.environ["QWEN_API_KEY"]),
    "qwen-coder-plus": OpenAI(api_key=os.environ["QWEN_API_KEY"])
}


def main():

    question_file = r"./arena_hard_question.jsonl"
    with open(question_file, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]


    logger.info(f"Starting concurrent processing of {len(questions)} items...")


    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(process_single_question, q, CLIENTS, MODEL_PROMPTS, MODEL_SOURCES)
            for q in questions
        ]
        with tqdm(total=len(questions), desc="Processing") as pbar:
            for future in as_completed(futures):
                qid, status = future.result()
                pbar.update(1)

if __name__ == "__main__":
    main()
