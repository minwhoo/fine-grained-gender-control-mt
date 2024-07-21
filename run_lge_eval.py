import time
import argparse

import openai
import jsonlines
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, cohen_kappa_score


LGE_SYSTEM_PROMPT = """You are evaluating a gender-conditioned translation. \
Please specifically focus on whether the translation accurately reflects the gender representation of the provided entities. \
Check if the words related to the entities are translated in a way that is consistent with the entities' specified genders. \
After reviewing the input, provide your evaluation in the following format:
Comment: [Your explanation regarding the gender representation in relation to the entities in the translation.]
Gender Accuracy: [ACCURATE or INACCURATE]."""

LGE_USER_PROMPT_TEMPLATE = """Source [EN]: {src}
Condition: Entity '{entity}' should be translated as {gender}
Translation [{lang_code}]: {model_pred}
"""


def openai_batch_request(messages_batch, openai_model):
    client = openai.OpenAI()
    responses = []

    cnt = 0
    for messages in tqdm(messages_batch):
        while True:
            try:
                response = client.chat.completions.create(
                    model=openai_model,
                    messages=messages,
                    temperature=0,
                )
            except openai.APITimeoutError:
                print("Timeout! Waiting for 1 min and trying again")
                time.sleep(60)
            except Exception as e:
                print(e)
                print("Unhandled exception")
                output = [ r.to_dict()['choices'][0]['message']['content'] for r in responses ]
                return output
            else:
                break

        # print first few examples
        if cnt == 0:
            print("#### MODEL OUTPUT EXAMPLES #####")
        if cnt < 2:
            print(response.to_dict()['choices'][0]['message']['content'])
        if cnt == 2:
            print("################################")

        cnt += 1
        responses.append(response)

    resp_dict = [r.to_dict() for r in responses]

    output = []
    for r in resp_dict:
        out = r['choices'][0]['message']['content']
        output.append(out)
    return output


def construct_openai_messages(df, lang):
    messages_batch = []

    for _, row in df.iterrows():
        user_prompt_args = {
            'src': " ".join([row['src_sent1'], row['src_sent2']]),
            'entity': row['extracted_entities'],
            'lang_code': lang.upper(),
            'model_pred': row['pred'],
            'gender': 'male' if row['gender_label'] == 'm' else 'female',
        }
        messages_batch.append([
            { "role": "system", "content": LGE_SYSTEM_PROMPT },
            { "role": "user", "content": LGE_USER_PROMPT_TEMPLATE.format(**user_prompt_args)},
        ])
    return messages_batch


def parse_eval_results_from_out(model_out):
    scores = []
    for out in model_out:
        comment = None
        score_text = None
        score_value = None

        for line in out.split('\n'):
            line = line.strip()
            if line.startswith('Comment: '):
                comment = line[len('Comment: '):].strip()
            elif line.startswith('Gender Accuracy: '):
                score_text = line[len('Gender Accuracy: '):].strip()
        if score_text == 'ACCURATE':
            score_value = 1
        elif score_text == 'INACCURATE':
            score_value = 0
        else:
            # fallback heuristic for detecting model judgement
            if 'inaccurate' in out.lower():
                score_value = 0
            elif 'accurate' in out.lower():
                score_value = 1
        scores.append({
            'comment': comment,
            'score_text': score_text,
            'score_value': score_value
        })
    return scores


def compute_valid_accuracy(preds, labels):
    num_correct = 0
    num_total = 0
    for p, l in zip(preds, labels):
        if p is not None:
            num_total += 1
            if p == l:
                num_correct += 1
    if num_total == 0:
        return 0
    return num_correct / num_total


def main(args):
    # load data
    df = pd.read_csv("./human_annotations/mtgeneval_es_scores.csv")

    # construct messages
    messages_batch = construct_openai_messages(df, lang='es')

    # llm inference
    outputs = openai_batch_request(messages_batch, args.openai_model)

    # save model outputs
    with jsonlines.open(f'./out/lge_eval_{args.openai_model}.jsonl', 'w') as writer:
        writer.write_all(outputs)

    # parse model outputs
    eval_results = parse_eval_results_from_out(outputs)
    model_scores = [s['score_value'] for s in eval_results]

    # check parsing-failed outputs and filter them out
    no_scores_cnt = model_scores.count(None)
    if no_scores_cnt > 0:
        print(f"Parsing failed for {no_scores_cnt} outputs. Skipping them for evaluation")
    preds = [s for s in model_scores if s is not None]
    labels = [l for s, l in zip(model_scores, df.human_score) if s is not None]

    # compute scores
    acc = accuracy_score(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    print(f"Agreement Rate: {acc:.2%}")
    print(f"Cohen's Kappa : {kappa:.3}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--openai_model", type=str, default="gpt-4-1106-preview")
    args = parser.parse_args()
    main(args)
