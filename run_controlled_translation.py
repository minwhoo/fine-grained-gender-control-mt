import os
import json
import time
import argparse

import openai
import jsonlines
from tqdm import tqdm

from preprocessing import mustshe_dataset, gate_dataset, winomt_dataset, mtgeneval_dataset


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


def main(args):
    print("Language:", args.lang)
    print("Dataset:", args.dataset)
    print("Gender Control:", args.control)
    print("OpenAI Model:", args.openai_model)

    # check if given argument combination is valid
    assert args.dataset in {'winomt', 'mtgeneval', 'mustshe', 'gate'}
    if args.dataset in {'mtgeneval', 'gate'}:
        assert args.split is not None
        print("Split:", args.split)
    else:
        assert args.split is None

    # construct run id used for naming output files
    if args.split is not None:
        run_id = f"{args.dataset}_{args.lang}_{args.split}_{args.control}_{args.openai_model}"
    else:
        run_id = f"{args.dataset}_{args.lang}_{args.control}_{args.openai_model}"

    # load dataset
    if args.dataset == "mustshe":
        df = mustshe_dataset.load_df(args.lang)
    elif args.dataset == "gate":
        df = gate_dataset.load_df(args.lang, args.split)
    elif args.dataset == "mtgeneval":
        df = mtgeneval_dataset.load_df(args.lang, args.split)
    elif args.dataset == "winomt":
        df = winomt_dataset.load_df()
    print(f"Loaded dataset of size: {len(df)}")

    # construct prompts
    if args.dataset == "mustshe":
        assert args.control in {'none', 'goe'}
        messages_batch = mustshe_dataset.construct_openai_messages(df, args.lang, args.control)
    elif args.dataset == "gate":
        assert args.control in {'none', 'goe'}
        messages_batch = gate_dataset.construct_openai_messages(df, args.lang, args.control)
    elif args.dataset == "mtgeneval":
        assert args.control in {'none', 'goe', 'none_fewshot', 'igoe_fewshot'}
        messages_batch = mtgeneval_dataset.construct_openai_messages(df, args.lang, args.control)
    elif args.dataset == "winomt":
        assert args.control in {'none', 'goe_ambig', 'goe_full'}
        messages_batch = winomt_dataset.construct_openai_messages(df, args.lang, args.control)

    print(f"Number of inference samples: {len(messages_batch)}")
    print("#### INPUT MESSAGE EXAMPLES ####")
    print(json.dumps(messages_batch[0], indent=4))
    print(json.dumps(messages_batch[1], indent=4))
    print("################################")

    # check files
    out_dir = "../out"
    messages_file = f'../out/{run_id}_messages.jsonl'
    outputs_file = f'../out/{run_id}_outputs.jsonl'
    if os.path.exists(messages_file):
        print("Found existing messages file")

        with jsonlines.open(messages_file, 'r') as reader:
            existing_messages_batch = list(reader)
        if messages_batch != existing_messages_batch:
            print("file length doesn't match! aborting...")
            return

        # if output file exists, check if the output data is the same size as the input data
        if os.path.exists(outputs_file):
            with jsonlines.open(outputs_file, 'r') as reader:
                existing_outputs = list(reader)
            if len(existing_outputs) == len(messages_batch):
                print("Already completed run! Aborting...")
                return
            else:
                # continue translating remaining inputs, assuming previous run has aborted due to request error
                print("Skipping first {} messages due to already run output...".format(len(existing_outputs)))
                messages_batch = messages_batch[len(existing_outputs):]
    else:
        if os.path.exists(outputs_file):
            print("Messages file not found but outputs file exists! Aborting...")
            return
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        with jsonlines.open(messages_file, 'w') as writer:
            writer.write_all(messages_batch)

    # request translation with openai api
    outputs = openai_batch_request(messages_batch, args.openai_model)

    # save output
    with jsonlines.open(outputs_file, 'a') as writer:
        writer.write_all(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--control", type=str, required=True)
    parser.add_argument("--split", type=str)
    parser.add_argument("--openai_model", type=str, default="gpt-3.5-turbo")
    args = parser.parse_args()
    main(args)
