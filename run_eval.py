import argparse

import jsonlines
from spacy.lang.es import Spanish
from spacy.lang.it import Italian
from spacy.lang.fr import French

from evaluation import gate_eval, mustshe_eval, mtgeneval_eval
from evaluation.postprocessing import process_llama2_fewshot_translation_output, process_llama2_zeroshot_translation_output, \
    process_llm_fewshot_translation_output, process_llm_zeroshot_translation_output
from preprocessing import gate_dataset, mustshe_dataset, mtgeneval_dataset


def get_spacy_model(lang):
    # spacy tokenizer and sentencizer
    if lang == "es":
        nlp = Spanish()
    elif lang == "it":
        nlp = Italian()
    elif lang == "fr":
        nlp = French()
    nlp.add_pipe("sentencizer")
    return nlp


def print_results_table(results):
    columns = ['type', 'cov', 'acc']
    for r in results:
        assert set(r.keys()) == set(columns)

    print("{:14} {:>7} {:>7}".format(*columns))
    print('-' * (14+8+8))
    for res in results:
        print("{:14} {:>7.2%} {:>7.2%}".format(*[res[c] for c in columns]))


def main(args):
    if args.split is not None:
        output_file = f"../out/{args.dataset}_{args.lang}_{args.split}_{args.control}_{args.model}_outputs.jsonl"
    else:
        output_file = f"../out/{args.dataset}_{args.lang}_{args.control}_{args.model}_outputs.jsonl"
    print("Eval file:", output_file)

    print("Loading data...")
    with jsonlines.open(output_file) as reader:
        outputs = list(reader)

    # clean-up model output to contain only translation
    if args.model == "llama2":
        if args.control.endswith('fewshot'):
            preds = process_llama2_fewshot_translation_output(outputs)
        else:
            preds = process_llama2_zeroshot_translation_output(outputs)
    else:
        if args.control.endswith('fewshot'):
            preds = process_llm_fewshot_translation_output(outputs)
        else:
            preds = process_llm_zeroshot_translation_output(outputs)

    # get spacy tokenizer
    nlp = get_spacy_model(args.lang)

    # load dataset and run evaluation
    if args.dataset == "gate":
        df = gate_dataset.load_df(args.lang, args.split)
        results = gate_eval.evaluate(df, preds, nlp, args.control)
    elif args.dataset == "mustshe":
        df = mustshe_dataset.load_df(args.lang)
        results = mustshe_eval.evaluate(df, preds, nlp)
    elif args.dataset == "mtgeneval":
        df = mtgeneval_dataset.load_df(args.lang, args.split)
        results = mtgeneval_eval.evaluate(df, preds, nlp)

    # print results table
    print("Dataset: {} / Language: {} / Control: {}".format(args.dataset, args.lang, args.control))
    print_results_table(results)
    for r in results:
        r['dataset'] = args.dataset
        r['lang'] = args.lang
        r['control'] = args.control

    # save eval results
    with jsonlines.open("eval_results_3.jsonl", "a") as writer:
        writer.write_all(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--control", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--split", type=str)
    args = parser.parse_args()
    main(args)
