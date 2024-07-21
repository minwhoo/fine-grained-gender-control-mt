import difflib

from evaluation.common import find_phrase_in_sent, compute_gender_match_df, compute_cov_acc


def _compute_diff(tokens_a, tokens_b):
    def _make_phrase_token_exclusive(phrase_token, tokens_orig, tokens_oppo, phrase_last_idx):
        orig_phrase_token = phrase_token
        offset = 0
        while find_phrase_in_sent(phrase_token, tokens_oppo) is not None:
            offset += 1
            phrase_token.append(tokens_orig[phrase_last_idx+offset])
            if offset >= 2:
                return orig_phrase_token
        return phrase_token

    matcher = difflib.SequenceMatcher(a=tokens_a, b=tokens_b)

    idx_a = 0
    idx_b = 0
    phrases_a = []
    phrases_b = []

    for match in matcher.get_matching_blocks():
        phrase_a = [tokens_a[idx] for idx in range(idx_a, match.a) if tokens_a[idx] != '.']
        if len(phrase_a) > 0:
            phrase_a = _make_phrase_token_exclusive(phrase_a, tokens_a, tokens_b, match.a - 1)
            phrases_a.append(phrase_a)
        idx_a = match.a + match.size

        phrase_b = [tokens_b[idx] for idx in range(idx_b, match.b) if tokens_b[idx] != '.']
        if len(phrase_b) > 0:
            phrase_b = _make_phrase_token_exclusive(phrase_b, tokens_b, tokens_a, match.b - 1)
            phrases_b.append(phrase_b)

        idx_b = match.b + match.size
    return phrases_a, phrases_b


def prepare_eval_df(df, tokenize_fn):
    correct_gender_tokens, wrong_gender_tokens = [], []
    for sent1, sent2 in zip(df['tgt_original'].map(tokenize_fn).tolist(), df['tgt_flipped'].map(tokenize_fn).tolist()):
        t1, t2 = _compute_diff(sent1, sent2)
        correct_gender_tokens.append(t1)
        wrong_gender_tokens.append(t2)
    df['correct_gender_tokens'] = correct_gender_tokens
    df['wrong_gender_tokens'] = wrong_gender_tokens
    # df[['correct_gender_tokens', 'wrong_gender_tokens']] = df.apply(compute_diff, axis=1, result_type='expand').copy()
    return df


def evaluate(df, preds, nlp):
    def _tokenize(sent):
        return [token.text.lower() for token in nlp.tokenizer(sent)]

    df_eval = prepare_eval_df(df, _tokenize)

    # only evaluate gender accuracy of second output sentence
    preds = [str(list(nlp(p).sents)[-1]) for p in preds]

    df_eval['pred_token'] = [_tokenize(p) for p in preds]

    df_score = compute_gender_match_df(df_eval)

    results = []

    cov, acc = compute_cov_acc(df_score)
    results.append({"cov": cov, "acc": acc, "type": "overall"})
    cov, acc = compute_cov_acc(df_score.query('gender_label == "m"'))
    results.append({"cov": cov, "acc": acc, "type": "m"})
    cov, acc = compute_cov_acc(df_score.query('gender_label == "f"'))
    results.append({"cov": cov, "acc": acc, "type": "f"})

    return results
