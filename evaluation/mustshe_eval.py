from evaluation.common import compute_gender_match_df, compute_cov_acc


def prepare_eval_df(df):
    df['correct_gender_tokens'] = df['gender_tokens'].map(lambda tokens: [[t.lower()] for t in tokens])
    df['wrong_gender_tokens'] = df['wrong_gender_tokens'].map(lambda tokens: [[t.lower()] for t in tokens])
    return df


def evaluate(df, preds, nlp):
    def _tokenize(sent):
        return [token.text.lower() for token in nlp(sent)]

    df_eval = prepare_eval_df(df)

    df_eval['pred_token'] = [_tokenize(p) for p in preds]

    df_score = compute_gender_match_df(df_eval)

    results = []

    cov, acc = compute_cov_acc(df_score)
    results.append({"cov": cov, "acc": acc, "type": "overall"})
    cov, acc = compute_cov_acc(df_score.query('gender_label == "male"'))
    results.append({"cov": cov, "acc": acc, "type": "m"})
    cov, acc = compute_cov_acc(df_score.query('gender_label == "female"'))
    results.append({"cov": cov, "acc": acc, "type": "f"})

    return results
