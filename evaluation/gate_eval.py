import pandas as pd

from preprocessing.gate_dataset import get_entity_gender_combinations
from evaluation.common import compute_cov_acc, compute_gender_match_df


def _parse_gendered_terms(gender_annotation):
    # parse gender terms from keyword annotations in GATE dataset
    terms = []
    for terms_per_entity_str in gender_annotation.split(';'):
        terms_per_entity = []
        for term in terms_per_entity_str.split('='):
            terms_per_entity.append(term.replace('[', '').replace(']', ''))
        terms.append(terms_per_entity)
    return terms


def _get_exclusive_gendered_terms(row):
    exclusive_terms_m = []
    exclusive_terms_f = []
    assert len(row.terms_m) == len(row.terms_f)
    for terms_per_entity_m, terms_per_entity_f in zip(row.terms_m, row.terms_f):
        assert len(terms_per_entity_m) == len(terms_per_entity_f)
        exclusive_terms_per_entity_m = []
        exclusive_terms_per_entity_f = []
        for term_m, term_f in zip(terms_per_entity_m, terms_per_entity_f):
            if term_m != term_f:
                exclusive_terms_per_entity_m.append(term_m)
                exclusive_terms_per_entity_f.append(term_f)
        exclusive_terms_m.append(exclusive_terms_per_entity_m)
        exclusive_terms_f.append(exclusive_terms_per_entity_f)

    return exclusive_terms_m, exclusive_terms_f


def prepare_eval_df(df, tokenize_fn):

    df['terms_m'] = df['kw_m'].map(_parse_gendered_terms)
    df['terms_f'] = df['kw_f'].map(_parse_gendered_terms)

    df[['terms_m', 'terms_f']] = df.apply(_get_exclusive_gendered_terms, axis=1, result_type='expand').copy()

    df['terms_m'] = df['terms_m'].map(lambda terms: [[tokenize_fn(term) for term in terms_per_entity] for terms_per_entity in terms] )
    df['terms_f'] = df['terms_f'].map(lambda terms: [[tokenize_fn(term) for term in terms_per_entity] for terms_per_entity in terms] )

    # prepare gender labels for each target gender combination
    eval_list = []
    for _, row in df.iterrows():
        combs = get_entity_gender_combinations(row.entities, ['m', 'f'])
        for comb in combs:
            val = row.to_dict()
            assert len(val['terms_m']) == len(val['terms_f']) == len(comb)
            val['correct_gender_tokens'] = []
            val['wrong_gender_tokens'] = []
            val['entity_indices'] = []
            val['gender_labels'] = []
            for i, (m, f, (e, g)) in enumerate(zip(val['terms_m'], val['terms_f'], comb)):
                assert len(f) == len(m)
                val['correct_gender_tokens'].extend(m if g == 'm' else f)
                val['wrong_gender_tokens'].extend(f if g == 'm' else m)
                val['entity_indices'].extend([i] * len(f))
                val['gender_labels'].extend([g] * len(f))
            eval_list.append(val)
    df_eval = pd.DataFrame(eval_list)
    return df_eval


def evaluate(df, preds, nlp, control):
    def _tokenize(sent):
        return [token.text.lower() for token in nlp(sent)]

    df_eval = prepare_eval_df(df, _tokenize)

    # extend count of non-controlled predictions to match gender combinations
    if control == "none":
        assert len(df) == len(preds)
        extended_preds = []
        for i, (_, row) in enumerate(df.iterrows()):
            combs = get_entity_gender_combinations(row.entities, ['m', 'f'])
            for _ in combs:
                extended_preds.append(preds[i])
        preds = extended_preds

    assert len(df_eval) == len(preds)

    df_eval['pred_token'] = [_tokenize(p) for p in preds]

    df_score = compute_gender_match_df(df_eval)

    results = []

    cov, acc = compute_cov_acc(df_score)
    results.append({"cov": cov, "acc": acc, "type": "overall"})
    cov, acc = compute_cov_acc(df_score.query('num_agme == 1'))
    results.append({"cov": cov, "acc": acc, "type": "num_agme == 1"})
    cov, acc = compute_cov_acc(df_score.query('num_agme >= 2'))
    results.append({"cov": cov, "acc": acc, "type": "num_agme >= 2"})
    cov, acc = compute_cov_acc(df_score.query('gender_label == "m"'))
    results.append({"cov": cov, "acc": acc, "type": "m"})
    cov, acc = compute_cov_acc(df_score.query('gender_label == "f"'))
    results.append({"cov": cov, "acc": acc, "type": "f"})

    return results
