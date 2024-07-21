import pandas as pd


def find_phrase_in_sent(phrase_tokens, sent_tokens, search_start_idx=0):
    l = len(phrase_tokens)
    for i in range(search_start_idx, len(sent_tokens)):
        if sent_tokens[i:i+l] == phrase_tokens:
            return i
    return None


def find_new_phrase_in_sent(phrase_token, sent_token, matched_indices):
    idx = find_phrase_in_sent(phrase_token, sent_token)
    while idx is not None:
        if idx not in matched_indices:
            return True, idx
        idx = find_phrase_in_sent(phrase_token, sent_token, search_start_idx=idx+1)
    return False, None


def compute_gender_match_df(df):
    results = []
    skipped = 0
    for sample_idx, (original_idx, row) in enumerate(df.iterrows()):
        # skip incorrectly annotated samples
        if len(row['correct_gender_tokens']) != len(row['wrong_gender_tokens']):
            skipped += 1
            continue

        # check correct gender tokens and incorrect gender tokens exists
        matched_indices = []
        for term_idx, (correct_tokens, wrong_tokens) in enumerate(zip(row['correct_gender_tokens'], row['wrong_gender_tokens'])):
            ct_found, ct_idx = find_new_phrase_in_sent(correct_tokens, row['pred_token'], matched_indices)
            if ct_found:
                matched_indices.append(ct_idx)

            wt_found, wt_idx = find_new_phrase_in_sent(wrong_tokens, row['pred_token'], matched_indices)
            if wt_found:
                matched_indices.append(wt_idx)

            r = {
                "sample_idx": sample_idx,
                "original_idx": original_idx,
                "term_idx": term_idx,
                'correct_term_found': ct_found,
                'wrong_term_found': wt_found,
                'gender_label': row['gender_labels'][term_idx] if 'gender_labels' in row else row['gender_label'],
            }
            if 'num_agme' in row:
                r['num_agme'] = row['num_agme']
            if 'entity_idx' in row:
                r['entity_idx'] = row['entity_idx'][term_idx]
            results.append(r)

    if skipped > 0:
        print(f'WARNING: skipped {skipped} samples due to asymmetric m/f gender annotations')
    return pd.DataFrame(results)


def compute_cov_acc(df_result):
    cov = (df_result['correct_term_found'] | df_result['wrong_term_found']).mean()
    acc_v1 = df_result['correct_term_found'].sum() / (df_result['correct_term_found'].sum() + df_result['wrong_term_found'].sum())
#     acc_v2 = df_result.query('correct_mention_found | wrong_mention_found')['correct_mention_found'].mean()
    return cov, acc_v1


def compute_sample_level_cov_acc(df_result):
    df_result = df_result[['sample_idx', 'correct_term_found', 'wrong_term_found']].copy()
    df_result['term_found'] = df_result['correct_term_found'] | df_result['wrong_term_found']
    df_result_by_sample = df_result.groupby(['sample_idx']).mean()

    cov = (df_result_by_sample.term_found == 1).mean()
    acc = ((df_result_by_sample['correct_term_found'] == 1) & (df_result_by_sample['wrong_term_found'] == 0)).sum() / (df_result_by_sample.term_found == 1).sum()
    return cov, acc
