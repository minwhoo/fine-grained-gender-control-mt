import re

import pandas as pd

from .common import LANG_FULL_NAME_MAP, SYSTEM_PROMPT_TEMPLATE, NO_CONTROL_USER_PROMPT_TEMPLATE, GOE_USER_PROMPT_TEMPLATE


GATE_DATASET_PATH_TEMPLATE = "./data/GATE/data/{lang}/{lang}_{num_gender_combinations}_variants.tsv"
ENTITY_EXTRACT_REGEX_PTN = r'\[([^\]]+)\]'


def load_df(lang, split):
    df_all = []
    num_agmes = [1,2,3]
    if lang == "fr":
        # no agme=3 data available for french
        num_agmes = [1,2]

    for num_agme in num_agmes:
        df = pd.read_csv(GATE_DATASET_PATH_TEMPLATE.format(lang=lang.upper(), num_gender_combinations=2**num_agme), sep='\t')
        if lang == "fr":
            # manual split for french due to missing annotation
            if split == "dev":
                df = df[:len(df)//2]
            else:
                df = df[len(df)//2:]
        else:
            df = df.query(f'data_split == "{split}"').copy()

        def _extract_entity_words(kw_s):
            entities = []
            for annotation in kw_s.split(';'):
                matches = re.findall(ENTITY_EXTRACT_REGEX_PTN, annotation)
                if matches:
                    entity = matches[0]
                elif '=' in annotation:
                    entity = annotation.split('=')[0]
                else:
                    entity = annotation
                entities.append(entity)
            return entities

        df.loc[:,'entities'] = df.kw_s.map(_extract_entity_words).copy()
        df.loc[:,'num_agme'] = num_agme
        num_entity_agme_mismatch = sum(df['entities'].map(len) != num_agme)
        if num_entity_agme_mismatch > 0:
            print(f"for {num_agme} agme subset, {num_entity_agme_mismatch} samples mismatching entity are skipped")
        df = df[df['entities'].map(len) == num_agme]
        assert all(df['entities'].map(len) == num_agme)

        df_all.append(df)
    df = pd.concat(df_all)
    return df


def get_entity_gender_combinations(entities, gender_pronouns):
    assert len(gender_pronouns) == 2
    all_combs = []
    for idx in range(2**len(entities)):
        comb = []
        cur = idx
        for ent in reversed(entities):
            comb.append((ent, gender_pronouns[cur % 2]))
            cur //= 2
        all_combs.append(list(reversed(comb)))
    return all_combs


def construct_openai_messages(df, lang, control):
    assert control in {"none", "goe"}

    lang_name = LANG_FULL_NAME_MAP[lang]

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(lang_name=lang_name)
    if control == "none":
        user_prompt_template = NO_CONTROL_USER_PROMPT_TEMPLATE
    elif control == "goe":
        user_prompt_template = GOE_USER_PROMPT_TEMPLATE

    messages_batch = []
    for _, row in df.iterrows():
        if control == "none":
            user_prompt_kwargs = {
                'lang_name': lang_name,
                'sentence': row['source'],
            }

            messages_batch.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_template.format(**user_prompt_kwargs)},
            ])
        elif control == "goe":
            combs = get_entity_gender_combinations(row.entities, ['he/him', 'she/her'])
            for comb in combs:
                gender_annotation = "; ".join([f'for "{e}", use {g}' for e, g in comb])
                user_prompt_kwargs = {
                    'lang_name': lang_name,
                    'sentence': row['source'],
                    'gender_annotation': gender_annotation,
                }

                messages_batch.append([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt_template.format(**user_prompt_kwargs)},
                ])

    return messages_batch
