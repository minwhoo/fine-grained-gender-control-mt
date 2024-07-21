import pandas as pd

from .common import LANG_FULL_NAME_MAP, SYSTEM_PROMPT_TEMPLATE, NO_CONTROL_USER_PROMPT_TEMPLATE, GOE_USER_PROMPT_TEMPLATE


WINOMT_DATASET_PATH = "./data/mt_gender/data/aggregates/en.txt"
WINOMT_SECONDARY_DATASET_PATH = "./data/winomt_secondary/en.txt"


def load_df():
    df = pd.read_csv(WINOMT_SECONDARY_DATASET_PATH, sep='\t', header=None, names=['gender', 'index', 'sent', 'secondary_entity'])
    df_primary = pd.read_csv(WINOMT_DATASET_PATH, sep='\t', header=None, names=['gender', 'index', 'sent', 'entity'])

    df.loc[:,'primary_entity'] = df_primary['entity'].copy()

    assert all(df.secondary_entity != df.primary_entity)

    df = df.query('gender != "neutral"')
    return df


def construct_openai_messages(df, lang, control):
    assert control in {'none', 'goe_ambig', 'goe_full'}

    lang_name = LANG_FULL_NAME_MAP[lang]

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(lang_name=lang_name)
    if control == "none":
        user_prompt_template = NO_CONTROL_USER_PROMPT_TEMPLATE
        gender_annotation_template = ""
    elif control == "goe_ambig":
        user_prompt_template = GOE_USER_PROMPT_TEMPLATE
        gender_annotation_template = "for {secondary_entity}, use {secondary_gender_pronoun}"
    elif control == "goe_full":
        user_prompt_template = GOE_USER_PROMPT_TEMPLATE
        gender_annotation_template = "for {primary_entity}, use {primary_gender_pronoun}; for {secondary_entity}, use {secondary_gender_pronoun}"

    messages_batch = []
    for _, row in df.iterrows():
        assert row['gender'] in {'male', 'female'}
        gender_annotation = gender_annotation_template.format(
            primary_entity=row['primary_entity'],
            primary_gender_pronoun="he/him" if row['gender'] == 'male' else "she/her",
            secondary_entity=row['secondary_entity'],
            secondary_gender_pronoun="he/him" if row['gender'] == 'female' else "she/her",
        )
        user_prompt_kwargs = {
            'lang_name': lang_name,
            'sentence': row['sent'],
            'gender_annotation': gender_annotation,
        }

        messages_batch.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_template.format(**user_prompt_kwargs)},
        ])

    return messages_batch
