import pandas as pd

from .common import SYSTEM_PROMPT_TEMPLATE, LANG_FULL_NAME_MAP, NO_CONTROL_USER_PROMPT_TEMPLATE, GOE_USER_PROMPT_TEMPLATE


MUSTSHE_DATASET_PATH_TEMPLATE = "./data/MuST-SHE_v1.2.1/MuST-SHE-v1.2.1-data/tsv/MONOLINGUAL.{lang}_v1.2.1.tsv"


def load_df(lang):
    df = pd.read_csv(MUSTSHE_DATASET_PATH_TEMPLATE.format(lang=lang), sep='\t')

    # get ambiguous, non-gender-mixed subset from the dataset
    df_ambig = df.query('CATEGORY == "1F" | CATEGORY == "1M"').query('GENDER != "Multi-Mix"')

    # standardize gender label
    def get_gender_label(gender):
        if "She" in gender:
            return "female"
        if "He" in gender:
            return "male"
        return "unknown"

    df_ambig["gender_label"] = df_ambig.GENDER.map(get_gender_label).copy()
    assert df_ambig.gender_label.nunique() == 2

    # parse gendered tokens from gender term annotations
    def get_gendered_tokens(row):
        correct_tokens = []
        wrong_tokens = []
        for p in row['GENDERTERMS'].split(';'):
            c, w = p.split(' ')
            correct_tokens.append(c)
            wrong_tokens.append(w)
        assert len(set(correct_tokens).intersection(wrong_tokens)) == 0
        return correct_tokens, wrong_tokens

    df_ambig[['gender_tokens', 'wrong_gender_tokens']] = df_ambig.apply(get_gendered_tokens, axis=1, result_type='expand').copy()

    return df_ambig


def construct_openai_messages(df, lang, control):
    assert control in {"none", "goe"}

    lang_name = LANG_FULL_NAME_MAP[lang]

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(lang_name=lang_name)
    if control == "none":
        user_prompt_template = NO_CONTROL_USER_PROMPT_TEMPLATE
        gender_annotation_template = ""
    elif control == "goe":
        user_prompt_template = GOE_USER_PROMPT_TEMPLATE
        gender_annotation_template = "the speaker is {gender}"

    messages_batch = []
    for _, row in df.iterrows():
        gender_annotation = gender_annotation_template.format(gender=row['gender_label'])
        user_prompt_kwargs = {
            'lang_name': lang_name,
            'sentence': row['SRC'],
            'gender_annotation': gender_annotation,
        }

        messages_batch.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_template.format(**user_prompt_kwargs)},
        ])

    return messages_batch
