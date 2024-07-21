import spacy
import pandas as pd

from .common import LANG_FULL_NAME_MAP, SYSTEM_PROMPT_TEMPLATE, NO_CONTROL_USER_PROMPT_TEMPLATE, GOE_USER_PROMPT_TEMPLATE


MTGENEVAL_CONTEXTUAL_DATASET_ROOT_PATH = "./data/machine-translation-gender-eval/data/context/"
GENDER_WORD_PAIRS_FILE_PATH = './data/gender_word_pairs.tsv'


# sampled from mtgeneval dev
MTGENEVAL_FEWSHOT_EXAMPLES = {
    "es": [ {
        'data_idx': 315,
        'src_sent1': 'Martin Amis, who was harshly criticized in America Alone but gave it a positive review, said of the style: Mark Steyn is an oddity: his thoughts and themes are sane and serious.',
        'src_sent2': "Longtime editor and admirer Fulford also wrote, Steyn, a self-styled 'right-wing bastard,' violates everyone's sense of good taste.",
        'tgt_sent1': 'Martin Amis, quien fue duramente criticado en America Alone pero le dio una crítica positiva, dijo sobre el estilo: Mark Steyn es una rareza: sus pensamientos y temas son cuerdos y serios.',
        'tgt_sent2': 'Fulford, un admirador y editor de toda la vida, también escribió: "Steyn, un capullo derechista con estilo propio, viola el sentido del buen gusto de cualquiera".',
        'gender_annotation': 'Steyn uses he/him',
    }, {
        'data_idx': 403,
        'src_sent1': 'Indeed, Milutinović introduced Njegoš to his own poetry, which Professor Svetlana Slapšak describes as being written in unusual syntax, with unparalleled neologisms and fantastic etymologies.',
        'src_sent2': "Milaković operated the printing press at Cetinje Monastery, served as editor-in-chief of Grlica and edited all Njegoš's works prior to their publication.",
        'tgt_sent1': 'De hecho, Milutinović introdujo a Njegoš en su propia poesía, que la profesora Svetlana Slapšak describe como escrita con una sintaxis inusual, con neologismos incomparables y etimologías fantásticas.',
        'tgt_sent2': 'Milaković operó la imprenta en el monasterio Cetinje, actuó como editor en jefe de Grlica y editó todas las obras de Njegoš antes de su publicación.',
        'gender_annotation': 'Milaković uses he/him',
    }, {
        'data_idx': 6,
        'src_sent1': "Five of Unaipon's traditional stories were published in 1929 as Native Legends, under his own name and with his picture on the cover.",
        'src_sent2': "They have been republished in their original form, under the author's name, as Legendary Tales of the Australian Aborigines.",
        'tgt_sent1': 'Cinco de los cuentos tradicionales de Unaipon se publicaron en 1929 como Native Legends, bajo su propio nombre y con su foto en la portada.',
        'tgt_sent2': 'Han sido publicados nuevamente en su forma original bajo el nombre del autor, como los Cuentos legendarios de los aborígenes australianos.',
        'gender_annotation': 'Unaipon the author uses he/him',
    }, {
        'data_idx': 188,
        'src_sent1': 'Grote went on to accept a faculty position as an associate professor of chemistry at his alma mater in 1931, and eventually moved on to professor of chemistry in 1941.',
        'src_sent2': 'From 1941 to 1972 Grote also served as scientific advisor for the Chattanooga Medicine Company, later to be called Chattem Co.',
        'tgt_sent1': 'Grote aceptó un puesto en la facultad como profesor asociado de química en su alma mater en 1931 y, finalmente, pasó a profesor de química en 1941.',
        'tgt_sent2': 'De 1941 a 1972 Grote también fue asesor científico para Chattanooga Medicine Company, después llamada Chattem Co.',
        'gender_annotation': 'Grote uses he/him',
    }, {
        'data_idx': 79,
        'src_sent1': "In December she was invited to Garsington Manor, the home of Russell's then mistress Ottoline Morell, and there encountered Clive Bell and other Bloomsbury Group members, and in 1917 she introduced Russell to Dora Black who would later become his second wife.",
        'src_sent2': 'From 1917 Wrinch was funded by Girton College as a research student, officially supervised by G.H.',
        'tgt_sent1': 'En diciembre fue invitada a Garsington Manor, la casa de la entonces amante de Russell, Ottoline Morell, y allí conoció a Clive Bell y otros miembros del Grupo Bloomsbury, y en 1917 le presentó a Russell a Dora Black, quien más tarde se convertiría en su segunda esposa.',
        'tgt_sent2': 'Desde 1967 Wrinch fue financiada por Girton College como estudiante de investigación supervisada oficialmente por G.H.',
        'gender_annotation': 'Wrinch uses she/her'
    } ],
    "fr": [ {
        'data_idx': 55,
        'src_sent1': "She marketed the book among family and friends and sent a copy to her publisher who made numerous cuts in both text and illustrations for the trade edition, chiefly among the tale's many nursery rhymes.",
        'src_sent2': 'Both were published in deluxe editions bound in a flowered chintz of scattered pansies the author selected.',
        'tgt_sent1': "Elle a commercialisé le livre auprès de sa famille et de ses amis et en a envoyé un exemplaire à son éditeur qui a réalisé de nombreuses coupes dans le texte et les illustrations pour l'édition commerciale, principalement parmi les nombreuses comptines du conte.",
        'tgt_sent2': "Les deux furent publiés sous forme d'éditions de luxe reliées dans du chintz fleuri avec un imprimé de pensées disséminées choisi par l'auteure.",
        'gender_annotation': 'the author uses she/her'
    }, {
        'data_idx': 73,
        'src_sent1': 'After several additional visiting professorships at the Technical University of Braunschweig, University of Göttingen, University of Stuttgart, and University of Linz, she settled at the University of Jena until her retirement.',
        'src_sent2': 'Tobies is the author or editor of books including:.',
        'tgt_sent1': "Après plusieurs postes de professeur invité supplémentaires à l'Université technique de Braunschweig, à l'Université de Göttingen, à l'Université de Stuttgart et à l'Université de Linz, elle s'installe à l'Université d'Iéna jusqu'à sa retraite.",
        'tgt_sent2': "Tobies est l'auteur ou éditeur de livres tels que :",
        'gender_annotation': 'Tobies uses she/her',
    }, {
        'data_idx': 33,
        'src_sent1': 'One of her first projects at Adobe was Trajan.',
        'src_sent2': 'As a designer, Twombly closely studied historical scripts for inspiration in creating digital fonts.',
        'tgt_sent1': "L'un de ses premiers projets chez Adobe était Trajan.",
        'tgt_sent2': "En tant que conceptrice, Twombly a étudié de près les anciennes écritures pour s'en servir d'inspiration pour créer des polices numériques.",
        'gender_annotation': 'Twombly the author uses she/her',
    }, {
        'data_idx': 279,
        'src_sent1': 'His two-volume book Cancer: Its Cause and Treatment, was widely reviewed in medical journals.',
        'src_sent2': 'Bulkley was the editor of Cancer: A Practical Quarterly Journal Devoted to the Best Interests of Cancer.',
        'tgt_sent1': 'Son livre en deux volumes Cancer : Its Cause and Treatment, a été largement commenté dans les revues médicales.',
        'tgt_sent2': 'Bulkley était le rédacteur en chef de Cancer: A Practical Quarterly Journal Devoted to the Best Interests of Cancer.',
        'gender_annotation': 'Bulkley uses he/him',
    }, {
        'data_idx': 244,
        'src_sent1': 'Although details of his early life are scant, he lost his father and brother sometime in the Mexican War of Independence and fled to the United States where he became fluent in the English language.',
        'src_sent2': 'Mexía became active in government service as Secretary of State for Tamaulipas and the Tuxpan customs collector.',
        'tgt_sent1': "Bien que les détails de sa jeunesse soient rares, il a perdu son père et son frère au cours de la guerre d'indépendance du Mexique et a fui aux États-Unis où il a appris à parler couramment l'anglais.",
        'tgt_sent2': "Mexía devint actif au sein du gouvernement en tant que Secrétaire d'État pour le Tamaulipas, ainsi que percepteur des douanes de Tuxpan.",
        'gender_annotation': 'Mexía uses he/him'
    } ],
    "it": [ {
        'data_idx': 128,
        'src_sent1': "It's like he's had it – it ain't no fun no more.",
        'src_sent2': 'The curse part of it is the business you have to deal with, and then the blessing part is you get to be a musician and have fun….',
        'tgt_sent1': 'È amico intimo del padre di Harley.[vol. 49:extra] Il suo doppiatore giapponese è Masato Sako e Shinji Ogawa mentre il suo doppiatore inglese è Doug Burks.',
        'tgt_sent2': 'La sua parte maledetta è il business con il quale devi interfacciarti, e invece la parte buona è che puoi essere un musicista e divertirti...',
        'gender_annotation': 'Otaki uses he/him',
    }, {
        'data_idx': 32,
        'src_sent1': 'Settling in Utica, New York, her father became a traveling salesman.',
        'src_sent2': 'Mengers entered the talent agency business in 1955 as a receptionist at MCA.',
        'tgt_sent1': "Sebbene abbia vissuto a Edimburgo per la maggior parte della sua vita, c'erano molti altri luoghi in Scozia che raffigurava regolarmente nei suoi dipinti.",
        'tgt_sent2': 'Mengers è entrata nel business delle agenzie di talenti nel 1955, quando divenne receptionist per la MCA.',
        'gender_annotation': 'Black uses she/her',
    }, {
        'data_idx': 203,
        'src_sent1': 'USS Washakie, a United States Navy harbor tug in service from 1944 to 1946 and from 1953 to 1975, also was named for him.',
        'src_sent2': 'Washakie was a hide painter.',
        'tgt_sent1': "Anne Elwood, dalle sue Memorie di donne letterarie: Era sua invariabile abitudine scrivere nella sua camera da letto, - una stanza dall'aspetto familiare, quasi scomoda, affacciata sulla strada e appena arredata - con un semplice letto bianco, ai piedi di cui c'era una piccola, vecchia specie di toeletta di forma oblunga, completamente coperta da una comune scrivania logora, piena di carte, mentre alcune erano sparse per terra, poiché il tavolo era troppo piccolo per qualcos'altro oltre alla scrivania.",
        'tgt_sent2': 'Washakie era un pittore di pelli.',
        'gender_annotation': 'the author uses she/her',
    }, {
        'data_idx': 444,
        'src_sent1': 'In the novel, a young woman lawyer goes to court to reclaim land for a native Band, but from the outset of the trial, things go badly, and a disturbing level of confrontation builds.',
        'src_sent2': 'When the young lawyer and the aging judge journey back to their childhoods, it becomes clear that the courtroom drama merely brushes the surface of both wider and more personal dramas.',
        'tgt_sent1': 'Nello show, oltre alle commedie e agli ospiti famosi, Pocher si è offerto di affittarsi a uno spettatore.',
        'tgt_sent2': "Quando la giovane avvocata e l'anziano giudice ripercorrono la loro infanzia, diventa chiaro che il dramma giudiziario rappresenti solo la punta dell'iceberg, e che sotto la superficie si celino drammi più profondi e personali.",
        'gender_annotation': 'Pocher uses he/him',
    }, {
        'data_idx': 158,
        'src_sent1': 'A hockey player and his friend Jay MacDonald were inspired to bring a team to St.',
        'src_sent2': 'Pilous would be coach & general manager of the team, and MacDonald would be secretary & treasurer.',
        'tgt_sent1': 'Weizmann rispose sei mesi dopo, sostenendo il suo desiderio di ricevere un posto di patologo ma spiegando che prima era necessario istituire strutture amministrative.',
        'tgt_sent2': "Pilous sarebbe stato l'allenatore e il manager della squadra, mentre MacDonald avrebbe svolto le funzioni di segretario e tesoriere.",
        'gender_annotation': 'Getzowa uses she/her'
    } ],
}


FEWSHOT_FIRST_USER_PROMPT_TEMPLATE = "Help me translate the following source text into {lang_name}."
FEWSHOT_FIRST_ASSISTANT_PROMPT = "Sure, I'd be happy to!"
FEWSHOT_USER_PROMPT_TEMPLATE = "{sentence}"
NO_CONTROL_FEWSHOT_ASSISTANT_PROMPT_TEMPLATE = "The {lang_name} translation with correct gender inflection is:\n\n{tgt_sent1} {tgt_sent2}"  # ADD_EXP 1 FEW SHOT BASELINE
IGOE_FEWSHOT_ASSISTANT_PROMPT_TEMPLATE = "From the given source text, we can infer that {gender_annotation}. Therefore, the {lang_name} translation with correct gender inflection is:\n\n{tgt_sent1} {tgt_sent2}"


def _get_fewshot_messages(lang, control):
    lang_name = LANG_FULL_NAME_MAP[lang]
    if control == "none_fewshot":
        assistant_prompt_template = NO_CONTROL_FEWSHOT_ASSISTANT_PROMPT_TEMPLATE
    elif control == "igoe_fewshot":
        assistant_prompt_template = IGOE_FEWSHOT_ASSISTANT_PROMPT_TEMPLATE

    fewshot_messages = [ {
            "role": "user",
            "content": FEWSHOT_FIRST_USER_PROMPT_TEMPLATE.format(lang_name=lang_name)
        }, {
            "role": "assistant",
            "content": FEWSHOT_FIRST_ASSISTANT_PROMPT
        } ]

    for ex in MTGENEVAL_FEWSHOT_EXAMPLES[lang]:
        sentence = " ".join([ex['src_sent1'], ex['src_sent2']])
        fewshot_messages.extend([ {
            'role': 'user',
            'content': FEWSHOT_USER_PROMPT_TEMPLATE.format(sentence=sentence),
        }, {
            'role': 'assistant',
            'content': assistant_prompt_template.format(
                lang_name=lang_name,
                gender_annotation=ex['gender_annotation'],
                tgt_sent1=ex['tgt_sent1'],
                tgt_sent2=ex['tgt_sent2'])
        } ])
    return fewshot_messages


def _get_gender_related_sets_and_maps():
    df_gender_word_list = pd.read_csv(GENDER_WORD_PAIRS_FILE_PATH, sep='\t', header=None, names=['m', 'f'])

    male_word_set = set(df_gender_word_list.m.tolist())
    female_word_set = set(df_gender_word_list.f.tolist())

    flip_gender_map = {}
    for _, row in df_gender_word_list.iterrows():
        if row.m in flip_gender_map:
            pass
            # print(f"Tried adding '{row.m}'->'{row.f}', but already found '{row.m}'->'{flip_gender_map[row.m]}'")
        else:
            flip_gender_map[row.m] = row.f


        if row.f in flip_gender_map:
            pass
            # print(f"Tried adding '{row.f}'->'{row.m}', but already found '{row.f}'->'{flip_gender_map[row.f]}'")
        else:
            flip_gender_map[row.f] = row.m
        if row.m == 'guy':
            flip_gender_map['guy'] = 'girl'
        elif row.m == 'guys':
            flip_gender_map['guys'] = 'girls'
        elif row.f == 'lady':
            flip_gender_map['lady'] = 'gentleman'
        elif row.f == 'ladies':
            flip_gender_map['ladies'] = 'gentlemen'
        elif row.m == 'sir':
            flip_gender_map['sir'] = 'miss'
        elif row.m == 'mr.':
            flip_gender_map['mr.'] = 'ms.'
        elif row.m == 'mr':
            flip_gender_map['mr'] = 'ms'
    return male_word_set, female_word_set, flip_gender_map


def load_df(lang, split):
    src_contextual = []
    with open(MTGENEVAL_CONTEXTUAL_DATASET_ROOT_PATH + f"geneval-context-wikiprofessions-2to1-{split}.en_{lang}.en") as f:
        for line in f:
            src_contextual.append(line.strip())

    tgt_original = []
    with open(MTGENEVAL_CONTEXTUAL_DATASET_ROOT_PATH + f"geneval-context-wikiprofessions-original-{split}.en_{lang}.{lang}") as f:
        for line in f:
            tgt_original.append(line.strip())

    tgt_flipped = []
    with open(MTGENEVAL_CONTEXTUAL_DATASET_ROOT_PATH + f"geneval-context-wikiprofessions-flipped-{split}.en_{lang}.{lang}") as f:
        for line in f:
            tgt_flipped.append(line.strip())

    df = pd.DataFrame({
        "src": src_contextual,
        "tgt_original": tgt_original,
        "tgt_flipped": tgt_flipped,
    })

    def split_sents(row):
        sents = row['src'].split('<sep>')
        assert len(sents) == 2, sents
        sents_formatted = []
        for sent in sents:
            sent = sent.strip()
            if len(sent) > 2:
                if sent[-1] != "." and sent[-2:] != '.”' and sent[-2:] != ".'"  and sent[-2:] != '."':
                    sent += "."
            sents_formatted.append(sent)
        return sents_formatted

    df[['src_sent1', 'src_sent2']] = df.apply(split_sents, axis=1, result_type='expand').copy()

    num_empty_sents = (df['src_sent1'].map(len) == 0).sum()
    if num_empty_sents > 0:
        print(f"Skipping {num_empty_sents} samples with empty first sentence...")
        df = df[df['src_sent1'].map(len) > 0]
    num_empty_sents = (df['src_sent2'].map(len) == 0).sum()
    if num_empty_sents > 0:
        print(f"Skipping {num_empty_sents} samples with empty second sentence...")
        df = df[df['src_sent2'].map(len) > 0]

    nlp = spacy.load("en_core_web_lg")

    def _extract_entities_with_spacy(sent):
        doc = nlp(sent)
        entities = []
        for token in doc:
            if token.pos_ == "PROPN" and token.dep_ == "nsubj":
                entities.append(str(token))
        if len(entities) == 0:
            for token in doc:
                if token.pos_ == "PRON" and token.dep_ == "nsubj":
                    entities.append(str(token))
        if len(entities) == 0:
            for token in doc:
                if token.pos_ == "NOUN" and token.dep_ == "nsubj":
                    entities.append(str(token))
        if len(entities) == 0:
            for token in doc:
                if token.dep_ == "nsubjpass":
                    entities.append(str(token))
        return entities

    df['extracted_entities'] = df.src_sent2.map(_extract_entities_with_spacy).copy()

    male_word_set, female_word_set, flip_gender_map = _get_gender_related_sets_and_maps()

    def _extract_gender_label_via_spacy(sent):
        sent_tokens = {token.lemma_.lower() if token.tag_ in {"NNS", "NNPS"} else token.text.lower() for token in nlp(sent)}
        gender_label = None
        if len(sent_tokens.intersection(male_word_set)) > 0:
            gender_label = 'm'
        if len(sent_tokens.intersection(female_word_set)) > 0:
            if gender_label is not None:
                gender_label = 'ambig'
            else:
                gender_label = 'f'
        if gender_label is not None:
            return gender_label
        return 'unknown'
    df['gender_label'] = df.src_sent1.map(_extract_gender_label_via_spacy).copy()

    def _flip_sent_gender(sent):
        modified = ""
        for token in nlp(sent):
            check_tok = token.lemma_.lower() if token.tag_ in {"NNS", "NNPS"} else token.text.lower()
            if check_tok in male_word_set or check_tok in female_word_set:
                if check_tok == "her":
                    if token.dep_ == "poss":
                        new_tok = "his"
                    else:
                        new_tok = "him"
                else:
                    new_tok = flip_gender_map[token.text.lower()]

                if token.text.isupper():
                    new_tok = new_tok.upper()
                elif token.text[0].isupper():
                    new_tok = new_tok.capitalize()
    #             modified += f"[{new_tok}]"
                modified += new_tok
                modified += token.whitespace_
            else:
                modified += token.text_with_ws

        return modified
    df['flipped_src_sent1'] = df.src_sent1.map(_flip_sent_gender).copy()

    def _flip_gender_label(gender_label):
        if gender_label == "m":
            return "f"
        if gender_label == "f":
            return "m"
        return gender_label
    df['flipped_gender_label'] = df.gender_label.map(_flip_gender_label).copy()

    print(f"Skipping {len(df[df['gender_label'] == 'unknown'])} sentences with unidentified gender...")
    df = df.query('gender_label != "unknown"')
    return df


def construct_openai_messages(df, lang, control):
    assert control in {'none', 'goe', 'none_fewshot', 'igoe_fewshot'}

    lang_name = LANG_FULL_NAME_MAP[lang]

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(lang_name=lang_name)
    if control == "none":
        fewshot_messages = None
        user_prompt_template = NO_CONTROL_USER_PROMPT_TEMPLATE
    elif control == "goe":
        fewshot_messages = None
        user_prompt_template = GOE_USER_PROMPT_TEMPLATE
    elif control in {"none_fewshot", "igoe_fewshot"}:
        fewshot_messages = _get_fewshot_messages(lang, control)
        user_prompt_template = FEWSHOT_USER_PROMPT_TEMPLATE

    messages_batch = []
    for _, row in df.iterrows():
        sentence = " ".join([row['src_sent1'], row['src_sent2']])
        if row['gender_label'] == 'ambig':
            gender_annotation = ""
        else:
            if row['gender_label'] == 'm':
                gender = "he/him"
            else:
                gender = "she/her"
            if len(row['extracted_entities']) == 1:
                gender_annotation = f"for \"{row['extracted_entities'][0]}\", use {gender}"
            else:  # fallback if spacy failed to extract entity from text
                gender_annotation = f"use \"{gender}\" gender inflection"

        user_prompt_kwargs = {
            'lang_name': lang_name,
            'sentence': sentence,
            'gender_annotation': gender_annotation,
        }

        messages = [
            {"role": "system", "content": system_prompt},
        ]
        if fewshot_messages is not None:
            messages.extend(fewshot_messages)
        if control == "goe" and gender_annotation == "":
            # for goe prompting of ambiguous gender label, just use default template
            messages.append(
                {"role": "user", "content": NO_CONTROL_USER_PROMPT_TEMPLATE.format(**user_prompt_kwargs)},
            )
        else:
            messages.append(
                {"role": "user", "content": user_prompt_template.format(**user_prompt_kwargs)},
            )
        messages_batch.append(messages)

    return messages_batch
