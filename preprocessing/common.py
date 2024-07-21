LANG_FULL_NAME_MAP = {"es": "spanish", "fr": "french", "it": "italian"}
SYSTEM_PROMPT_TEMPLATE = "You are a professional {lang_name} translator that especially considers translating gender inflections correctly."
NO_CONTROL_USER_PROMPT_TEMPLATE = "Translate the following sentence into {lang_name}: {sentence}"
GOE_USER_PROMPT_TEMPLATE = "Translate the following sentence into {lang_name} ({gender_annotation}): {sentence}"