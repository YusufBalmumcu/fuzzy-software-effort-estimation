import re


RULE_PATTERN = re.compile(
    r"^\s*IF\s+(?P<body>.*?)\s+THEN\s+\(\s*(?P<output_var>[\w.]+)\s+IS\s+(?P<label>[\w_]+)\s*\)\s*$",
    re.IGNORECASE,
)
CONDITION_PATTERN = re.compile(r"\(\s*(?P<variable>[\w.]+)\s+IS\s+(?P<term>[\w_]+)\s*\)")


def parse_rule(rule_text):
    match = RULE_PATTERN.match(rule_text)
    if not match:
        raise ValueError(f"Kural formati okunamadi: {rule_text}")

    body = match.group("body").strip()
    output_var = match.group("output_var").strip()
    original_label = match.group("label").strip()
    conditions = [
        {"variable": item.group("variable").strip(), "term": item.group("term").strip()}
        for item in CONDITION_PATTERN.finditer(body)
    ]

    if not conditions:
        raise ValueError(f"Kural kosullari okunamadi: {rule_text}")

    return {
        "antecedent": f"IF {body}",
        "output_var": output_var,
        "original_label": original_label,
        "conditions": conditions,
    }


def convert_rules_to_rule_level(raw_rules, input_vars=None):
    """
    Klasik etiket seviyeli Sugeno kurallarini kural seviyeli Sugeno kurallarina cevirir.

    Eski yaklasimda birden fazla kural ayni cikti etiketini kullanir:
    THEN (Effort IS Low)

    Tam kural seviyeli Sugeno'da her kuralin kendine ait cikti terimi vardir:
    THEN (Effort IS R1_OUT)

    Bu sayede Very_Low/Low/Medium/High/Very_High icin 5 denklem yerine,
    her kural icin ayri bir birinci derece denklem ogrenilir.
    """
    converted = []
    expected_vars = set(input_vars or [])

    for idx, rule_text in enumerate(raw_rules, start=1):
        parsed = parse_rule(rule_text)
        rule_id = f"R{idx}"
        sugeno_output = f"{rule_id}_OUT"
        variables = {condition["variable"] for condition in parsed["conditions"]}

        if expected_vars and not variables.issubset(expected_vars):
            missing = sorted(variables - expected_vars)
            raise ValueError(f"{rule_id} beklenmeyen degiskenler iceriyor: {missing}")

        converted.append({
            "rule_id": rule_id,
            "original_rule": rule_text,
            "antecedent": parsed["antecedent"],
            "conditions": parsed["conditions"],
            "original_label": parsed["original_label"],
            "sugeno_output": sugeno_output,
            "converted_rule": f"{parsed['antecedent']} THEN ({parsed['output_var']} IS {sugeno_output})",
        })

    return converted
