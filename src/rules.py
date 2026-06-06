import json
import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


LLM_FILE_ALIASES = {
    "gpt": "chatgpt",
    "chatgpt": "chatgpt",
    "gemini": "gemini",
    "claude": "claude",
}


def load_rules(file_path, dataset_name):
    """
    Belirtilen JSON dosyasından ilgili veri setine ait kuralları okur.
    """
    if not os.path.exists(file_path):
        print(f"[HATA] Kural dosyası bulunamadı: {file_path}")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rules = data.get(dataset_name.lower(), [])
    return rules


def get_all_rules(dataset_name, llm_name="gemini"):
    """
    Örnek: get_all_rules('albrecht', 'chatgpt') -> models/rules_chatgpt.json içindeki 'albrecht' dizisi.
    'gpt' adı da mevcut dosya adıyla uyumlu olması için 'chatgpt' olarak ele alınır.
    """
    base_dir = "models"
    normalized_llm_name = LLM_FILE_ALIASES.get(llm_name.lower(), llm_name.lower())
    file_name = f"rules_{normalized_llm_name}.json"
    file_path = os.path.join(base_dir, file_name)

    return load_rules(file_path, dataset_name)


if __name__ == "__main__":
    # Test amaçlı
    rules = get_all_rules("albrecht", "gemini")
    print(f"Gemini Albrecht için {len(rules)} kural yüklendi.")
    for r in rules[:3]:
        print(" -", r)
