import os
import json
import logging
import re
import time
import requests
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ===============================
# CONFIGURATION
# ===============================
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

HF_MODEL_URL = config.get("hf_model_url")
MAX_NEW_TOKENS = config.get("max_new_tokens", 1000)
TEMPERATURE = config.get("temperature", 0.7)
HISTORY_FILE = config.get("history_file", "conversations.json")
KNOWLEDGE_FILE = config.get("knowledge_file", "knowledge_base.json")

# Paramètres pour la complétion du dernier mot
COMPLETION_MAX_NEW_TOKENS = config.get("completion_max_new_tokens", 20)
COMPLETION_TEMPERATURE = config.get("completion_temperature", 0.2)
MIN_WORD_LENGTH_FOR_TRUNCATION = config.get("min_word_length_for_truncation", 2)
MIN_TOTAL_LENGTH_FOR_TRUNCATION = config.get("min_total_length_for_truncation", 40)

HF_API_KEY = os.getenv("HF_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

if not HF_API_KEY or not SERPAPI_API_KEY:
    raise ValueError("Les variables HF_API_KEY et SERPAPI_API_KEY doivent être définies.")

HEADERS_HF = {"Authorization": f"Bearer {HF_API_KEY}"}

# ===============================
# LOGGING
# ===============================
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])

# ===============================
# HISTORIQUE & BASE DE CONNAISSANCES
# ===============================
conversation_history = []
if os.path.exists(HISTORY_FILE):
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            conversation_history = json.load(f)
    except Exception as e:
        logging.warning(f"Impossible de charger l'historique: {e}")

if os.path.exists(KNOWLEDGE_FILE):
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        knowledge_base = json.load(f)
else:
    knowledge_base = {}

def save_history():
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(conversation_history, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde de l'historique: {e}")

def save_knowledge():
    try:
        with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
            json.dump(knowledge_base, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde de la base de connaissances: {e}")

# ===============================
# UTIL: détection mot tronqué & complétion
# ===============================
def is_last_word_truncated(text: str, min_word_length: int = MIN_WORD_LENGTH_FOR_TRUNCATION, min_total_length: int = MIN_TOTAL_LENGTH_FOR_TRUNCATION) -> bool:
    """
    Heuristique pour détecter si le dernier mot semble tronqué.
    - Retourne True si :
      * le texte ne se termine pas par ponctuation (.!?…),
      * le texte est suffisamment long (>= min_total_length),
      * et le dernier token alphanumérique est très court (<= min_word_length),
      ou si on remarque un mélange lettre+chiffre incomplet à la fin.
    """
    if not text or not isinstance(text, str):
        return False

    t = text.strip()

    # Si terminaison claire => pas tronqué
    if t.endswith(('.', '!', '?', '…', ';', ':')):
        return False

    # Si texte court -> tolérer
    if len(t) < min_total_length:
        return False

    # tokens alphanumériques
    tokens = re.findall(r"\w+", t, flags=re.UNICODE)
    if not tokens:
        return False

    last = tokens[-1]

    # Si dernier token est très court -> suspect
    if len(last) <= min_word_length:
        logging.debug(f"Détection truncation: dernier token court '{last}' (len={len(last)})")
        return True

    # Si mélange lettre+chiffre à la fin -> suspect
    if re.search(r"[A-Za-z]\d$|\d[A-Za-z]$", last):
        logging.debug(f"Détection truncation: mélange suspect dans '{last}'")
        return True

    return False

def complete_last_word(hf_model_url: str, headers: dict, original_prompt: str, partial_text: str, max_new_tokens: int = COMPLETION_MAX_NEW_TOKENS) -> str | None:
    """
    Demande au modèle de compléter uniquement le dernier mot du texte partiel.
    Retourne la complétion (chaîne) ou None en cas d'échec.
    """
    # Construire un prompt court demandant uniquement la fin du dernier mot
    completion_prompt = (
        f"{original_prompt}\n\n"
        f"Le texte suivant s'est arrêté en plein mot. Complète uniquement le dernier mot pour que la phrase soit lisible.\n\n"
        f"Texte : \"{partial_text}\"\n\n"
        "Réponds uniquement par la suite nécessaire pour compléter le dernier mot (ne répète pas tout le texte)."
    )

    payload = {
        "inputs": completion_prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": COMPLETION_TEMPERATURE,
            "top_p": 0.9,
            "do_sample": False
        }
    }

    try:
        resp = requests.post(hf_model_url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # extraction robuste
        if isinstance(data, list) and "generated_text" in data[0]:
            cont = data[0]["generated_text"]
        elif isinstance(data, dict) and "generated_text" in data:
            cont = data.get("generated_text", "")
        else:
            logging.warning(f"Format inattendu de complétion last word: {data}")
            return None

        cont = cont.strip()

        # Si le modèle renvoie le texte complet, essayer d'extraire seulement la portion en plus
        # Cherche la première occurrence de partial_text dans cont
        if cont.startswith(partial_text):
            extra = cont[len(partial_text):].lstrip()
            return extra if extra else None
        else:
            # Le modèle a renvoyé juste la suite — retourne tel quel
            return cont
    except Exception as e:
        logging.warning(f"Échec complétion du dernier mot: {e}")
        return None

# ===============================
# FONCTIONS D'APPEL HF / SERPAPI
# ===============================
def query_hf(prompt: str) -> str:
    """Interroge le modèle Hugging Face choisi et retourne le texte brut généré."""
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": 0.9,
            "do_sample": True
        }
    }
    try:
        response = requests.post(HF_MODEL_URL, headers=HEADERS_HF, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        # Selon le format renvoyé par le modèle
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        elif isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        elif isinstance(data, dict) and "error" in data:
            logging.error(f"Erreur API Hugging Face : {data['error']}")
            return "Erreur lors de la génération de texte."
        else:
            logging.warning(f"Réponse inattendue Hugging Face : {data}")
            # Essayer de retourner une représentation textuelle si possible
            try:
                return str(data)
            except Exception:
                return "Désolé, je n'ai pas pu générer de réponse."
    except requests.Timeout:
        logging.error("⏳ Timeout Hugging Face")
        return "Le serveur Hugging Face met trop de temps à répondre."
    except requests.RequestException as e:
        logging.error(f"⚠️ Erreur réseau Hugging Face: {e}")
        return "Erreur de connexion à Hugging Face."
    except Exception as e:
        logging.error(f"⚠️ Erreur inattendue Hugging Face: {e}")
        return "Une erreur est survenue lors du traitement."

def search_serapi(query: str) -> str:
    """Recherche sur le web via SerpAPI"""
    url = "https://serpapi.com/search"
    params = {"q": query, "hl": "fr", "gl": "fr", "api_key": SERPAPI_API_KEY}
    try:
        logging.info(f"🔍 Requête SerpAPI : {query}")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        result = (
            data.get("answer_box", {}).get("answer") or
            data.get("answer_box", {}).get("snippet") or
            (data.get("organic_results") and data["organic_results"][0].get("snippet")) or
            ""
        )
        logging.info(f"✅ Résultat SerpAPI : {result}")
        return result
    except requests.Timeout:
        logging.error("⚠️ Timeout SerpAPI")
        return "Erreur : le serveur de recherche met trop de temps à répondre."
    except requests.RequestException as e:
        logging.error(f"⚠️ Erreur réseau SerpAPI : {e}")
        return "Erreur de connexion à SerpAPI."
    except Exception as e:
        logging.error(f"⚠️ Erreur inattendue SerpAPI : {e}")
        return "Erreur inattendue lors de la recherche."

# ===============================
# ROUTES FLASK
# ===============================
@app.route("/", methods=["GET"])
def home():
    return render_template("mesgbairai.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "").strip()
    username = data.get("username", "Utilisateur")

    if not user_input:
        return jsonify({"error": "Message invalide"}), 400

    conversation_history.append({"role": "user", "name": username, "content": user_input})

    # 1️⃣ Vérifier base de connaissances
    if user_input.lower() in knowledge_base:
        answer = knowledge_base[user_input.lower()]
        source = "exact"
    else:
        # 2️⃣ Rechercher sur le web
        web_answer = search_serapi(user_input)
        if web_answer and web_answer.strip() != "":
            # si SerpAPI retourne un texte trop court ou une erreur string, on laisse au modèle AI
            # ici on considère web_answer valide si ce n'est pas un message d'erreur
            if web_answer.startswith("Erreur"):
                web_answer = ""  # ignorer ce résultat pour fallback IA
            else:
                answer = web_answer
                source = "web"

        # si web_answer n'a pas fourni de contenu utile, on génère via HF
        if not ('answer' in locals() and answer):
            history_text = "\n".join([f"{msg['name']} ({msg['role'].capitalize()}): {msg['content']}" 
                                      for msg in conversation_history])
            prompt = f"Tu es un assistant en Côte d'Ivoire qui lutte contre la désinformation. Réponds de façon factuelle et claire.\n{history_text}\nAssistant:"
            # génération initiale
            generated = query_hf(prompt)
            source = "ai"

            # Si la réponse ressemble à un message d'erreur renvoyé par query_hf -> on l'utilise telle quelle
            if generated.startswith("Erreur") or generated.startswith("Désolé"):
                answer = generated
            else:
                # vérification du mot tronqué
                if is_last_word_truncated(generated):
                    logging.info("Mot tronqué détecté — tentative de complétion ciblée du dernier mot.")
                    extra = complete_last_word(HF_MODEL_URL, HEADERS_HF, prompt, generated, max_new_tokens=COMPLETION_MAX_NEW_TOKENS)
                    if extra:
                        # Joindre proprement
                        # si generated ne se termine pas par espace, assurer séparation correcte
                        if not generated.endswith(" "):
                            generated = generated + ""
                        generated = (generated + extra).strip()
                        # s'assurer de ponctuation finale
                        if not generated.endswith(('.', '?', '!', '…', ';', ':')):
                            generated = generated + "."
                    else:
                        logging.info("Aucune complétion trouvée pour le dernier mot.")
                # finaliser la réponse
                answer = generated

    conversation_history.append({"role": "assistant", "name": "Assistant", "content": answer})
    save_history()

    return jsonify({"response": answer, "source": source})

@app.route("/teach", methods=["POST"])
def teach():
    data = request.json
    question = data.get("question", "").strip()
    answer = data.get("answer", "").strip()

    if not question or not answer:
        return jsonify({"error": "Question et réponse requises"}), 400

    knowledge_base[question.lower()] = answer
    save_knowledge()

    return jsonify({"message": "Nouvelle connaissance enregistrée ✅"})

# ===============================
# LANCEMENT
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
