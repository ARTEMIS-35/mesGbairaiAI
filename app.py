import os
import json
import logging
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
# FONCTIONS
# ===============================
def query_hf(prompt: str) -> str:
    """Interroge le modèle Hugging Face DeepSeek"""
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE
        }
    }
    try:
        response = requests.post(HF_MODEL_URL, headers=HEADERS_HF, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        else:
            logging.warning(f"Réponse inattendue du modèle: {data}")
            return "Désolé, je n'ai pas pu générer de réponse."
    except requests.RequestException as e:
        logging.error(f"Erreur réseau Hugging Face: {e}")
        return "Erreur de connexion au serveur Hugging Face."
    except Exception as e:
        logging.error(f"Erreur inattendue Hugging Face: {e}")
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
        if web_answer:
            answer = web_answer
            source = "web"
        else:
            # 3️⃣ Générer via DeepSeek
            history_text = "\n".join([f"{msg['name']} ({msg['role'].capitalize()}): {msg['content']}" 
                                      for msg in conversation_history])
            prompt = f"Tu es un assistant en Côte d'Ivoire qui lutte contre la désinformation.\n{history_text}\nAssistant:"
            answer = query_hf(prompt)
            source = "ai"

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
