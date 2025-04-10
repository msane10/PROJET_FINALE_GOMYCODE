# === PHASE 4 : Phase de laboratoire / Performance ===
# Objectif : L'objectif de votre présentation doit être de présenter au public
# le concept de chatbots de traitement du langage naturel (NLP) avec NLTK pour la classification de texte.

# === CLASSBOT - INTERFACE UTILISATEUR ===

import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
import os
import requests

# ⛳ DOIT ÊTRE TOUT EN HAUT juste après les imports
st.set_page_config(page_title="ClassBot", page_icon="🤖")

# Configuration NLP
nltk.download('stopwords')
stop_words = list(stopwords.words('french'))
stemmer = FrenchStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ0-9\s'?-]", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)

# Téléchargement des modèles depuis GitHub si nécessaire
def download_model_from_github():
    # Liste des fichiers à télécharger
    files = [
        ("model.pkl", "https://github.com/msane10/PROJET_FINALE_GOMYCODE/raw/main/model.pkl"),
        ("vectorizer.pkl", "https://github.com/msane10/PROJET_FINALE_GOMYCODE/raw/main/vectorizer.pkl"),
        ("label_encoder.pkl", "https://github.com/msane10/PROJET_FINALE_GOMYCODE/raw/main/label_encoder.pkl")
    ]
    
    # Télécharger chaque fichier
    for filename, url in files:
        if not os.path.exists(filename):  # Vérifie si le fichier existe déjà
            print(f"Téléchargement de {filename} depuis GitHub...")
            response = requests.get(url)
            with open(filename, "wb") as file:
                file.write(response.content)
            print(f"{filename} téléchargé avec succès.")
        else:
            print(f"{filename} existe déjà.")

# Appel de la fonction pour télécharger les modèles
download_model_from_github()

@st.cache_resource
def load_assets():
    return (
        joblib.load("model.pkl"),
        joblib.load("vectorizer.pkl"),
        joblib.load("label_encoder.pkl")
    )

# Détection de mots-clés
shipping_keywords = [
    "livraison", "colis", "expédition", "livreur", "relais", "suivi", "retard", "commande",
    "tri", "délais", "réception", "transport", "point relais", "envoi", "douane", "réexpédition"
]

billing_keywords = [
    "facture", "paiement", "remboursement", "carte", "prix", "montant", "tva", "réduction",
    "code promo", "frais", "prélèvement", "solde", "reçu", "mode de paiement", "échéance",
    "proforma", "annulation", "transaction", "historique de paiement"
]

support_keywords = [
    "compte", "application", "connexion", "mot de passe", "bug", "erreur", "support technique",
    "profil", "interface", "authentification", "sms", "email", "notifications", "accès",
    "déconnexion", "mobile", "mise à jour", "biométrie", "langue", "suppression", "paramètres"
]

hors_sujet_keywords = [
    "capital", "recette", "météo", "psychologue", "médecin", "piano", "voiture", "sport", "restaurant",
    "hôtel", "film", "concert", "vaccin", "wifi", "notaire", "banque", "podcast", "voyage", "nutrition", "maladie",
    "symptômes", "gâteau", "tarte", "temps", "distance", "terre", "lune", "livre", "film", "série"
]

def get_category_response(user_input, model, vectorizer, encoder):
    cleaned = clean_text(user_input)
    vect = vectorizer.transform([cleaned])
    probas = model.predict_proba(vect)[0]
    pred_class = model.predict(vect)[0]
    confidence = max(probas)
    category = encoder.inverse_transform([pred_class])[0]

    lower_input = user_input.lower()
    if confidence < 0.5:
        if any(word in lower_input for word in shipping_keywords):
            return "expédition", 0.7
        elif any(word in lower_input for word in billing_keywords):
            return "facturation", 0.7
        elif any(word in lower_input for word in support_keywords):
            return "assistance", 0.7
        elif any(word in lower_input for word in hors_sujet_keywords):
            return "hors_sujet", 0.95

    return category, confidence

def main():
    model, vectorizer, encoder = load_assets()

    st.title("🤖 ClassBot : Chatbot Intelligent pour la Classification Automatique des Requêtes Clients")
    st.markdown("""
    **Classifiez automatiquement vos requêtes clients en 3 catégories :**
    - 📦 Expédition
    - 💳 Facturation
    - 🛠️ Assistance
    """)

    user_input = st.text_area("💬 Écrivez votre requête ici :", height=100)

    if st.button("✨ Analyser la requête", type="primary"):
        if user_input:
            category, confidence = get_category_response(user_input, model, vectorizer, encoder)

            threshold = 0.5
            if len(user_input.split()) < 3:
                threshold = 0.6

            if category == "hors_sujet":
                st.error("""
                ❗ Nous n'avons pas pu identifier clairement votre demande.

                Ce chatbot est conçu pour vous aider avec :
                - 📦 Livraison / Expédition
                - 💳 Facturation / Paiement
                - 🛠️ Assistance technique

                Veuillez reformuler votre demande en utilisant des termes plus spécifiques.
                """)
            elif confidence < threshold:
                st.error("""
                ❗ Nous n'avons pas pu identifier clairement votre demande.

                Voici des exemples de requêtes que nous traitons :
                - "Mon colis n'est pas arrivé" (Expédition)
                - "Ma facture contient une erreur" (Facturation)
                - "Je n'arrive pas à me connecter" (Assistance)

                Veuillez reformuler votre demande en utilisant des termes plus spécifiques.
                """)
            else:
                if category == "expédition":
                    st.success("📦 **Service Expédition**")
                    st.write("""
                    Notre équipe logistique va traiter votre requête concernant :
                    - Suivi de colis
                    - Retards de livraison
                    - Problèmes d'adresse
                    - Retour de marchandise
                    """)
                elif category == "facturation":
                    st.success("💳 **Service Facturation**")
                    st.write("""
                    Notre service financier peut vous aider pour :
                    - Factures et reçus
                    - Remboursements
                    - Paiements en ligne
                    - Questions de TVA
                    """)
                elif category == "assistance":
                    st.success("🛠️ **Service Assistance**")
                    st.write("""
                    Notre équipe support peut vous aider pour :
                    - Problèmes de compte
                    - Bugs d'application
                    - Connexion et sécurité
                    - Mise à jour des informations
                    """)
        else:
            st.warning("⚠️ Veuillez saisir une requête à analyser")

if __name__ == "__main__":
    main()
