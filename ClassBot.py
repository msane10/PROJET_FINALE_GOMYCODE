# Étape 1 : Chargement et prétraitement des données

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Charger les données depuis le fichier texte
with open("requetes_clients.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Séparer chaque ligne en message et étiquette
data = [line.strip().split(" ||| ") for line in lines]
df = pd.DataFrame(data, columns=["message", "categorie"])

# Nettoyage de texte de base
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ\s]", "", text)  # supprimer ponctuations
    return text

df["message_clean"] = df["message"].apply(clean_text)

# Encodage des étiquettes
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["categorie"])

# Vectorisation TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["message_clean"])
y = df["label"]

# Séparation en jeu d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Afiicher
#print(df.head())

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Fonction utilitaire pour entraîner et évaluer un modèle
def evaluate_model(model, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n📊 Résultats - {name}")
    print(f"Accuracy : {acc:.4f}")
    print("Rapport de classification :")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Affichage de la matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f"Matrice de Confusion - {name}")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.show()
    return acc

# Modèles à tester
models = {
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# Évaluation
results = {}
for name, model in models.items():
    acc = evaluate_model(model, name)
    results[name] = acc

# Sélection du meilleur modèle
best_model_name = max(results, key=results.get)
print(f"\n🏆 Le modèle le plus performant est : {best_model_name} avec une accuracy de {results[best_model_name]:.4f}")
