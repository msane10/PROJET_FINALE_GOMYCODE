# === PHASE 1 : Classification de textes ===

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
from sklearn.model_selection import GridSearchCV

nltk.download('stopwords')
stop_words = list(stopwords.words('french'))
stemmer = FrenchStemmer()

# Nettoyage de texte avancé
def clean_text(text):
    """Nettoyage amélioré qui conserve plus d'informations"""
    text = text.lower()
    # Garder certains caractères spéciaux utiles
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ0-9\s'?-]", "", text)
    tokens = text.split()
    # Remplacer le stemming par une lemmatisation ou le supprimer
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Chargement des données - Version améliorée avec vérification
with open("requetes_clients.txt", "r", encoding="utf-8") as f:
    lines = [line.strip().split(" ||| ") for line in f if line.strip() and line.count(" ||| ") == 1]

df = pd.DataFrame(lines, columns=["message", "categorie"])
df = df.dropna()  # Supprime les lignes vides
df = df.reset_index(drop=True)  # Réinitialise les index

# Création de la colonne nettoyée - Ajout crucial manquant dans la version originale
df["message_clean"] = df["message"].apply(clean_text)

# Encodage des étiquettes
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["categorie"])

# Vérification des données
print("=== VÉRIFICATION DES DONNÉES ===")
print("Exemples de messages :")
print(df["message"].head())
print("\nCatégories uniques :", label_encoder.classes_)
print("Distribution des catégories :")
print(df['categorie'].value_counts())

# Vectorisation TF-IDF améliorée
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_df=0.95,
    min_df=2,
    stop_words=stop_words
)
X = vectorizer.fit_transform(df["message_clean"])
y = df["label"].values  # Conversion en array numpy

# Split train/test - Version plus robuste
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Fonction d'évaluation - Conservée identique
def evaluate_model(model, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n📊 Résultats - {name}")
    print(f"Accuracy : {acc:.4f}")
    print("Rapport de classification :")

    # Utilisez les classes présentes dans y_test au lieu de label_encoder.classes_
    present_labels = label_encoder.inverse_transform(sorted(set(y_test)))
    print(classification_report(y_test, y_pred, target_names=present_labels, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=present_labels,
                yticklabels=present_labels)
    plt.title(f"Matrice de Confusion - {name}")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.show()
    return acc

# === PHASE 2 : Comparaison Naive Bayes vs Decision Tree ===
# Objectif : évaluer les performances initiales de deux modèles simples sur des requêtes client
# Modèles comparés : Naive Bayes (texte) et Arbre de Décision (logique explicable)

phase2_models = {
    "Naive Bayes": MultinomialNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

phase2_results = {}
for name, model in phase2_models.items():
    acc = evaluate_model(model, name)
    phase2_results[name] = acc

best_model_phase2 = max(phase2_results, key=phase2_results.get)
print(f"\n🏆 Phase 2 - Meilleur modèle : {best_model_phase2} avec une accuracy de {phase2_results[best_model_phase2]:.4f}")

# === PHASE 3 : Réglage des Hyperparamètres ===
# Objectif : améliorer les performances des modèles Naive Bayes et Arbre de Décision
# en testant différents paramètres à l'ade de GridSearchCV (validation croisée).

# 🔍 Réglage de l'Arbre de Décision :
# Test de plusieurs valeurs pour max_depth, min_samples_split et criterion.
# Objectif : limiter le surapprentissage et améliorer la généralisation.
param_grid_tree = {
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

grid_tree = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_tree, cv=5, scoring='accuracy')
grid_tree.fit(X_train, y_train)

print("\n📈 Meilleurs paramètres trouvés pour l'arbre de décision:", grid_tree.best_params_)
best_tree_model = grid_tree.best_estimator_
evaluate_model(best_tree_model, "Decision Tree Optimisé")

# 🔍 Réglage de Naive Bayes :
# Test de différentes valeurs de alpha (lissage de Laplace) pour mieux gérer les zéros.
param_grid_nb = {
    'alpha': [0.01, 0.1, 0.5, 1.0],  # Valeurs plus basses
    'fit_prior': [True, False]  # Ajouter ce paramètre
}

grid_nb = GridSearchCV(MultinomialNB(), param_grid_nb, cv=5, scoring='accuracy')
grid_nb.fit(X_train, y_train)

print("\n📈 Meilleur alpha pour Naive Bayes:", grid_nb.best_params_)
best_nb_model = grid_nb.best_estimator_
evaluate_model(best_nb_model, "Naive Bayes Optimisé")

# 🔚 Résultat attendu :
# - Une meilleure accuracy pour les deux modèles
# - Des paramètres optimaux identifiés automatiquement
# - Une comparaison claire entre modèles de base et modèles optimisés


# === PHASE 4 : Phase de laboratoire / Performance ===
# Objectif : L'objectif de votre présentation doit être de présenter au public
# le concept de chatbots de traitement du langage naturel (NLP) avec NLTK pour la classification de texte.


import joblib

# Sauvegarde du modèle Naive Bayes optimisé et des objets nécessaires
joblib.dump(best_nb_model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("✅ Modèle, vectorizer et label_encoder sauvegardés avec succès.")