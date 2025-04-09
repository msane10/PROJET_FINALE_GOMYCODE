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
stop_words = set(stopwords.words('french'))
stemmer = FrenchStemmer()

# Nettoyage de texte avancé
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ\s]", "", text)
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Chargement des données
with open("requetes_clients.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

data = [line.strip().split(" ||| ") for line in lines]
df = pd.DataFrame(data, columns=["message", "categorie"])
df["message_clean"] = df["message"].apply(clean_text)

# Encodage des étiquettes
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["categorie"])

# Vectorisation TF-IDF améliorée
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_df=0.95,
    min_df=2,
    stop_words = list(stopwords.words('french'))
)
X = vectorizer.fit_transform(df["message_clean"])
y = df["label"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fonction d'évaluation
def evaluate_model(model, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n📊 Résultats - {name}")
    print(f"Accuracy : {acc:.4f}")
    print("Rapport de classification :")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f"Matrice de Confusion - {name}")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.show()
    return acc

# === PHASE 2 : Comparaison Naive Bayes vs Decision Tree ===
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

# === PHASE 2 BIS : Optimisation de l'Arbre de Décision ===
param_grid = {
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print("\n📈 Meilleurs paramètres trouvés :", grid.best_params_)
best_tree_model = grid.best_estimator_
evaluate_model(best_tree_model, "Decision Tree Optimisé")
