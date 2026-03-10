# scripts/analyze_ton_iot_simplified.py
import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# Configuration simplifiée
CONFIG = {
    'base_path': "/home/abdou/Bureau/AI_for_cybersecurity/myenv/datasets",
    'file_patterns': {
        'iot': ["Train_Test_IoT_*.csv"],
        'linux': ["**/*linux*.csv"],
        'windows': ["**/*windows*.csv"],
        'network': ["**/*network*.csv"]
    },
    'encodings': ['utf-8', 'latin-1'],
    'target_columns': ['label', 'attack', 'Label']
}


# =============================================
# 📁 DATA LOADING FUNCTIONS
# =============================================

def setup_paths(base_path=None):
    """Configure les chemins pour le dataset TON-IoT"""
    if base_path is None:
        base_path = Path(CONFIG['base_path'])
    else:
        base_path = Path(base_path)

    ton_iot_path = base_path / "ton_iot" / "ton_iot"
    train_test_path = base_path / "ton_iot" / "Train_Test_datasets"

    print(f"📁 Chargement depuis: {base_path}")
    print(f"📁 Chemin TON-IoT: {ton_iot_path}")
    print(f"📁 Chemin Train-Test: {train_test_path}")

    return {
        'base_path': base_path,
        'ton_iot_path': ton_iot_path,
        'train_test_path': train_test_path
    }


def check_paths(paths):
    """Vérifie que les chemins existent"""
    for name, path in [("TON-IoT", paths['ton_iot_path']),
                       ("Train-Test", paths['train_test_path'])]:
        if path.exists():
            print(f"✅ {name} trouvé")
        else:
            print(f"❌ {name} introuvable")


def find_files(paths):
    """Recherche tous les fichiers CSV du dataset"""
    print("\n🔍 Recherche des fichiers...")

    found_files = {}
    for category, patterns in CONFIG['file_patterns'].items():
        category_files = []
        for pattern in patterns:
            files = list(paths['train_test_path'].glob(pattern)) + list(paths['ton_iot_path'].glob(pattern))
            category_files.extend(files)

        found_files[category] = list(set(category_files))
        print(f"📂 {category}: {len(found_files[category])} fichiers")

    return found_files


def load_file(file_path, max_rows=None):
    """Charge un fichier CSV avec gestion d'encodage"""
    for encoding in CONFIG['encodings']:
        try:
            df = pd.read_csv(file_path, nrows=max_rows, encoding=encoding, low_memory=False)
            print(f"   ✅ Chargé avec {encoding}")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            return None
    return None


def convert_labels(label_series):
    """Convertit les labels en format binaire (0=normal, 1=attaque)"""
    labels = label_series.astype(str).str.lower()
    return labels.map({
        'normal': 0, 'benign': 0, '0': 0,
        'attack': 1, 'anomaly': 1, '1': 1, 'malicious': 1
    }).fillna(0).astype(int)


def quick_analysis(df, name):
    """Analyse rapide d'un dataset"""
    # Trouve la colonne cible
    target_col = None
    for col in CONFIG['target_columns']:
        if col in df.columns:
            target_col = col
            break

    # Analyse de base
    print(f"      📊 Shape: {df.shape}")
    print(f"      🔧 Colonnes: {len(df.columns)}")

    if target_col:
        # Convertit les labels en binaire
        df[target_col] = convert_labels(df[target_col])
        attack_rate = df[target_col].mean()
        print(f"      🎯 Taux d'attaque: {attack_rate:.1%}")


def load_all_data(paths, max_rows=50000):
    """Charge tous les datasets"""
    print("\n🚗 Chargement des données...")

    file_structure = find_files(paths)
    datasets = {}

    for category, files in file_structure.items():
        print(f"\n📱 Catégorie: {category.upper()}")
        category_data = {}

        for file_path in files[:3]:  # Limite à 3 fichiers par catégorie pour la démo
            if file_path.exists():
                print(f"   📖 {file_path.name}")
                df = load_file(file_path, max_rows)
                if df is not None:
                    category_data[file_path.stem] = df
                    quick_analysis(df, file_path.stem)

        datasets[category] = category_data

    return datasets


def generate_report(datasets):
    """Génère un rapport sommaire"""
    print("\n" + "=" * 50)
    print("📈 RAPPORT TON-IoT")
    print("=" * 50)

    total_files = total_rows = total_memory = 0

    for category, data_dict in datasets.items():
        if data_dict:
            print(f"\n📂 {category.upper()}:")
            for name, df in data_dict.items():
                total_files += 1
                total_rows += len(df)
                memory_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
                total_memory += memory_mb

                # Info attaques
                attack_info = ""
                for target in CONFIG['target_columns']:
                    if target in df.columns:
                        attack_rate = df[target].mean()
                        attack_info = f" | Attaques: {attack_rate:.1%}"
                        break

                print(f"   {name}: {df.shape} | {memory_mb:.1f}MB{attack_info}")

    print(f"\n📊 TOTAUX:")
    print(f"   Fichiers: {total_files}")
    print(f"   Lignes: {total_rows:,}")
    print(f"   Mémoire: {total_memory:.1f}MB")


# =============================================
# 🔧 PREPROCESSING FUNCTIONS
# =============================================

def preprocess_data(df, name=""):
    """
    Nettoie et prépare les données pour le machine learning
    """
    print(f"\n🔧 Préprocessing: {name}")

    if df is None or df.empty:
        return None

    df_clean = df.copy()

    # 1. Gestion des labels
    target_col = None
    for col in CONFIG['target_columns']:
        if col in df_clean.columns:
            target_col = col
            break

    if target_col:
        df_clean.rename(columns={target_col: 'label'}, inplace=True)
        df_clean['label'] = df_clean['label'].astype(str).str.lower().map({
            'normal': 0, 'benign': 0, '0': 0,
            'attack': 1, 'anomaly': 1, '1': 1
        }).fillna(0).astype(int)

    # 2. Nettoyage des données
    # Supprime les colonnes vides
    df_clean.dropna(axis=1, how='all', inplace=True)

    # Gère les valeurs manquantes
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(include=['object']).columns

    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())

    # 3. Encodage des variables catégorielles
    for col in categorical_cols:
        if col != 'label' and df_clean[col].nunique() <= 50:  # Limite le nombre de catégories
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        else:
            if col != 'label':
                df_clean.drop(columns=[col], inplace=True)

    # 4. Normalisation des features numériques
    features_to_scale = [col for col in numeric_cols if col != 'label' and col in df_clean.columns]
    if features_to_scale:
        scaler = StandardScaler()
        df_clean[features_to_scale] = scaler.fit_transform(df_clean[features_to_scale])

    print(f"   ✅ Terminé: {df.shape} → {df_clean.shape}")

    if 'label' in df_clean.columns:
        attack_rate = df_clean['label'].mean()
        print(f"   🎯 Attaques: {attack_rate:.1%}")

    return df_clean


def prepare_all_datasets(datasets):
    """Applique le préprocessing à tous les datasets chargés"""
    print("\n🔧 PRÉPROCESSING DE TOUS LES DATASETS")
    processed_data = {}

    for category, data_dict in datasets.items():
        if data_dict:
            print(f"\n📂 Catégorie: {category.upper()}")
            for name, df in data_dict.items():
                print(f"   🧮 Traitement: {name}")
                processed = preprocess_data(df, f"{category}_{name}")
                if processed is not None:
                    processed_data[f"{category}_{name}"] = processed

    print(f"\n🎉 PRÉPROCESSING TERMINÉ! {len(processed_data)} datasets prêts pour le ML")
    return processed_data


# =============================================
# 🤖 MACHINE LEARNING FUNCTIONS
# =============================================

def evaluate_model(model, X_test, y_test, model_name):
    """Évalue un modèle et affiche les métriques de performance"""
    print(f"\n{'=' * 50}")
    print(f"📊 ÉVALUATION - {model_name.upper()}")
    print(f"{'=' * 50}")

    # Prédictions
    y_pred = model.predict(X_test)

    # Métriques
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nRapport de classification:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.title(f'Matrice de Confusion - {model_name}')
    plt.ylabel('Vraie étiquette')
    plt.xlabel('Étiquette prédite')
    plt.show()

    return accuracy, y_pred


def train_decision_tree(X_train, y_train):
    """Entraîne un modèle Decision Tree"""
    print("\n" + "=" * 40)
    print("🌳 ENTRAÎNEMENT DU DECISION TREE")
    print("=" * 40)

    dt_classifier = DecisionTreeClassifier(
        random_state=42,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10
    )

    print("Entraînement du Decision Tree en cours...")
    dt_classifier.fit(X_train, y_train)
    print("✅ Decision Tree entraîné avec succès!")

    return dt_classifier


def train_random_forest(X_train, y_train):
    """Entraîne un modèle Random Forest"""
    print("\n" + "=" * 40)
    print("🌲 ENTRAÎNEMENT DU RANDOM FOREST")
    print("=" * 40)

    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=15,
        min_samples_split=15,
        min_samples_leaf=5,
        n_jobs=-1
    )

    print("Entraînement du Random Forest en cours...")
    rf_classifier.fit(X_train, y_train)
    print("✅ Random Forest entraîné avec succès!")

    return rf_classifier


def analyze_feature_importance(model, feature_names, model_name):
    """Analyse l'importance des features pour un modèle"""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 10 des features les plus importantes ({model_name}):")
    print(importance_df.head(10))

    return importance_df


def compare_models(results):
    """Compare les performances de tous les modèles"""
    print("\n" + "=" * 50)
    print("📈 COMPARAISON DES PERFORMANCES")
    print("=" * 50)

    comparison = pd.DataFrame({
        'Modèle': list(results.keys()),
        'Accuracy': [results[model]['accuracy'] for model in results.keys()]
    })

    print(comparison)

    # Visualisation
    plt.figure(figsize=(10, 6))
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]

    bars = plt.bar(models, accuracies, color=['skyblue', 'lightcoral'])
    plt.title('Comparaison des Performances des Modèles - TON-IoT')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)

    for bar, accuracy in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{accuracy:.4f}', ha='center', va='bottom')

    plt.grid(axis='y', alpha=0.3)
    plt.show()


def run_machine_learning_pipeline(processed_datasets):
    """Exécute le pipeline complet de machine learning sur TON-IoT"""
    print("\n" + "=" * 60)
    print("🤖 DÉMARRAGE DU PIPELINE DE MACHINE LEARNING - TON-IoT")
    print("=" * 60)

    if not processed_datasets:
        print("❌ Aucun dataset disponible pour l'entraînement")
        return

    # Utiliser le premier dataset disponible pour l'entraînement
    first_dataset_name = list(processed_datasets.keys())[0]
    df = processed_datasets[first_dataset_name]
    print(f"📊 Utilisation du dataset: {first_dataset_name}")

    # Préparation des données
    X = df.drop('label', axis=1)
    y = df['label']

    # Split en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"📊 Données d'entraînement: {X_train.shape}")
    print(f"📊 Données de test: {X_test.shape}")
    print(f"🎯 Distribution des labels - Train: {y_train.value_counts().to_dict()}")
    print(f"🎯 Distribution des labels - Test: {y_test.value_counts().to_dict()}")

    # Entraînement des modèles
    dt_model = train_decision_tree(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # Évaluation des modèles
    results = {}

    dt_accuracy, dt_predictions = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
    dt_importance = analyze_feature_importance(dt_model, X.columns, "Decision Tree")

    rf_accuracy, rf_predictions = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    rf_importance = analyze_feature_importance(rf_model, X.columns, "Random Forest")

    # Stockage des résultats
    results['Decision Tree'] = {
        'model': dt_model,
        'accuracy': dt_accuracy,
        'predictions': dt_predictions,
        'importance': dt_importance
    }

    results['Random Forest'] = {
        'model': rf_model,
        'accuracy': rf_accuracy,
        'predictions': rf_predictions,
        'importance': rf_importance
    }

    # Comparaison finale
    compare_models(results)

    # Analyse des résultats
    print(f"\n🔍 ANALYSE DES RÉSULTATS TON-IoT:")
    print(f"🏆 Meilleur modèle: {'Random Forest' if rf_accuracy > dt_accuracy else 'Decision Tree'}")
    print(f"📊 Différence d'accuracy: {abs(rf_accuracy - dt_accuracy):.4f}")

    # Informations sur les datasets
    print(f"\n📂 Datasets utilisés: {len(processed_datasets)}")
    for name, dataset in processed_datasets.items():
        if 'label' in dataset.columns:
            attack_rate = dataset['label'].mean()
            print(f"   📊 {name}: {dataset.shape} | Taux d'attaque: {attack_rate:.1%}")

    return results


# =============================================
# 🚀 MAIN EXECUTION
# =============================================

def main():
    """Fonction principale"""
    print("🚀 ANALYSE TON-IoT - Version Complète avec Machine Learning")

    # 1. Configuration des chemins
    paths = setup_paths()
    check_paths(paths)

    # 2. Chargement des données
    datasets = load_all_data(paths, max_rows=30000)

    if not datasets:
        print("❌ Échec du chargement des données")
        return None, None, None

    # 3. Rapport initial
    generate_report(datasets)

    # 4. Préprocessing
    processed_data = prepare_all_datasets(datasets)

    # 5. Machine Learning
    if processed_data:
        ml_results = run_machine_learning_pipeline(processed_data)
        print("\n🎯 MACHINE LEARNING TON-IoT TERMINÉ AVEC SUCCÈS!")
    else:
        print("❌ Aucun dataset préparé pour le machine learning")
        ml_results = None

    return paths, datasets, processed_data, ml_results


if __name__ == "__main__":
    paths, datasets, processed_data, ml_results = main()