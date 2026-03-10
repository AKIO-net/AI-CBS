import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


# =============================================
# 📁 DATA LOADING FUNCTIONS
# =============================================

def setup_paths(base_path=None):
    """Configure les chemins pour le dataset CIC-IDS2017"""
    if base_path is None:
        base_path = Path("/home/abdou/Bureau/AI_for_cybersecurity/myenv/datasets")
    else:
        base_path = Path(base_path)

    cic_path = base_path / "cicids2017"
    traffic_labelling_path = cic_path / "GeneratedLabelledFlows" / "TrafficLabelling"
    machine_learning_path = cic_path / "MachineLearningCSV" / "MachineLearningCVE"

    print(f"📁 Chemin CIC-IDS2017: {cic_path}")
    print(f"📁 Chemin Traffic Labelling: {traffic_labelling_path}")
    print(f"📁 Chemin Machine Learning: {machine_learning_path}")

    return {
        'base_path': base_path,
        'cic_path': cic_path,
        'traffic_labelling_path': traffic_labelling_path,
        'machine_learning_path': machine_learning_path
    }


def verify_paths(paths):
    """Vérifie l'existence des chemins principaux"""
    paths_found = []

    if paths['traffic_labelling_path'].exists():
        paths_found.append("TrafficLabelling")
    if paths['machine_learning_path'].exists():
        paths_found.append("MachineLearningCSV")

    if paths_found:
        print(f"✅ Chemins trouvés: {', '.join(paths_found)}")
        return True
    else:
        print("❌ Aucun chemin CIC-IDS2017 trouvé")
        return False


def discover_dataset_structure(paths):
    """Découvre la structure du dataset CIC-IDS2017"""
    print("\n🔍 Découverte de la structure CIC-IDS2017...")

    structure = {
        'traffic_labelling': {
            'path': paths['traffic_labelling_path'],
            'files': [],
            'description': 'Fichiers de flux réseau étiquetés par jour'
        },
        'machine_learning': {
            'path': paths['machine_learning_path'],
            'files': [],
            'description': 'Fichiers consolidés pour le machine learning'
        }
    }

    for category, info in structure.items():
        if not info['path'].exists():
            print(f"❌ Chemin inaccessible: {info['path']}")
            continue

        files = list(info['path'].rglob("*.csv")) + list(info['path'].rglob("*.CSV"))
        if files:
            structure[category]['files'] = files

    # Affichage récapitulatif
    for category, info in structure.items():
        print(f"\n📂 {category.upper()}:")
        print(f"   Description: {info['description']}")
        print(f"   Chemin: {info['path']}")

        if info['files']:
            print(f"   Fichiers trouvés: {len(info['files'])}")
            for file_path in sorted(info['files'])[:5]:
                if file_path.exists():
                    file_size = file_path.stat().st_size / (1024 ** 3)
                    print(f"   ✅ {file_path.name} ({file_size:.2f} GB)")
            if len(info['files']) > 5:
                print(f"   ... et {len(info['files']) - 5} autres fichiers")
        else:
            print("   ❌ Aucun fichier trouvé")

    return structure


def optimize_dataframe_dtypes(df):
    """Optimise les types de données d'un DataFrame"""
    print("   🔧 Optimisation des types de données...")

    original_memory = df.memory_usage(deep=True).sum() / 1024 ** 2

    for col in df.select_dtypes(include=[np.number]).columns:
        col_type = df[col].dtype

        if col_type in [np.int8, np.int16, np.int32, np.uint8, np.uint16, np.uint32, np.float32]:
            continue

        if pd.api.types.is_integer_dtype(col_type):
            if df[col].isna().any():
                df[col] = df[col].astype('float32')
            else:
                c_min, c_max = df[col].min(), df[col].max()
                if c_min >= 0:
                    if c_max < 256:
                        df[col] = df[col].astype('uint8')
                    elif c_max < 65536:
                        df[col] = df[col].astype('uint16')
                    elif c_max < 4294967296:
                        df[col] = df[col].astype('uint32')
                else:
                    if c_min > -128 and c_max < 127:
                        df[col] = df[col].astype('int8')
                    elif c_min > -32768 and c_max < 32767:
                        df[col] = df[col].astype('int16')
                    elif c_min > -2147483648 and c_max < 2147483647:
                        df[col] = df[col].astype('int32')

        elif pd.api.types.is_float_dtype(col_type):
            df[col] = df[col].astype('float32')

    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df[col]) < 0.5:
            df[col] = df[col].astype('category')

    optimized_memory = df.memory_usage(deep=True).sum() / 1024 ** 2
    savings = original_memory - optimized_memory

    print(f"   💾 Mémoire: {original_memory:.1f}MB → {optimized_memory:.1f}MB")
    print(f"   📉 Économie: {savings:.1f}MB ({savings / original_memory * 100:.1f}%)")

    return df


def load_with_chunks(file_path, sample_rows=None):
    """Charge un fichier CSV efficacement avec gestion d'encodage"""
    try:
        df = pd.read_csv(file_path, nrows=sample_rows, low_memory=False)
    except UnicodeDecodeError:
        print(f"⚠️ Encodage UTF-8 échoué pour {file_path.name}, tentative avec ISO-8859-1...")
        df = pd.read_csv(file_path, nrows=sample_rows, low_memory=False, encoding='ISO-8859-1')
    return df


def analyze_traffic_labelling_files(paths, sample_rows=100000):
    """Analyse les fichiers de traffic labelling"""
    print("\n" + "=" * 70)
    print("ANALYSE DES FICHIERS TRAFFIC LABELLING")
    print("=" * 70)

    if not paths['traffic_labelling_path'].exists():
        print("❌ Chemin Traffic Labelling introuvable")
        return {}

    files = list(paths['traffic_labelling_path'].glob("*.csv"))
    if not files:
        print("❌ Aucun fichier CSV trouvé")
        return {}

    traffic_data = {}
    for file_path in sorted(files):
        print(f"\n📅 Analyse du fichier: {file_path.name}")
        df = load_with_chunks(file_path, sample_rows=sample_rows)
        if df is not None:
            df = optimize_dataframe_dtypes(df)
            traffic_data[file_path.stem] = df

    return traffic_data


def analyze_machine_learning_files(paths, sample_rows=100000):
    """Analyse les fichiers machine learning"""
    print("\n" + "=" * 70)
    print("ANALYSE DES FICHIERS MACHINE LEARNING")
    print("=" * 70)

    if not paths['machine_learning_path'].exists():
        print("❌ Chemin Machine Learning introuvable")
        return {}

    files = list(paths['machine_learning_path'].glob("*.csv"))
    if not files:
        print("❌ Aucun fichier CSV trouvé")
        return {}

    ml_data = {}
    for file_path in sorted(files):
        print(f"\n🤖 Analyse du fichier: {file_path.name}")
        df = load_with_chunks(file_path, sample_rows=sample_rows)
        if df is not None:
            df = optimize_dataframe_dtypes(df)
            ml_data[file_path.stem] = df

    return ml_data


def generate_comprehensive_report(datasets):
    """Génère un rapport complet"""
    print("\n" + "=" * 80)
    print("RAPPORT COMPLET CIC-IDS2017")
    print("=" * 80)

    total_files = total_rows = total_memory = 0
    for category, data_dict in datasets.items():
        if not data_dict:
            continue
        print(f"\n📂 CATÉGORIE: {category.upper()}")
        for name, df in data_dict.items():
            total_files += 1
            total_rows += len(df)
            memory_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
            total_memory += memory_mb
            print(f"   {name}: {df.shape} | {memory_mb:.1f}MB")

    print(f"\n📈 TOTAUX:")
    print(f"   📄 Fichiers: {total_files}")
    print(f"   📊 Lignes: {total_rows:,}")
    print(f"   💾 Mémoire totale: {total_memory:.1f} MB")


def load_complete_dataset(base_path=None, sample_rows=100000):
    """Charge et analyse tout le dataset"""
    print("🚀 CHARGEMENT COMPLET DU DATASET CIC-IDS2017")

    paths = setup_paths(base_path)
    if not verify_paths(paths):
        return {}

    structure = discover_dataset_structure(paths)

    datasets = {}
    datasets['traffic_labelling'] = analyze_traffic_labelling_files(paths, sample_rows)
    datasets['machine_learning'] = analyze_machine_learning_files(paths, sample_rows)

    generate_comprehensive_report(datasets)
    return datasets, paths


# =============================================
# 🔧 FEATURE EXTRACTION FUNCTIONS
# =============================================

def extract_features(df, file_name=""):
    """Pipeline robuste de feature extraction et preprocessing pour CIC-IDS2017"""
    print(f"\n⚙️ DÉMARRAGE DE LA FEATURE EXTRACTION {file_name}")

    # 1️⃣ Suppression colonnes inutiles
    drop_cols = ['Timestamp', 'Src IP', 'Dst IP', 'Flow ID', 'Source IP', 'Destination IP']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # 2️⃣ Détection du label
    possible_labels = ["Label", "label", " Label", "Class", "class", "Attack", "attack_cat", "Category", "category"]
    label_col = next((c for c in df.columns if c.strip() in possible_labels), None)
    if label_col is None:
        raise ValueError("Aucune colonne de label détectée.")

    # 3️⃣ Normalisation du label → 0 (normal/benign) / 1 (attack)
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()
    df[label_col] = df[label_col].apply(lambda x: 0 if "normal" in x or "benign" in x else 1)

    # 4️⃣ Identification des colonnes
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)

    print(f"🧩 Colonnes catégorielles détectées: {len(categorical_cols)}")
    print(f"🔢 Colonnes numériques détectées: {len(numeric_cols)}")

    # 5️⃣ Encodage catégoriel
    for col in categorical_cols:
        df[col] = pd.Categorical(df[col])
        df[col] = df[col].cat.codes

    # 6️⃣ Nettoyage des données numériques (inf, NaN, valeurs extrêmes)
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df[numeric_cols] = df[numeric_cols].clip(-1e10, 1e10)

    # 7️⃣ Standardisation
    if numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # 8️⃣ Suppression des features à faible variance
    features = df.drop(columns=[label_col])
    selector = VarianceThreshold(threshold=0.0)
    selected = selector.fit_transform(features)
    kept_columns = features.columns[selector.get_support(indices=True)]

    df_clean = pd.DataFrame(selected, columns=kept_columns)
    df_clean["Label"] = df[label_col].values  # normalisation du nom

    print(f"✅ Features finales: {df_clean.shape[1]} colonnes (dont label)")
    print(f"✅ Lignes: {df_clean.shape[0]}")
    return df_clean


def prepare_all_datasets(datasets, paths):
    """Applique extract_features() à tous les datasets chargés"""
    print("\n🚀 PRÉPARATION DES DATASETS POUR LE MACHINE LEARNING")

    processed = {}
    for category, data_dict in datasets.items():
        if not data_dict:
            continue
        print(f"\n📂 Catégorie: {category.upper()}")
        processed[category] = {}
        for name, df in data_dict.items():
            print(f"   🧮 Traitement du fichier: {name}")
            try:
                df_clean = extract_features(df)
                processed[category][name] = df_clean

                # Sauvegarde optionnelle
                save_path = paths['cic_path'] / f"{name}_prepared.csv"
                df_clean.to_csv(save_path, index=False)
                print(f"   💾 Sauvegardé: {save_path}")
            except Exception as e:
                print(f"   ⚠️ Erreur dans {name}: {e}")

    print("\n🎉 Préparation terminée pour tous les fichiers!")
    return processed


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
    plt.title('Comparaison des Performances des Modèles')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)

    for bar, accuracy in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{accuracy:.4f}', ha='center', va='bottom')

    plt.grid(axis='y', alpha=0.3)
    plt.show()


def run_machine_learning_pipeline(processed_datasets):
    """Exécute le pipeline complet de machine learning"""
    print("\n" + "=" * 60)
    print("🤖 DÉMARRAGE DU PIPELINE DE MACHINE LEARNING")
    print("=" * 60)

    # Utiliser le premier dataset disponible pour l'entraînement
    for category, data_dict in processed_datasets.items():
        if data_dict:
            first_dataset_name = list(data_dict.keys())[0]
            df = data_dict[first_dataset_name]
            print(f"📊 Utilisation du dataset: {first_dataset_name}")
            break
    else:
        print("❌ Aucun dataset disponible pour l'entraînement")
        return

    # Préparation des données
    X = df.drop('Label', axis=1)
    y = df['Label']

    # Pour la démonstration, on split en train/test
    from sklearn.model_selection import train_test_split
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

    print(f"\n🎉 PIPELINE TERMINÉ AVEC SUCCÈS!")
    print(f"🏆 Meilleur modèle: {'Random Forest' if rf_accuracy > dt_accuracy else 'Decision Tree'}")

    return results


# =============================================
# 🚀 MAIN EXECUTION
# =============================================

def main():
    """Fonction principale"""
    print("🚀 DÉMARRAGE DE L'ANALYSE CIC-IDS2017")

    # 1. Chargement des données
    datasets, paths = load_complete_dataset(
        base_path="/home/abdou/Bureau/AI_for_cybersecurity/myenv/datasets",
        sample_rows=50000
    )

    if not datasets:
        print("❌ Échec du chargement des données")
        return

    print("\n✅ ANALYSE TERMINÉE AVEC SUCCÈS!")

    # 2. Feature extraction et prétraitement
    processed_datasets = prepare_all_datasets(datasets, paths)
    print("\n✅ FEATURE EXTRACTION TERMINÉE AVEC SUCCÈS!")

    # 3. Machine Learning
    if processed_datasets:
        ml_results = run_machine_learning_pipeline(processed_datasets)
        print("\n🎯 MACHINE LEARNING TERMINÉ AVEC SUCCÈS!")
    else:
        print("❌ Aucun dataset préparé pour le machine learning")


if __name__ == "__main__":
    main()