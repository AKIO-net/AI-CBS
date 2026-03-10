import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# =============================================
# 📊 DATA LOADING AND EXPLORATION FUNCTIONS
# =============================================

def setup_display_config():
    """Configure l'affichage pandas"""
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print("✅ Configuration d'affichage activée")


def load_datasets(train_path, test_path):
    """Charge les datasets d'entraînement et de test"""
    print(f"📂 Chargement des datasets...")
    print(f"   - Training: {train_path}")
    print(f"   - Testing: {test_path}")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    print(f"✅ Datasets chargés avec succès!")
    return df_train, df_test


def explore_datasets(df_train, df_test):
    """Effectue l'exploration des données"""
    print("\n" + "=" * 60)
    print("🔍 EXPLORATION DES DONNÉES")
    print("=" * 60)

    # Forme des datasets
    print("\n=== FORME DES DATASETS ===")
    print(f"Training set shape: {df_train.shape}")
    print(f"Testing set shape: {df_test.shape}")

    # Colonnes
    print("\n=== COLONNES ===")
    print(df_train.columns.tolist())

    # Types de données
    print("\n=== TYPES DE DONNÉES ===")
    print(df_train.dtypes)

    # Valeurs manquantes
    print("\n=== VALEURS MANQUANTES ===")
    missing_train = df_train.isnull().sum()
    missing_test = df_test.isnull().sum()
    print("Training set:", missing_train[missing_train > 0].to_dict())
    print("Testing set:", missing_test[missing_test > 0].to_dict())

    # Information détaillée
    print("\n=== INFORMATION DÉTAILLÉE ===")
    df_train.info(memory_usage='deep')

    # Statistiques descriptives
    print("\n=== STATISTIQUES DESCRIPTIVES ===")
    print("Training set (premières 5 lignes):")
    print(df_train.describe(include='all').head())

    return df_train, df_test


def analyze_categorical_features(df_train):
    """Analyse les variables catégorielles"""
    print("\n" + "=" * 60)
    print("📊 ANALYSE DES VARIABLES CATÉGORIELLES")
    print("=" * 60)

    categorical_columns = df_train.select_dtypes(include=['object']).columns

    print(f"Nombre de variables catégorielles: {len(categorical_columns)}")
    for col in categorical_columns:
        print(f"\n{col}:")
        print(f"  Valeurs uniques: {df_train[col].nunique()}")
        print(f"  Top 5 valeurs: {df_train[col].value_counts().head().to_dict()}")

    return categorical_columns


def analyze_target_distribution(df_train, df_test):
    """Analyse la distribution de la variable cible"""
    print("\n" + "=" * 60)
    print("🎯 ANALYSE DE LA VARIABLE CIBLE")
    print("=" * 60)

    print("Training set:")
    print(df_train['label'].value_counts())
    attack_rate_train = df_train['label'].mean()
    print(f"Ratio d'attaque: {attack_rate_train:.3f}")

    print("\nTesting set:")
    print(df_test['label'].value_counts())
    attack_rate_test = df_test['label'].mean()
    print(f"Ratio d'attaque: {attack_rate_test:.3f}")

    return attack_rate_train, attack_rate_test


def analyze_memory_usage(df, dataset_name):
    """Analyse l'utilisation mémoire d'un DataFrame"""
    memory_per_column = df.memory_usage(deep=True)
    total_memory = memory_per_column.sum() / 1024 ** 2

    print(f"\n=== UTILISATION MÉMOIRE - {dataset_name.upper()} ===")
    print(f"Mémoire totale: {total_memory:.2f} MB")

    print("\nTop 10 des colonnes utilisant le plus de mémoire:")
    print(memory_per_column.sort_values(ascending=False).head(10))

    return total_memory


def generate_dataset_summary(df, name):
    """Génère un résumé complet d'un dataset"""
    print(f"\n{'=' * 50}")
    print(f"RÉSUMÉ - {name.upper()}")
    print(f"{'=' * 50}")
    print(f"Nombre d'observations: {df.shape[0]:,}")
    print(f"Nombre de features: {df.shape[1]}")
    print(f"Types de données:")
    print(df.dtypes.value_counts())
    print(f"Valeurs manquantes: {df.isnull().sum().sum()}")
    if 'label' in df.columns:
        print(f"Attaques: {df['label'].sum():,} ({df['label'].mean():.2%})")


# =============================================
# 🔧 PREPROCESSING AND FEATURE EXTRACTION FUNCTIONS
# =============================================

def preprocess_features(df_train, df_test, categorical_columns):
    """Pipeline complet de prétraitement et extraction de features"""
    print("\n" + "=" * 60)
    print("🔧 PIPELINE DE PRÉTRAITEMENT")
    print("=" * 60)

    # Convertir les colonnes catégorielles en liste
    categorical_columns = categorical_columns.tolist()

    # Identifier les colonnes numériques
    numeric_columns = df_train.select_dtypes(include=[np.number]).columns.tolist()
    if 'label' in numeric_columns:
        numeric_columns.remove('label')

    print(f"Colonnes catégorielles: {len(categorical_columns)}")
    print(f"Colonnes numériques: {len(numeric_columns)}")

    # 1. Encodage des variables catégorielles
    print("\n1️⃣ Encodage des variables catégorielles...")
    for col in categorical_columns:
        df_train[col] = pd.Categorical(df_train[col])
        df_test[col] = pd.Categorical(df_test[col], categories=df_train[col].cat.categories)
        df_train[col] = df_train[col].cat.codes
        df_test[col] = df_test[col].cat.codes

    # 2. Standardisation des variables numériques
    print("2️⃣ Standardisation des variables numériques...")
    scaler = StandardScaler()
    df_train[numeric_columns] = scaler.fit_transform(df_train[numeric_columns])
    df_test[numeric_columns] = scaler.transform(df_test[numeric_columns])

    # 3. Sélection de caractéristiques à faible variance
    print("3️⃣ Suppression des caractéristiques à faible variance...")
    selector = VarianceThreshold(threshold=0.0)
    selector.fit(df_train[numeric_columns + categorical_columns])

    selected_columns = [col for col, keep in zip(numeric_columns + categorical_columns, selector.get_support()) if keep]

    df_train = df_train[selected_columns + ['label']]
    df_test = df_test[selected_columns + ['label']]

    print(f"✅ Nombre de caractéristiques après filtrage: {len(selected_columns)}")
    print(f"✅ Shape final - Train: {df_train.shape}, Test: {df_test.shape}")

    return df_train, df_test, selected_columns


def save_preprocessed_data(df_train, df_test, output_dir):
    """Sauvegarde les données prétraitées"""
    print("\n💾 Sauvegarde des données prétraitées...")

    df_train.to_csv(f'{output_dir}/UNSW_NB15_train_prepared.csv', index=False)
    df_test.to_csv(f'{output_dir}/UNSW_NB15_test_prepared.csv', index=False)

    print(f"✅ Données sauvegardées dans: {output_dir}")


# =============================================
# 🤖 MACHINE LEARNING FUNCTIONS
# =============================================

def prepare_ml_data(df_train, df_test):
    """Prépare les données pour le machine learning"""
    X_train = df_train.drop('label', axis=1)
    y_train = df_train['label']
    X_test = df_test.drop('label', axis=1)
    y_test = df_test['label']

    print(f"\n📊 Données pour ML:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")

    return X_train, y_train, X_test, y_test


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
    plt.tight_layout()
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


def train_isolation_forest(X_train, y_train, attack_rate):
    """Entraîne un modèle Isolation Forest seulement sur les paquets bénins"""
    print("\n" + "=" * 40)
    print("🏝️ ENTRAÎNEMENT DE L'ISOLATION FOREST (BENIGN ONLY)")
    print("=" * 40)

    # Séparer les données bénignes des attaques
    X_train_benign = X_train[y_train == 0]

    print(f"Données totales d'entraînement: {X_train.shape[0]}")
    print(f"Données bénignes pour l'entraînement: {X_train_benign.shape[0]}")
    print(f"Données d'attaque exclues de l'entraînement: {(y_train == 1).sum()}")

    # Calculer le taux d'anomalie attendu dans les données de test
    # C'est le ratio d'attaques dans les données de test
    # Note: Pour Isolation Forest entraîné seulement sur des données normales,
    # le paramètre contamination peut être estimé comme le taux d'anomalies attendu
    if attack_rate > 0.5:
        print(f"⚠️ Taux d'attaque élevé ({attack_rate:.3f}) > 0.5, utilisation de 'auto' pour contamination")
        contamination = 'auto'
    else:
        contamination = min(attack_rate, 0.5)  # Maximum 0.5 pour Isolation Forest
        print(f"Contamination parameter: {contamination:.3f}")

    isolation_forest = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )

    print("Entraînement de l'Isolation Forest sur les données bénignes uniquement...")

    # Isolation Forest s'entraîne seulement sur les données bénignes
    isolation_forest.fit(X_train_benign)
    print("✅ Isolation Forest entraîné uniquement sur les données bénignes avec succès!")

    return isolation_forest


def evaluate_isolation_forest(model, X_test, y_test):
    """Évalue le modèle Isolation Forest (nécessite une conversion spéciale)"""
    print(f"\n{'=' * 50}")
    print(f"📊 ÉVALUATION - ISOLATION FOREST")
    print(f"{'=' * 50}")

    # Prédictions: 1 pour normal, -1 pour anomalie
    y_pred_iso = model.predict(X_test)

    # Convertir les prédictions: -1 (anomalie) → 1 (attack), 1 (normal) → 0 (normal)
    y_pred = (y_pred_iso == -1).astype(int)
    #creates an array of booleans where -1 is false and 0 is 1 is true, then transform it to intigers

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
    plt.title('Matrice de Confusion - Isolation Forest')
    plt.ylabel('Vraie étiquette')
    plt.xlabel('Étiquette prédite')
    plt.tight_layout()
    plt.show()

    return accuracy, y_pred


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
    plt.figure(figsize=(12, 6))
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]

    colors = ['skyblue', 'lightcoral', 'lightgreen']
    bars = plt.bar(models, accuracies, color=colors[:len(models)])
    plt.title('Comparaison des Performances des Modèles - UNSW-NB15')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)

    for bar, accuracy in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{accuracy:.4f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()


def run_machine_learning_pipeline(X_train, y_train, X_test, y_test, attack_rate_train):
    """Exécute le pipeline complet de machine learning"""
    print("\n" + "=" * 60)
    print("🤖 PIPELINE DE MACHINE LEARNING")
    print("=" * 60)

    results = {}

    # 1. Decision Tree
    dt_model = train_decision_tree(X_train, y_train)
    dt_accuracy, dt_predictions = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
    dt_importance = analyze_feature_importance(dt_model, X_train.columns, "Decision Tree")

    results['Decision Tree'] = {
        'model': dt_model,
        'accuracy': dt_accuracy,
        'predictions': dt_predictions,
        'importance': dt_importance
    }

    # 2. Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_accuracy, rf_predictions = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    rf_importance = analyze_feature_importance(rf_model, X_train.columns, "Random Forest")

    results['Random Forest'] = {
        'model': rf_model,
        'accuracy': rf_accuracy,
        'predictions': rf_predictions,
        'importance': rf_importance
    }

    # 3. Isolation Forest (seulement si le taux d'attaque est raisonnable)
    if attack_rate_train <= 0.7:
        iso_model = train_isolation_forest(X_train, y_train, attack_rate_train)
        iso_accuracy, iso_predictions = evaluate_isolation_forest(iso_model, X_test, y_test)

        results['Isolation Forest'] = {
            'model': iso_model,
            'accuracy': iso_accuracy,
            'predictions': iso_predictions,
            'importance': None
        }
    else:
        print(f"\n⚠️ Isolation Forest non utilisé car taux d'attaque ({attack_rate_train:.3f}) > 0.5")
        print("   Isolation Forest fonctionne mieux avec contamination <= 0.5")
        print("   Pour les données très déséquilibrées, utilisez les modèles supervisés")
    # Comparaison des modèles
    if len(results) > 1:
        compare_models(results)

    # Analyse détaillée
    print("\n" + "=" * 50)
    print("🔍 ANALYSE DÉTAILLÉE DES RÉSULTATS")
    print("=" * 50)

    if results:
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"🏆 Meilleur modèle: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")

    print(f"\n📊 Scores détaillés:")
    for model_name, result in results.items():
        print(f"  {model_name}: {result['accuracy']:.4f}")

    # Informations sur le dataset
    print(f"\n📈 Information du dataset:")
    print(f"  Taux d'attaque (train): {attack_rate_train:.3f}")
    print(f"  Nombre de features: {X_train.shape[1]}")
    print(f"  Échantillons d'entraînement: {X_train.shape[0]}")

    return results


# =============================================
# 🚀 MAIN FUNCTION
# =============================================

def main():
    """Fonction principale orchestrant tout le pipeline"""
    print("🚀 DÉMARRAGE DE L'ANALYSE UNSW-NB15")
    print("=" * 60)

    # Configuration
    setup_display_config()

    # Chemins des fichiers
    train_file_path = 'unsw_nb15/UNSW_NB15_training-set.csv'
    test_file_path = 'unsw_nb15/UNSW_NB15_testing-set.csv'
    output_dir = 'unsw_nb15'

    try:
        # 1. Chargement des données
        df_train, df_test = load_datasets(train_file_path, test_file_path)

        # 2. Exploration des données
        df_train, df_test = explore_datasets(df_train, df_test)

        # 3. Analyse des variables catégorielles
        categorical_columns = analyze_categorical_features(df_train)

        # 4. Analyse de la variable cible
        attack_rate_train, attack_rate_test = analyze_target_distribution(df_train, df_test)

        # 5. Analyse mémoire
        analyze_memory_usage(df_train, "training set")
        analyze_memory_usage(df_test, "testing set")

        # 6. Résumés
        generate_dataset_summary(df_train, "Training Set")
        generate_dataset_summary(df_test, "Testing Set")

        # 7. Prétraitement
        df_train_processed, df_test_processed, selected_columns = preprocess_features(
            df_train, df_test, categorical_columns
        )

        # 8. Sauvegarde des données prétraitées
        save_preprocessed_data(df_train_processed, df_test_processed, output_dir)

        # 9. Préparation pour le machine learning
        X_train, y_train, X_test, y_test = prepare_ml_data(df_train_processed, df_test_processed)

        # 10. Pipeline de machine learning
        ml_results = run_machine_learning_pipeline(X_train, y_train, X_test, y_test, attack_rate_train)

        print("\n" + "=" * 60)
        print("🎉 ANALYSE TERMINÉE AVEC SUCCÈS!")
        print("=" * 60)

        return {
            'train_data': df_train,
            'test_data': df_test,
            'processed_train': df_train_processed,
            'processed_test': df_test_processed,
            'ml_results': ml_results,
            'attack_rate_train': attack_rate_train
        }

    except FileNotFoundError as e:
        print(f"\n❌ ERREUR: Fichier non trouvé - {e}")
        print("Vérifiez les chemins des fichiers:")
        print(f"  Train: {train_file_path}")
        print(f"  Test: {test_file_path}")
        return None

    except Exception as e:
        print(f"\n❌ ERREUR inattendue: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Exécution du pipeline
    results = main()

    if results:
        # Affichage des résultats finaux
        print("\n📊 RÉCAPITULATIF FINAL:")
        print(f"- Données brutes: Train={results['train_data'].shape}, Test={results['test_data'].shape}")
        print(f"- Données traitées: Train={results['processed_train'].shape}, Test={results['processed_test'].shape}")
        print(f"- Taux d'attaque: {results['attack_rate_train']:.3f}")


            # Afficher un warning si Isolation Forest n'a pas été utilisé
        if 'Isolation Forest' not in results['ml_results']:
            print(f"\n⚠️ REMARQUE: Isolation Forest n'a pas été utilisé")
            print(f"   Car taux d'attaque ({results['attack_rate_train']:.3f}) > 0.5")
            print(f"   Isolation Forest fonctionne mieux pour les datasets avec <= 50% d'anomalies")
    else:
        print("\n❌ L'analyse a échoué. Vérifiez les erreurs ci-dessus.")
