from pathlib import Path
import json

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "BASE_DATOS_120_JUGADORES_CON_RIESGO.xlsx"
OUTPUT_DIR = BASE_DIR / "resultados_modelos"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "Riesgo_Lesion"
POSITIVE_CLASS = "Riesgo Alto"
FEATURES = [
    "Peso(kg)", "Talla (m)", "IMC", "SUM 3 PL", "SUM 6 PL",
    "Perimetro de cintura", "Perimetro Brazo relajado", "Perimetro de muslo",
    "Perimetro de pantorrilla", "Diametro de Humero", "Diametro de Femur",
    "Indice_Sobrecarga_Muscular", "Horas_Sueno", "Carga_Entrenamiento",
]

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

PALETTE = {
    "Random Forest": "#A3BEFA",
    "SVM": "#FFE15B",
    "XGBoost": "#F0986E",
    "Red Neuronal": "#A3D576",
}


def use_chart_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "figure.edgecolor": "none",
            "savefig.facecolor": TOKENS["surface"],
            "savefig.edgecolor": "none",
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.grid": True,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "Segoe UI",
            "patch.linewidth": 1.0,
        },
    )


def add_header(fig, ax, title: str, subtitle: str) -> None:
    ax.set_title("")
    fig.subplots_adjust(top=0.82)
    left = ax.get_position().x0
    fig.text(left, 0.965, title, ha="left", va="top", fontsize=15, fontweight="bold", color=TOKENS["ink"])
    fig.text(left, 0.915, subtitle, ha="left", va="top", fontsize=10, color=TOKENS["muted"])
    sns.despine(ax=ax)


def macro_specificity(y_true, y_pred, labels) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()
    values = []
    for i in range(len(labels)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        values.append(tn / (tn + fp) if (tn + fp) else 0)
    return float(np.mean(values))


def one_vs_rest_scores(y_true, y_pred, positive_class: str) -> dict:
    y_true_bin = np.array(y_true) == positive_class
    y_pred_bin = np.array(y_pred) == positive_class
    tp = int(np.sum(y_true_bin & y_pred_bin))
    fp = int(np.sum(~y_true_bin & y_pred_bin))
    fn = int(np.sum(y_true_bin & ~y_pred_bin))
    tn = int(np.sum(~y_true_bin & ~y_pred_bin))
    return {
        "recall_riesgo_alto": tp / (tp + fn) if (tp + fn) else 0,
        "precision_riesgo_alto": tp / (tp + fp) if (tp + fp) else 0,
        "specificity_riesgo_alto": tn / (tn + fp) if (tn + fp) else 0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def build_models() -> dict:
    scaler = ColumnTransformer(
        transformers=[("num", StandardScaler(), FEATURES)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_split=2,
            max_features="sqrt",
            random_state=RANDOM_STATE,
        ),
        "SVM": Pipeline([
            ("scaler", scaler),
            ("model", SVC(kernel="rbf", C=3.0, gamma="scale", random_state=RANDOM_STATE)),
        ]),
        "XGBoost": XGBClassifier(
            n_estimators=180,
            max_depth=3,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softmax",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
        ),
        "Red Neuronal": Pipeline([
            ("scaler", scaler),
            ("model", MLPClassifier(
                hidden_layer_sizes=(24, 12),
                activation="relu",
                alpha=0.001,
                max_iter=2000,
                random_state=RANDOM_STATE,
            )),
        ]),
    }


def evaluate_models():
    df = pd.read_excel(DATA_PATH)
    missing = [col for col in FEATURES + [TARGET] if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.mean(numeric_only=True))
    y = df[TARGET].astype(str)
    labels = sorted(y.unique())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    rows = []
    class_rows = []
    matrices = {}
    label_encoder = LabelEncoder().fit(y_train)

    for name, model in build_models().items():
        if name == "XGBoost":
            model.fit(X_train, label_encoder.transform(y_train))
            y_pred = label_encoder.inverse_transform(model.predict(X_test).astype(int))
            y_train_pred = label_encoder.inverse_transform(model.predict(X_train).astype(int))
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_train_pred = model.predict(X_train)

        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=labels, average="macro", zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=labels, average="weighted", zero_division=0
        )
        high = one_vs_rest_scores(y_test, y_pred, POSITIVE_CLASS)
        rows.append({
            "algoritmo": name,
            "accuracy_train": accuracy_score(y_train, y_train_pred),
            "accuracy_test": accuracy_score(y_test, y_pred),
            "precision_macro": precision_macro,
            "recall_macro_sensitivity": recall_macro,
            "specificity_macro": macro_specificity(y_test, y_pred, labels),
            "f1_macro": f1_macro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
            **high,
        })

        per_class = precision_recall_fscore_support(
            y_test, y_pred, labels=labels, average=None, zero_division=0
        )
        for label, precision, recall, f1, support in zip(labels, *per_class):
            class_rows.append({
                "algoritmo": name,
                "clase": label,
                "precision": precision,
                "recall_sensitivity": recall,
                "f1_score": f1,
                "support": int(support),
            })
        matrices[name] = confusion_matrix(y_test, y_pred, labels=labels)

    metrics = pd.DataFrame(rows)
    by_class = pd.DataFrame(class_rows)
    metrics.to_csv(OUTPUT_DIR / "metricas_modelos.csv", index=False, encoding="utf-8-sig")
    by_class.to_csv(OUTPUT_DIR / "metricas_por_clase.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "fuente": str(DATA_PATH.name),
        "filas": int(len(df)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "features": FEATURES,
        "target": TARGET,
        "clases": labels,
        "nota": "Etiquetas Riesgo_Lesion generadas por el propio proyecto; resultados experimentales no clínicos.",
    }
    (OUTPUT_DIR / "metodologia.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics, by_class, matrices, labels


def plot_metrics(metrics: pd.DataFrame) -> None:
    selected = metrics.melt(
        id_vars="algoritmo",
        value_vars=["accuracy_test", "recall_macro_sensitivity", "precision_macro", "specificity_macro"],
        var_name="metrica",
        value_name="valor",
    )
    selected["metrica"] = selected["metrica"].map({
        "accuracy_test": "Accuracy",
        "recall_macro_sensitivity": "Recall / Sensitivity",
        "precision_macro": "Precision",
        "specificity_macro": "Specificity",
    })

    fig, ax = plt.subplots(figsize=(11, 6.5))
    sns.barplot(
        data=selected,
        x="valor",
        y="algoritmo",
        hue="metrica",
        ax=ax,
        palette=["#A3BEFA", "#FFE15B", "#F0986E", "#A3D576"],
        edgecolor=TOKENS["ink"],
        linewidth=0.7,
    )
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Resultado en conjunto de prueba")
    ax.set_ylabel("")
    ax.legend(title="", loc="lower left", bbox_to_anchor=(0, 1.02), ncol=4, frameon=False)
    for container in ax.containers:
        ax.bar_label(container, labels=[f"{v.get_width():.1%}" for v in container], padding=3, fontsize=8)
    add_header(
        fig, ax,
        "Comparación de desempeño por algoritmo",
        "Métricas macro/multiclase sobre el 20% de prueba; n=120 jugadores, random_state=42.",
    )
    fig.savefig(OUTPUT_DIR / "comparacion_metricas_modelos.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "comparacion_metricas_modelos.svg", bbox_inches="tight")
    plt.close(fig)


def plot_high_risk(metrics: pd.DataFrame) -> None:
    selected = metrics.melt(
        id_vars="algoritmo",
        value_vars=["recall_riesgo_alto", "precision_riesgo_alto", "specificity_riesgo_alto"],
        var_name="metrica",
        value_name="valor",
    )
    selected["metrica"] = selected["metrica"].map({
        "recall_riesgo_alto": "Recall Riesgo Alto",
        "precision_riesgo_alto": "Precision Riesgo Alto",
        "specificity_riesgo_alto": "Specificity Riesgo Alto",
    })

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    sns.barplot(
        data=selected,
        x="valor",
        y="algoritmo",
        hue="metrica",
        ax=ax,
        palette=["#FFE15B", "#F0986E", "#A3D576"],
        edgecolor=TOKENS["ink"],
        linewidth=0.7,
    )
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Resultado one-vs-rest para la clase Riesgo Alto")
    ax.set_ylabel("")
    ax.legend(title="", loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=False)
    for container in ax.containers:
        ax.bar_label(container, labels=[f"{v.get_width():.1%}" for v in container], padding=3, fontsize=8)
    add_header(
        fig, ax,
        "Capacidad de detección de jugadores con Riesgo Alto",
        "Lectura one-vs-rest: sensibilidad, precisión y especificidad para la clase de mayor prioridad.",
    )
    fig.savefig(OUTPUT_DIR / "enfoque_riesgo_alto.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "enfoque_riesgo_alto.svg", bbox_inches="tight")
    plt.close(fig)


def plot_train_test(metrics: pd.DataFrame) -> None:
    selected = metrics.melt(
        id_vars="algoritmo",
        value_vars=["accuracy_train", "accuracy_test"],
        var_name="particion",
        value_name="accuracy",
    )
    selected["particion"] = selected["particion"].map({
        "accuracy_train": "Entrenamiento",
        "accuracy_test": "Prueba",
    })

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    sns.barplot(
        data=selected,
        x="accuracy",
        y="algoritmo",
        hue="particion",
        ax=ax,
        palette=["#C5CAD3", "#A3BEFA"],
        edgecolor=TOKENS["ink"],
        linewidth=0.7,
    )
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("")
    ax.legend(title="", loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False)
    for container in ax.containers:
        ax.bar_label(container, labels=[f"{v.get_width():.1%}" for v in container], padding=3, fontsize=8)
    add_header(
        fig, ax,
        "Accuracy de entrenamiento frente a prueba",
        "Brecha train-test para revisar posible sobreajuste con la misma partición estratificada.",
    )
    fig.savefig(OUTPUT_DIR / "accuracy_train_vs_test.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "accuracy_train_vs_test.svg", bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(matrices: dict, labels: list) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.ravel()
    for ax, (name, matrix) in zip(axes, matrices.items()):
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap=sns.blend_palette(["#FFFFFF", "#CEDFFE", "#A3BEFA"], as_cmap=True),
            cbar=False,
            linewidths=1,
            linecolor=TOKENS["panel"],
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_title(name, fontsize=11, fontweight="bold", color=TOKENS["ink"])
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
    fig.suptitle("Matrices de confusión por algoritmo", x=0.125, y=0.99, ha="left", fontsize=15, fontweight="bold", color=TOKENS["ink"])
    fig.text(0.125, 0.955, "Conteos del conjunto de prueba; filas = clase real, columnas = clase predicha.", ha="left", fontsize=10, color=TOKENS["muted"])
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUTPUT_DIR / "matrices_confusion_modelos.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "matrices_confusion_modelos.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    use_chart_theme()
    metrics, by_class, matrices, labels = evaluate_models()
    plot_metrics(metrics)
    plot_high_risk(metrics)
    plot_train_test(metrics)
    plot_confusion_matrices(matrices, labels)
    print(metrics.round(4).to_string(index=False))
    print(f"\nArchivos exportados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
