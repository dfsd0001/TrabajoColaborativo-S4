"""
utils.py — Funciones auxiliares para el proyecto XAI Predictive Maintenance
============================================================================
Contiene funciones reutilizables para preprocesamiento, evaluación,
visualización y exportación de resultados.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score
)


# ──────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL DE ESTILO
# ──────────────────────────────────────────────

PALETTE = {
    "primary":   "#2563EB",   # Azul principal
    "secondary": "#7C3AED",   # Violeta
    "success":   "#16A34A",   # Verde
    "danger":    "#DC2626",   # Rojo
    "warning":   "#D97706",   # Ámbar
    "gray":      "#6B7280",   # Gris neutro
}

def set_style():
    """Aplica estilo visual consistente a todas las gráficas del proyecto."""
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.titlesize":    14,
        "axes.titleweight":  "bold",
        "axes.labelsize":    12,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "font.family":       "sans-serif",
        "figure.dpi":        120,
    })
    sns.set_theme(style="whitegrid", palette="muted")


# ──────────────────────────────────────────────
# EVALUACIÓN DE MODELOS
# ──────────────────────────────────────────────

def evaluar_modelo(nombre, modelo, X_test, y_test, umbral=0.5):
    """
    Evalúa un modelo de clasificación binaria y retorna un diccionario
    con las métricas principales.

    Parámetros
    ----------
    nombre   : str   — Nombre del modelo para mostrar en logs
    modelo   : obj   — Modelo entrenado con método predict_proba o decision_function
    X_test   : array — Features de prueba
    y_test   : array — Etiquetas reales
    umbral   : float — Umbral de clasificación (default 0.5)

    Retorna
    -------
    dict con accuracy, precision, recall, f1, roc_auc
    """
    if hasattr(modelo, "predict_proba"):
        y_proba = modelo.predict_proba(X_test)[:, 1]
    else:
        y_proba = modelo.decision_function(X_test)

    y_pred = (y_proba >= umbral).astype(int)

    reporte = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    metricas = {
        "Modelo":     nombre,
        "Accuracy":   round(reporte["accuracy"], 4),
        "Precision":  round(reporte["1"]["precision"], 4),
        "Recall":     round(reporte["1"]["recall"], 4),
        "F1-Score":   round(reporte["1"]["f1-score"], 4),
        "ROC-AUC":    round(auc, 4),
    }

    print(f"\n{'='*50}")
    print(f"  {nombre}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"  ROC-AUC: {auc:.4f}")

    return metricas, y_pred, y_proba


def tabla_comparativa(lista_metricas):
    """
    Genera una tabla comparativa de métricas entre modelos.

    Parámetros
    ----------
    lista_metricas : list[dict] — Lista de diccionarios retornados por evaluar_modelo

    Retorna
    -------
    DataFrame ordenado por F1-Score descendente
    """
    df = pd.DataFrame(lista_metricas).set_index("Modelo")
    df_sorted = df.sort_values("F1-Score", ascending=False)

    # Resaltar el mejor valor por columna
    styled = df_sorted.style \
        .highlight_max(axis=0, props="background-color: #bbf7d0; font-weight: bold") \
        .format("{:.4f}")

    return df_sorted, styled


# ──────────────────────────────────────────────
# VISUALIZACIONES
# ──────────────────────────────────────────────

def plot_confusion_matrix(y_test, y_pred, nombre_modelo, save_path=None):
    """Genera y guarda una matriz de confusión con formato visual claro."""
    set_style()
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Sin falla (0)", "Falla (1)"],
        yticklabels=["Sin falla (0)", "Falla (1)"],
        ax=ax, linewidths=0.5, linecolor="white"
    )
    ax.set_title(f"Matriz de Confusión — {nombre_modelo}")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor Real")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_roc_curves(modelos_info, X_test, y_test, save_path=None):
    """
    Genera curvas ROC superpuestas para comparar múltiples modelos.

    Parámetros
    ----------
    modelos_info : list[tuple] — Lista de (nombre, modelo)
    X_test, y_test : arrays de prueba
    save_path : str opcional — Ruta para guardar la figura
    """
    set_style()
    colores = list(PALETTE.values())
    fig, ax = plt.subplots(figsize=(7, 5))

    for i, (nombre, modelo) in enumerate(modelos_info):
        if hasattr(modelo, "predict_proba"):
            y_proba = modelo.predict_proba(X_test)[:, 1]
        else:
            y_proba = modelo.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, color=colores[i], lw=2,
                label=f"{nombre} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Clasificador aleatorio")
    ax.set_xlabel("Tasa de Falsos Positivos")
    ax.set_ylabel("Tasa de Verdaderos Positivos")
    ax.set_title("Comparación de Curvas ROC")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_distribucion_objetivo(y, save_path=None):
    """Grafica la distribución de la variable objetivo para visualizar el desbalance."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    conteos = pd.Series(y).value_counts()
    etiquetas = ["Sin falla (0)", "Falla (1)"]
    colores = [PALETTE["primary"], PALETTE["danger"]]

    # Barras
    axes[0].bar(etiquetas, conteos.values, color=colores, edgecolor="white", width=0.5)
    axes[0].set_title("Distribución de Clases")
    axes[0].set_ylabel("Cantidad de registros")
    for i, v in enumerate(conteos.values):
        axes[0].text(i, v + 50, str(v), ha="center", fontweight="bold")

    # Pie
    axes[1].pie(conteos.values, labels=etiquetas, colors=colores,
                autopct="%1.1f%%", startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Proporción de Clases")

    plt.suptitle("⚠️  Desbalance de Clases Detectado", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_tabla_metricas(df_metricas, save_path=None):
    """Genera una tabla visual con colores para la comparación de modelos."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.axis("off")

    col_labels = list(df_metricas.columns)
    row_labels  = list(df_metricas.index)
    cell_text   = df_metricas.values.tolist()

    tabla = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center"
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(11)
    tabla.scale(1.2, 2)

    # Colorear encabezados
    for (row, col), cell in tabla.get_celld().items():
        if row == 0 or col == -1:
            cell.set_facecolor(PALETTE["primary"])
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#EFF6FF")

    ax.set_title("📊 Tabla Comparativa de Modelos", fontsize=14,
                 fontweight="bold", pad=20)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# ──────────────────────────────────────────────
# UTILIDADES
# ──────────────────────────────────────────────

def guardar_modelo(modelo, nombre_archivo, carpeta="outputs/models"):
    """Serializa y guarda un modelo entrenado con joblib."""
    import joblib
    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, nombre_archivo)
    joblib.dump(modelo, ruta)
    print(f"✅ Modelo guardado en: {ruta}")
    return ruta


def cargar_modelo(nombre_archivo, carpeta="outputs/models"):
    """Carga un modelo serializado con joblib."""
    import joblib
    ruta = os.path.join(carpeta, nombre_archivo)
    modelo = joblib.load(ruta)
    print(f"✅ Modelo cargado desde: {ruta}")
    return modelo


def guardar_figura(fig, nombre, carpeta="outputs/figures"):
    """Guarda una figura matplotlib en la carpeta de outputs."""
    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, nombre)
    fig.savefig(ruta, bbox_inches="tight", dpi=150)
    print(f"✅ Figura guardada en: {ruta}")
