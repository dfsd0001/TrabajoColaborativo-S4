# 🔍 XAI — Mantenimiento Predictivo con Explicabilidad

> **Taller Práctico | Ética, Sesgo y Calidad en el Aprendizaje Automático**  
> Técnicas de Inteligencia Artificial Explicable (XAI) aplicadas a un caso real de mantenimiento predictivo industrial.

---

## 📌 Descripción del Problema

Este proyecto aplica modelos de Machine Learning supervisado para **predecir fallas en maquinaria industrial**, utilizando el dataset público **AI4I Predictive Maintenance Dataset** (10,000 registros). Se implementan y comparan tres modelos, y se aplican cuatro técnicas de explicabilidad (XAI) para garantizar transparencia, detectar sesgos y reflexionar sobre los principios éticos del sistema.

---

## 🎯 Objetivos

- Implementar y comparar múltiples modelos de ML supervisado
- Aplicar técnicas XAI para mejorar la transparencia del modelo
- Detectar y mitigar sesgos en los datos
- Reflexionar sobre los riesgos éticos de sistemas automatizados de decisión
- Documentar el flujo completo del proyecto de forma reproducible

---

## 🗂️ Estructura del Repositorio

```
xai_predictive_maintenance/
│
├── 📁 data/
│   └── ai4i_predictive_maintenance.csv       # Dataset original
│
├── 📁 notebooks/
│   └── xai_predictive_maintenance.ipynb      # Notebook principal (ejecutable en VS Code)
│
├── 📁 outputs/
│   ├── figures/                              # Visualizaciones generadas
│   ├── models/                              # Modelos entrenados (.pkl)
│   └── reports/                             # Reportes de métricas
│
├── 📁 src/
│   └── utils.py                             # Funciones auxiliares reutilizables
│
├── requirements.txt                         # Dependencias del proyecto
├── .gitignore                               # Archivos ignorados por Git
└── README.md                                # Este archivo
```

---

## 🤖 Modelos Implementados

| Modelo | Rol en el proyecto | Técnica XAI aplicada |
|---|---|---|
| **Regresión Logística** | Baseline interpretable | Interpretación de coeficientes |
| **Árbol de Decisión** | Explicabilidad visual | Visualización del árbol |
| **Random Forest** | Mejor rendimiento | SHAP Values + Permutation Importance |

---

## 🔬 Técnicas XAI Aplicadas

1. **Interpretación de Coeficientes** — Regresión Logística
2. **Visualización del Árbol de Decisión** — Árbol de Decisión
3. **SHAP Values (Shapley)** — Random Forest (global e individual)
4. **Permutation Feature Importance** — Comparativa entre modelos

---

## ⚙️ Instalación y Ejecución

### 1. Clonar el repositorio
```bash
git clone https://github.com/TU_USUARIO/xai_predictive_maintenance.git
cd xai_predictive_maintenance
```

### 2. Crear entorno virtual (recomendado)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Abrir el notebook en VS Code
```bash
code notebooks/xai_predictive_maintenance.ipynb
```
> Asegúrate de tener instalada la extensión **Jupyter** en VS Code y seleccionar el kernel del entorno virtual creado.

---

## 📦 Dataset

**AI4I 2020 Predictive Maintenance Dataset**

| Característica | Detalle |
|---|---|
| Registros | 10,000 |
| Variables | 8 (6 numéricas, 1 categórica, 1 objetivo) |
| Variable objetivo | `Machine failure` (binaria: 0/1) |
| Desbalance | 96.6% sin falla / 3.4% con falla |
| Valores nulos | Ninguno |

**Variables del dataset:**

| Variable | Descripción | Tipo |
|---|---|---|
| `UDI` | Identificador único | Entero |
| `Type` | Tipo de máquina (L/M/H) | Categórica |
| `Air temperature [K]` | Temperatura del aire en Kelvin | Numérica |
| `Process temperature [K]` | Temperatura del proceso en Kelvin | Numérica |
| `Rotational speed [rpm]` | Velocidad rotacional | Numérica |
| `Torque [Nm]` | Torque en Newton-metro | Numérica |
| `Tool wear [min]` | Desgaste de herramienta en minutos | Numérica |
| `Machine failure` | Variable objetivo: falla (1) / no falla (0) | Binaria |

---

## 📊 Resultados Principales

> Los resultados completos se encuentran en el notebook y en `outputs/reports/`.

- **Random Forest** obtuvo el mejor rendimiento general (F1-score en clase minoritaria)
- **SHAP** identificó `Torque` y `Tool wear` como variables más influyentes
- Se detectó **desbalance de clases severo** (3.4% positivos) → mitigado con SMOTE
- Se identificaron posibles sesgos asociados al `Type` de máquina

---

## ⚖️ Reflexión Ética

El sistema automatizado de detección de fallas presenta riesgos éticos si se implementa sin explicabilidad:

- **Falta de transparencia**: decisiones opacas sin justificación comprensible
- **Sesgo por tipo de máquina**: el modelo puede discriminar entre tipos L, M y H
- **Coste asimétrico del error**: una falla no detectada (falso negativo) tiene mayor costo que una falsa alarma
- **Dependencia excesiva**: operadores podrían confiar ciegamente en el sistema

> Ver análisis completo en la sección de reflexión ética del notebook.

---

## 🛠️ Tecnologías Utilizadas

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)
![SHAP](https://img.shields.io/badge/SHAP-0.44+-red)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)

- Python 3.9+
- scikit-learn, imbalanced-learn (SMOTE)
- SHAP, LIME
- matplotlib, seaborn, plotly
- pandas, numpy
- joblib

---

## 👥 Autores

| Nombre | Rol |
|---|---|
| [Tu Nombre] | Desarrollo y análisis |

---

## 📄 Licencia

Este proyecto es de uso académico. Dataset original disponible en [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset).
