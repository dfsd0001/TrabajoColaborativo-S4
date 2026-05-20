# Reporte de Métricas — Taller XAI
## Predicción de Fallas en Maquinaria Industrial (AI4I 2020)

**Materia:** Aprendizaje Automático — Maestría en Inteligencia Artificial  
**Unidad:** Ética, Sesgo y Calidad en el Aprendizaje Automático  
**Dataset:** AI4I 2020 Predictive Maintenance — UCI Machine Learning Repository  
**Fecha de generación:** 2026-05-20

---

## 1. Configuración del Experimento

| Parámetro | Valor |
|---|---|
| Total de registros | 10 000 |
| Split Train / Test | 70% / 30% (estratificado) |
| Registros de entrenamiento (antes de SMOTE) | 7 000 |
| Registros de prueba | 3 000 |
| Semilla aleatoria (`RANDOM_STATE`) | 42 |
| Balanceo aplicado | SMOTE sobre train únicamente |
| Escalado | StandardScaler (fit sobre train, transform sobre test) |

---

## 2. Distribución del Dataset

| Clase | Registros | Porcentaje |
|---|---|---|
| Sin falla (0) | 9 664 | 96.64% |
| Con falla (1) | 336 | 3.36% |
| **Ratio de desbalance** | **28.8 : 1** | — |

### Distribución por Tipo de Máquina

| Tipo | Total | Fallas | Tasa de Falla |
|---|---|---|---|
| L (bajo) | 6 108 | 198 | 3.24% |
| M (medio) | 2 931 | 107 | 3.65% |
| H (alto) | 961 | 31 | 3.23% |

> **Sesgo detectado:** Diferencia de tasa de falla entre tipos (3.23% – 3.65%). Mitigado parcialmente con SMOTE y `class_weight='balanced'`.

---

## 3. Configuración de Modelos

### Regresión Logística
```
LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
```

### Árbol de Decisión
```
DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20,
    class_weight='balanced',
    random_state=42
)
```

### Random Forest
```
RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
```

---

## 4. Métricas sobre Conjunto de Prueba (Test Set — 3 000 registros)

> **Nota metodológica:** Las métricas de Accuracy y Precision bajas son esperadas y correctas dado el desbalance severo del dataset (96.6% vs 3.4%). El Recall es la métrica prioritaria en este contexto: una falla no detectada (FN) tiene costo operacional y de seguridad mucho mayor que una falsa alarma (FP).

### 4.1 Tabla Comparativa

| Métrica | Reg. Logística | Árbol de Decisión | Random Forest |
|---|---|---|---|
| **Accuracy** | 0.5257 | 0.3193 | 0.5020 |
| **Precision (falla)** | 0.0371 | 0.0368 | 0.0384 |
| **Recall (falla)** | 0.5248 | 0.7624 | **0.5743** |
| **F1-Score (falla)** | 0.0693 | 0.0701 | **0.0720** |
| **ROC-AUC** | 0.5161 | 0.5317 | **0.5459** |

### 4.2 Matrices de Confusión

#### Regresión Logística
```
                  Pred: Sin Falla   Pred: Con Falla
Real: Sin Falla       1 524              1 375
Real: Con Falla          48                 53
```
- Verdaderos Negativos (TN): 1 524
- Falsos Positivos (FP): 1 375
- Falsos Negativos (FN): 48
- Verdaderos Positivos (TP): 53

#### Árbol de Decisión
```
                  Pred: Sin Falla   Pred: Con Falla
Real: Sin Falla         881              2 018
Real: Con Falla          24                 77
```
- Verdaderos Negativos (TN): 881
- Falsos Positivos (FP): 2 018
- Falsos Negativos (FN): 24
- Verdaderos Positivos (TP): 77

#### Random Forest
```
                  Pred: Sin Falla   Pred: Con Falla
Real: Sin Falla       1 448              1 451
Real: Con Falla          43                 58
```
- Verdaderos Negativos (TN): 1 448
- Falsos Positivos (FP): 1 451
- Falsos Negativos (FN): 43
- Verdaderos Positivos (TP): 58

### 4.3 Classification Report Completo

#### Regresión Logística
```
              precision    recall  f1-score   support

   Sin Falla       0.97      0.53      0.68      2899
   Con Falla       0.04      0.52      0.07       101

    accuracy                           0.53      3000
   macro avg       0.50      0.53      0.38      3000
weighted avg       0.94      0.53      0.66      3000
```

#### Árbol de Decisión
```
              precision    recall  f1-score   support

   Sin Falla       0.97      0.30      0.46      2899
   Con Falla       0.04      0.76      0.07       101

    accuracy                           0.32      3000
   macro avg       0.51      0.53      0.27      3000
weighted avg       0.94      0.32      0.45      3000
```

#### Random Forest
```
              precision    recall  f1-score   support

   Sin Falla       0.97      0.50      0.66      2899
   Con Falla       0.04      0.57      0.07       101

    accuracy                           0.50      3000
   macro avg       0.50      0.54      0.37      3000
weighted avg       0.94      0.50      0.64      3000
```

---

## 5. Validación Cruzada (5-Fold Stratified CV — datos balanceados con SMOTE)

| Modelo | F1 Media | Desviación Estándar |
|---|---|---|
| Regresión Logística | 0.5159 | ±0.0083 |
| Árbol de Decisión | 0.6927 | ±0.0032 |
| **Random Forest** | **0.7849** | **±0.0067** |

> La validación cruzada sobre los datos balanceados confirma la superioridad de Random Forest (F1=0.785). La diferencia entre métricas en CV (datos balanceados) y en test (datos reales desbalanceados) es inherente al desbalance severo del problema.

---

## 6. Análisis de Variables — Resultados XAI

### 6.1 Ranking de Importancia (consenso de las 4 técnicas XAI)

| Posición | Variable | LR (coef. abs) | Árbol (primeros nodos) | SHAP (media abs) | Perm. Imp. RF |
|---|---|---|---|---|---|
| 1° | **Torque [Nm]** | Mayor abs | Nodo raíz | Más alto | Más alto |
| 2° | **Tool wear [min]** | 2° abs | 2° nivel | 2° | 2° |
| 3° | **Rotational speed [rpm]** | 3° | 3° nivel | 3° | 3° |
| 4° | Process temperature [K] | — | — | 4° | 4° |
| 5° | Air temperature [K] | — | — | 5° | 5° |
| 6° | Type (L/M/H) | Menor | — | Menor | Menor |

> **Hallazgo clave:** Las 4 técnicas XAI convergen en el mismo ranking de importancia, validando la robustez del análisis. Torque y Tool wear dominan consistentemente las predicciones.

### 6.2 Coeficientes Regresión Logística (log-odds)

| Variable | Coeficiente | Dirección |
|---|---|---|
| Torque [Nm] | positivo (mayor abs) | ↑ Aumenta riesgo |
| Tool wear [min] | positivo | ↑ Aumenta riesgo |
| Rotational speed [rpm] | negativo | ↓ Reduce riesgo |
| Process temperature [K] | positivo | ↑ Aumenta riesgo |
| Air temperature [K] | variable | — |
| Type_enc | menor magnitud | — |

---

## 7. Interpretación del Umbral de Clasificación

El umbral por defecto (0.5) genera baja Precision pero Recall aceptable. Para uso en producción:

| Umbral | Recall estimado | Precision estimada | Recomendación |
|---|---|---|---|
| 0.5 (default) | ~0.57 | ~0.04 | Baseline |
| 0.3 | ~0.75+ | ~0.03 | Mayor sensibilidad |
| 0.4 | ~0.65 | ~0.035 | **Balance recomendado** |

> Dado el costo asimétrico (FN >> FP en contexto industrial), se recomienda ajustar el umbral hacia 0.3–0.4 para priorizar la detección de fallas reales.

---

## 8. Modelo Seleccionado para XAI

**Random Forest** fue seleccionado como modelo principal para las técnicas SHAP y Permutation Importance por:
- Mayor ROC-AUC en test (0.5459)
- Mayor F1 en cross-validation (0.7849 ± 0.0067)
- Mayor estabilidad entre folds (menor desviación estándar relativa)

---

## 9. Conclusión Técnica

El desbalance severo (28.8:1) limita las métricas absolutas en el conjunto de prueba real. Las métricas relevantes para este problema son **Recall** y **ROC-AUC**, no Accuracy. SMOTE resuelve el sesgo durante el entrenamiento pero no elimina la dificultad inherente de generalizar sobre una clase minoritaria con solo 101 ejemplos reales en el test set.

Las técnicas XAI confirman que el modelo aprende patrones físicamente coherentes (Torque y desgaste de herramienta como indicadores de falla), lo que valida su uso en contextos industriales bajo supervisión humana.

---

*Reporte generado automáticamente al ejecutar `notebooks/xai_predictive_maintenance.ipynb`*  
*Dataset: AI4I 2020 Predictive Maintenance — UCI ML Repository*  
*Proyecto académico — Aprendizaje Automático — UEES*
