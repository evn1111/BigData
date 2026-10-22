from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
    f1_score, log_loss, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


folder = Path(__file__).resolve().parent
results = folder / "results"
figures = results / "figures"
figures.mkdir(parents=True, exist_ok=True)
data = pd.read_csv(folder / "data/customers.csv")
rng = np.random.default_rng(42)
score = (
    -0.6 + 0.16 * (data["PurchaseFrequency"] - 7)
    + 0.006 * (data["AveragePurchaseValue"] - 140)
    - 0.012 * (data["Age"] - 42)
    + 0.9 * ((data["PurchaseFrequency"] >= 8) & (data["AveragePurchaseValue"] >= 140))
)
data["Purchase"] = rng.binomial(1, expit(score))
data.to_csv(folder / "data/customers_with_purchase.csv", index=False)
assert data["CustomerID"].is_unique
assert not data.isna().any().any()
assert set(data["Purchase"]) == {0, 1}

X = data[["Age", "AveragePurchaseValue", "PurchaseFrequency"]]
y = data["Purchase"]
X_rest, X_test, y_rest, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_rest, y_rest, test_size=0.25, stratify=y_rest, random_state=42
)
scaler = StandardScaler()
train = scaler.fit_transform(X_train)
validation = scaler.transform(X_val)
test = scaler.transform(X_test)
split = pd.Series("train", index=data.index)
split.loc[X_val.index] = "validation"
split.loc[X_test.index] = "test"
pd.DataFrame({"CustomerID": data["CustomerID"], "Split": split}).to_csv(
    results / "splits.csv", index=False
)

linear = LogisticRegression(random_state=42, max_iter=1000)
linear.fit(train, y_train)
network = MLPClassifier(
    hidden_layer_sizes=(16, 8), activation="relu", solver="adam",
    learning_rate_init=0.003, alpha=0.01, batch_size=32, random_state=42,
)
history = []
best_loss = np.inf
best_epoch = 0
best_network = None
for epoch in range(1, 201):
    network.partial_fit(train, y_train, classes=[0, 1])
    train_probability = network.predict_proba(train)[:, 1]
    val_probability = network.predict_proba(validation)[:, 1]
    val_loss = log_loss(y_val, val_probability)
    history.append({
        "Epoch": epoch,
        "TrainLoss": log_loss(y_train, train_probability),
        "ValidationLoss": val_loss,
        "TrainAccuracy": accuracy_score(y_train, train_probability >= 0.5),
        "ValidationAccuracy": accuracy_score(y_val, val_probability >= 0.5),
    })
    if val_loss < best_loss:
        best_loss = val_loss
        best_epoch = epoch
        best_network = deepcopy(network)
history = pd.DataFrame(history)
history.to_csv(results / "learning_history.csv", index=False)

models = {"Логистическая регрессия": linear, "Нейронная сеть": best_network}
predictions = data.loc[X_test.index].copy()
metrics = []
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
roc_fig, roc_ax = plt.subplots(figsize=(7, 5))
for ax, (name, model) in zip(axes, models.items()):
    probability = model.predict_proba(test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, prediction, labels=[0, 1]).ravel()
    metrics.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, prediction),
        "Precision": precision_score(y_test, prediction, zero_division=0),
        "Recall": recall_score(y_test, prediction, zero_division=0),
        "F1": f1_score(y_test, prediction, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, probability),
        "LogLoss": log_loss(y_test, probability),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    })
    predictions[name + "_probability"] = probability
    predictions[name + "_prediction"] = prediction
    predictions[name + "_error"] = np.where(
        (prediction == 1) & (y_test == 0), "FP",
        np.where((prediction == 0) & (y_test == 1), "FN", "correct"),
    )
    ConfusionMatrixDisplay.from_predictions(
        y_test, prediction, labels=[0, 1], display_labels=["Нет", "Да"],
        colorbar=False, ax=ax,
    )
    ax.set_title(name)
    ax.set_xlabel("Прогноз")
    ax.set_ylabel("Факт покупки")
    fpr, tpr, _ = roc_curve(y_test, probability)
    roc_ax.plot(fpr, tpr, label=f"{name}, AUC={roc_auc_score(y_test, probability):.3f}")
metrics = pd.DataFrame(metrics)
metrics.to_csv(results / "metrics.csv", index=False)
predictions.to_csv(results / "test_predictions.csv", index=False)
errors = predictions.loc[
    (predictions["Логистическая регрессия_error"] != "correct")
    | (predictions["Нейронная сеть_error"] != "correct")
]
errors.to_csv(results / "errors.csv", index=False)
data.groupby("Purchase").size().to_csv(results / "class_counts.csv", header=["Count"])
pd.DataFrame([{
    "BestEpoch": best_epoch, "ValidationLogLoss": best_loss,
    "TrainSize": len(train), "ValidationSize": len(validation), "TestSize": len(test),
    "MajorityAccuracy": accuracy_score(y_test, np.repeat(y_train.mode()[0], len(y_test))),
}]).to_csv(results / "training_summary.csv", index=False)
fig.tight_layout()
fig.savefig(figures / "02_confusion_matrices.png", dpi=150)
plt.close(fig)
roc_ax.plot([0, 1], [0, 1], "k--")
roc_ax.set_xlabel("Доля ложных срабатываний")
roc_ax.set_ylabel("Доля найденных покупок")
roc_ax.legend()
roc_fig.tight_layout()
roc_fig.savefig(figures / "03_roc.png", dpi=150)
plt.close(roc_fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(history["Epoch"], history["TrainLoss"], label="Обучение")
axes[0].plot(history["Epoch"], history["ValidationLoss"], label="Проверка")
axes[0].set_ylabel("Log loss")
axes[1].plot(history["Epoch"], history["TrainAccuracy"], label="Обучение")
axes[1].plot(history["Epoch"], history["ValidationAccuracy"], label="Проверка")
axes[1].set_ylabel("Accuracy")
for ax in axes:
    ax.axvline(best_epoch, color="gray", linestyle="--", label=f"Выбрана эпоха {best_epoch}")
    ax.set_xlabel("Эпоха")
    ax.legend()
fig.tight_layout()
fig.savefig(figures / "01_learning.png", dpi=150)
plt.close(fig)

print("Классы:", y.value_counts().to_dict())
print("Лучшая эпоха:", best_epoch)
print(metrics.round(4).to_string(index=False))
