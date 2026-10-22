from pathlib import Path
import json

import numpy as np
import pandas as pd


folder = Path(__file__).resolve().parent
results = folder / "results"
results.mkdir(exist_ok=True)
data = pd.read_excel(folder / "data/sales_dataset.xlsx")
empty_rows = int(data.isna().all(axis=1).sum())
missing = data.isna().sum().astype(int).to_dict()
negative = data.select_dtypes("number").lt(0).sum().astype(int).to_dict()
data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
checks = {
    "rows": len(data), "empty_rows": empty_rows, "missing": missing,
    "negative": negative, "invalid_dates": int(data.Date.isna().sum()),
    "duplicate_orders": int(data.OrderID.duplicated().sum()),
    "zero_prices": int(data.Price.eq(0).sum()),
    "zero_quantities": int(data.Quantity.eq(0).sum()),
}
assert empty_rows == 0 and sum(missing.values()) == 0
assert sum(negative.values()) == 0 and checks["invalid_dates"] == 0
assert checks["duplicate_orders"] == 0
data = data.sort_values(["Date", "OrderID"]).reset_index(drop=True)
data["Total"] = (data.Price * data.Quantity).round(2)
data["Month"] = data.Date.dt.strftime("%Y-%m")
products = data.groupby("Product").agg(
    Quantity=("Quantity", "sum"), Total=("Total", "sum")
).sort_values(["Quantity", "Total"], ascending=False)
categories = data.groupby("Category").agg(
    Quantity=("Quantity", "sum"), Total=("Total", "sum"), AveragePrice=("Price", "mean")
)
regions = data.groupby("CustomerRegion").Total.sum().sort_values(ascending=False)
monthly = data.groupby("Month").agg(
    Total=("Total", "sum"), Orders=("OrderID", "count"), Quantity=("Quantity", "sum")
)
category_monthly = data.pivot_table(
    index="Month", columns="Category", values="Total", aggfunc="sum", fill_value=0
).sort_index()
growth = pd.DataFrame({
    "January": category_monthly.iloc[0],
    "December": category_monthly.iloc[-1],
    "ChangePct": category_monthly.iloc[-1] / category_monthly.iloc[0] - 1,
    "GrowingMonths": category_monthly.diff().gt(0).sum(),
    "MonthlySlope": [np.polyfit(np.arange(len(category_monthly)), category_monthly[c], 1)[0]
                     for c in category_monthly.columns],
})
top_orders = data.sort_values(["Total", "OrderID"], ascending=[False, True]).head(10)
data.to_csv(results / "sales_sorted.csv", index=False)
for name, table in [
    ("products", products), ("categories", categories), ("regions", regions),
    ("monthly", monthly), ("category_monthly", category_monthly), ("growth", growth),
]:
    table.to_csv(results / (name + ".csv"))
top_orders.to_csv(results / "top10_orders.csv", index=False)
summary = {
    "checks": checks, "revenue": round(float(data.Total.sum()), 2),
    "quantity": int(data.Quantity.sum()), "best_month": monthly.Total.idxmax(),
    "best_month_revenue": round(float(monthly.Total.max()), 2),
    "top_products": products.head(5).reset_index().to_dict("records"),
    "categories": categories.reset_index().to_dict("records"),
    "regions": regions.reset_index().to_dict("records"),
    "monthly": monthly.reset_index().to_dict("records"),
    "growth": growth.reset_index().to_dict("records"),
}
(results / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print("Проверка данных:", checks)
print("Выручка:", summary["revenue"])
print("Лучший месяц:", summary["best_month"], summary["best_month_revenue"])
print("\nТоп-5 товаров:\n", products.head(5).to_string())
print("\nРост по категориям:\n", growth.round(3).to_string())
