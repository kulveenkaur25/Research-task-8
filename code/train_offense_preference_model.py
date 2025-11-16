from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


TRAIN_DATA_PATH = Path("results/training_data_for_model.csv")
OUT_MODEL_SUMMARY = Path("results/model_summary.txt")


def main():
    print(f"📂 Loading training data from: {TRAIN_DATA_PATH}")
    df = pd.read_csv(TRAIN_DATA_PATH)

    # Drop any rows where target is missing, just in case
    df = df.dropna(subset=["llm_prefers_teamA"])

    # Target: 1 = LLM prefers Team A, 0 = prefers Team B
    y = df["llm_prefers_teamA"].astype(int)

    # Features: all diff_* columns
    feature_cols = [c for c in df.columns if c.startswith("diff_")]
    X = df[feature_cols]

    print("\n🧮 Using features:")
    print(feature_cols)

    # Small dataset, so keep test set small but non-zero
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    # Pipeline: standardize features + logistic regression
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42)),
        ]
    )

    print("\n🏋️ Training logistic regression model...")
    pipe.fit(X_train, y_train)

    # ---- Evaluation ----
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)

    print("\n📊 Evaluation on test set:")
    print(f"Accuracy: {acc:.3f}")
    print("Confusion matrix (rows = true, cols = predicted):")
    print(cm)
    print("\nClassification report:")
    print(report)

    # ---- Feature importance (coefficients) ----
    clf = pipe.named_steps["clf"]
    coefs = clf.coef_[0]

    coef_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "coef": coefs,
        }
    )
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False)

    print("\n⭐ Feature importance (larger |coef| = more influence):")
    print(coef_df[["feature", "coef"]])

    # ---- Save a text summary for your report ----
    with open(OUT_MODEL_SUMMARY, "w", encoding="utf-8") as f:
        f.write("Offense Preference Model (Logistic Regression)\n")
        f.write("=================================================\n\n")
        f.write(f"Features used:\n")
        for c in feature_cols:
            f.write(f"  - {c}\n")

        f.write("\nTest accuracy:\n")
        f.write(f"  {acc:.3f}\n\n")

        f.write("Confusion matrix (rows = true, cols = predicted):\n")
        f.write(str(cm) + "\n\n")

        f.write("Classification report:\n")
        f.write(report + "\n")

        f.write("\nFeature coefficients:\n")
        f.write("  (Positive coef => higher value for Team A makes model more likely\n")
        f.write("   to choose Team A as better offense.)\n\n")
        for _, row in coef_df.iterrows():
            f.write(f"  {row['feature']}: {row['coef']:.3f}\n")

    print(f"\n📝 Saved model summary to: {OUT_MODEL_SUMMARY}")
    print("🎉 Step complete: you now have a trained surrogate model + summary.")


if __name__ == "__main__":
    main()
