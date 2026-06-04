import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import joblib

print("Loading upgraded hybrid dataset...")
train_df = pd.read_csv("geometric_train_data.csv")
test_df = pd.read_csv("geometric_test_data.csv")

X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

print(f"Training on 77 feature layout across {len(X_train)} samples...")

print("Encoding labels...")
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_encoded)
print("Training XGBoost Classifier... 🚀")
model = XGBClassifier(
    n_estimators=300, 
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42, 
    n_jobs=-1
)

model.fit(X_train, y_train_encoded, sample_weight=sample_weights)

print("\n" + "="*40)
print("FINAL HYBRID XGBOOST EVALUATION")
print("="*40)

predictions_encoded = model.predict(X_test)
predictions = encoder.inverse_transform(predictions_encoded)

print(classification_report(y_test, predictions))

cm = confusion_matrix(y_test_encoded, predictions_encoded)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)

fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(cmap='Blues', ax=ax, xticks_rotation=45)

plt.title('Hybrid XGBoost Confusion Matrix')
plt.tight_layout()
plt.show()

joblib.dump(model, 'emotion_xgb_model.pkl')
joblib.dump(encoder, 'label_encoder.pkl')
print("\nSuccess! Saved 'emotion_xgb_model.pkl' and 'label_encoder.pkl'")