# === Imports ===
import numpy as np
import pandas as pd
import tensorflow as tf
import psutil
import os
import gc
import wandb
from wandb.integration.keras import WandbModelCheckpoint, WandbMetricsLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

# === W&B Init ===
wandb.init(
    project="dna-sequence-cancer-classification",
    name="forward-embeddings-training-3",
    entity="laavanya-mishra094-svkm-s-narsee-monjee-institute-of-man",
    config={
        "architecture": "Dense_Deep_Classifier_Simplified",
        "input_dim": 768,
        "optimizer": "adam",
        "batch_size": 1024,
        "epochs": 100,
        "patience": 10,
        "dropout_rate": 0.3,
        "layers": [512, 256, 128, 64],
    }
)
config = wandb.config

# === System-aware configuration ===
NUM_CORES = os.cpu_count()
RAM_GB = psutil.virtual_memory().available / (1024 ** 3)
print(f"🧠 CPU Cores: {NUM_CORES}")
print(f"💾 Available RAM: {RAM_GB:.2f} GB")

# === Load and Prepare Data ===
emb_cancer = np.load("/home/azureuser/dna_sequencing/model_training/embeddings_forw_can.npy", mmap_mode='r')
ids_cancer = pd.read_csv("/home/azureuser/dna_sequencing/model_training/embeddings_ids.csv")["id"]
labels_cancer = np.ones(len(ids_cancer), dtype=int)

emb_noncan = np.load("/home/azureuser/dna_sequencing/model_training/embeddings_forw_noncan.npy", mmap_mode='r')
ids_noncan = pd.read_csv("/home/azureuser/dna_sequencing/model_training/embeddings_ids2.csv")["id"]
labels_noncan = np.zeros(len(ids_noncan), dtype=int)

X_all = np.vstack([emb_cancer, emb_noncan]).astype(np.float32)
y_all = np.concatenate([labels_cancer, labels_noncan])
ids_all = pd.concat([ids_cancer, ids_noncan], ignore_index=True)

df_combined = pd.DataFrame({
    "id": ids_all,
    "label": y_all,
    "embedding": list(X_all)
})
df_combined_shuffled = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Cleanup
del emb_cancer, emb_noncan, ids_cancer, ids_noncan, labels_cancer, labels_noncan, ids_all, X_all, y_all
gc.collect()

# === Extract Features and Labels ===
X = np.stack(df_combined_shuffled["embedding"].values).astype(np.float32)
y = df_combined_shuffled["label"].values.astype(np.uint8)

del df_combined, df_combined_shuffled
gc.collect()

# === Normalize Embeddings ===
scaler = StandardScaler()
X = scaler.fit_transform(X)

# === Train/Validation Split ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# === Compute Class Weights ===
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# === Define Model Architecture ===
def build_model(input_dim=768, dropout_rate=0.3):
    from tensorflow.keras import Sequential, Input
    from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU

    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    for units in config.layers:
        model.add(Dense(units))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid'))
    return model

# === GPU Strategy ===
strategy = tf.distribute.MirroredStrategy()
print(f"🔌 Number of GPUs: {strategy.num_replicas_in_sync}")

with strategy.scope():
    model = build_model(input_dim=config.input_dim, dropout_rate=config.dropout_rate)
    model.compile(
        optimizer=config.optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), tf.keras.metrics.AUC(name='auc')]
    )

# === tf.data pipeline ===
def create_dataset(X, y, batch_size=config.batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_ds = create_dataset(X_train, y_train)
val_ds = create_dataset(X_val, y_val)

# === Train Model ===
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=config.epochs,
    class_weight=class_weights,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=config.patience, restore_best_weights=True),
        WandbModelCheckpoint(filepath='forw2_best_model.keras', monitor='val_auc', mode='max', save_best_only=True),
        WandbMetricsLogger()
    ]
)

# === Evaluate Model ===
y_pred_probs = model.predict(X_val).ravel()
y_pred = (y_pred_probs > 0.5).astype(int)

print("📊 Classification Report:\n", classification_report(y_val, y_pred))
print(f"📈 AUC-ROC: {roc_auc_score(y_val, y_pred_probs):.4f}")

# === Save Final Model ===
model.save("forw3_final_nn_model.keras")
print("💾 Saved model to: forw3_final_nn_model.keras")

# === Finish W&B Run ===
wandb.finish()