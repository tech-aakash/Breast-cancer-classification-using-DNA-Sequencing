# === Imports ===
import numpy as np
import pandas as pd
import tensorflow as tf  # type: ignore
import psutil
import os
import gc
import wandb
from wandb.integration.keras import WandbModelCheckpoint, WandbMetricsLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# === W&B Init ===
wandb.init(
    project="dna-sequence-cancer-classification",
    name="forward-embeddings-training-5",
    entity="laavanya-mishra094-svkm-s-narsee-monjee-institute-of-man",
    config={
        "architecture": "Dense_Deep_Classifier",
        "input_dim": 768,
        "optimizer": "adamw",
        "batch_size": 1024,
        "epochs": 100,
        "patience": 3,
        "dropout_rate": None,
        "layers": [4096, 2048, 1024, 512, 256, 128],
    }
)
config = wandb.config

# === System-aware configuration ===
NUM_CORES = os.cpu_count()
RAM_GB = psutil.virtual_memory().available / (1024 ** 3)
print(f"🧠 CPU Cores: {NUM_CORES}")
print(f"💾 Available RAM: {RAM_GB:.2f} GB")

# === Load and Prepare Data (FORWARD EMBEDDINGS) ===
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
df_combined_shuffled = df_combined.sample(frac=1, random_state=123).reset_index(drop=True)

# Cleanup
del emb_cancer, emb_noncan, ids_cancer, ids_noncan, labels_cancer, labels_noncan, ids_all, X_all, y_all
gc.collect()

# === Extract Features and Labels ===
X = np.stack(df_combined_shuffled["embedding"].values).astype(np.float32)
y = df_combined_shuffled["label"].values.astype(np.uint8)

del df_combined, df_combined_shuffled
gc.collect()

# === Train/Validation Split ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# === Define Model Architecture ===
def build_model(input_dim=768):
    from tensorflow.keras import Sequential, Input
    from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU

    model = Sequential()
    model.add(Input(shape=(input_dim,)))  # ✅ Proper Input layer

    for units in [4096, 2048, 1024, 512, 256, 128]:
        model.add(Dense(units))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

    model.add(Dense(1, activation='sigmoid'))
    return model

# === GPU Strategy ===
strategy = tf.distribute.MirroredStrategy()
print(f"🔌 Number of GPUs: {strategy.num_replicas_in_sync}")

with strategy.scope():
    model = build_model(input_dim=config.input_dim)

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=1e-4,
        weight_decay=1e-5
    )

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

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
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        WandbModelCheckpoint(filepath='forw5_best_model.keras', monitor='val_loss', save_best_only=True),
        WandbMetricsLogger()
    ]
)

# === Evaluate Model ===
y_pred_probs = model.predict(X_val).ravel()
y_pred = (y_pred_probs > 0.5).astype(int)

print("📊 Classification Report:\n", classification_report(y_val, y_pred))
print(f"📈 AUC-ROC: {roc_auc_score(y_val, y_pred_probs):.4f}")

# === Save Final Model ===
model.save("/home/azureuser/dna_sequencing/model_training/forw5_final_nn_model.keras")
print("💾 Saved model to: forw5_final_nn_model.keras")

# === Finish W&B Run ===
wandb.finish()
