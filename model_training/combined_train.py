import numpy as np
import pandas as pd
import tensorflow as tf
import psutil
import os
import gc
import wandb
from wandb.integration.keras import WandbModelCheckpoint, WandbMetricsLogger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, roc_auc_score

# === W&B Init ===
wandb.init(
    project="dna-sequence-cancer-classification",
    name="forward-backward-embeddings-training",
    entity="laavanya-mishra094-svkm-s-narsee-monjee-institute-of-man",
    config={
        "architecture": "Combined_Embeddings_Classifier",
        "input_dim": 1536,
        "optimizer": "adam",
        "batch_size": 1024,
        "epochs": 100,
        "patience": 10,
        "dropout_rate": 0.3,
        "layers": [1024, 512, 256, 128],
    }
)
config = wandb.config

# === System Info ===
NUM_CORES = os.cpu_count()
RAM_GB = psutil.virtual_memory().available / (1024 ** 3)
print(f"🧠 CPU Cores: {NUM_CORES}")
print(f"💾 Available RAM: {RAM_GB:.2f} GB")

# === Load Forward + Backward Embeddings ===
fwd_can = np.load("/home/azureuser/dna_sequencing/model_training/embeddings_forw_can.npy", mmap_mode='r')
bwd_can = np.load("/home/azureuser/dna_sequencing/model_training/embeddings_backw_can.npy", mmap_mode='r')
labels_can = np.ones(len(fwd_can), dtype=int)

fwd_noncan = np.load("/home/azureuser/dna_sequencing/model_training/embeddings_forw_noncan.npy", mmap_mode='r')
bwd_noncan = np.load("/home/azureuser/dna_sequencing/model_training/embeddings_backw_noncan.npy", mmap_mode='r')
labels_noncan = np.zeros(len(fwd_noncan), dtype=int)

# === Combine Embeddings (Forward + Backward) ===
X_can = np.hstack([fwd_can, bwd_can])
X_noncan = np.hstack([fwd_noncan, bwd_noncan])

X_all = np.vstack([X_can, X_noncan]).astype(np.float32)
y_all = np.concatenate([labels_can, labels_noncan])

# === Normalize Embeddings ===
scaler = StandardScaler()
X_all = scaler.fit_transform(X_all)

# === Train/Validation Split ===
X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)

# === Class Weights ===
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# === Define Model ===
def build_model(input_dim=1536, dropout_rate=0.3):
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

# === Dataset Creation ===
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
        WandbModelCheckpoint(filepath='forwback_best_model.keras', monitor='val_auc', mode='max', save_best_only=True),
        WandbMetricsLogger()
    ]
)

# === Evaluate ===
y_pred_probs = model.predict(X_val).ravel()
y_pred = (y_pred_probs > 0.5).astype(int)

print("📊 Classification Report:\n", classification_report(y_val, y_pred))
print(f"📈 AUC-ROC: {roc_auc_score(y_val, y_pred_probs):.4f}")

# === Save Model ===
model.save("forwback_final_nn_model.keras")
print("💾 Saved model to: forwback_final_nn_model.keras")

# === W&B Wrap-up ===
wandb.finish()