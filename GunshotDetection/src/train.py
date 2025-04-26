import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# === SETTINGS ===
FEATURES_DIR = '../features/'
MODEL_DIR = '../models/'
MODEL_NAME = 'gunshot_classifier.h5'

# === Ensure output directory exists ===
os.makedirs(MODEL_DIR, exist_ok=True)

# === Load data ===
print("Loading features...")
X = np.load(os.path.join(FEATURES_DIR, 'X.npy'))
y = np.load(os.path.join(FEATURES_DIR, 'y.npy'))

# Expand dims to add channel (CNN expects 4D input)
X = np.expand_dims(X, -1)  # Now shape is (samples, 64, time_steps, 1)

# Normalize inputs (important for training stability)
X = (X - np.min(X)) / (np.max(X) - np.min(X))

print(f"Features shape: {X.shape}, Labels shape: {y.shape}")

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# === Build CNN model ===
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=X_train.shape[1:]))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # Binary classification (urban vs gunshot)

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# === Train ===
print("Training model...")
checkpoint = ModelCheckpoint(os.path.join(MODEL_DIR, MODEL_NAME), monitor='val_accuracy', save_best_only=True, verbose=1)
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint]
)

# === Final evaluation ===
print("Evaluating best model on test set...")
model.load_weights(os.path.join(MODEL_DIR, MODEL_NAME))
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("Training complete! Model saved in /models/")