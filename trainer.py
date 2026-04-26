import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,random_split
import os

from datapreprocess import ExerciseDataset
from model import WorkoutClassifier

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

LEARNING_RATE=0.001
BATCH_SIZE=32
EPOCHS=100

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100):
    train_losses = []
    val_losses = []
    
    # Check where the model is (CPU or GPU) so we can send data to the same place
    device = next(model.parameters()).device 
    
    print(f"🚀 Starting Training on {device}...")
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # =========================
        # 1. TRAINING PHASE
        # =========================
        model.train()
        running_train_loss = 0.0
        
        for inputs, labels in train_loader:
            # ---> NEW LINE HERE: Send data to GPU <---
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * inputs.size(0)
            
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # =========================
        # 2. VALIDATION PHASE
        # =========================
        model.eval()
        running_val_loss = 0.0
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                # ---> NEW LINE HERE: Send data to GPU <---
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(labels.cpu().numpy())
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            # Save the "best" weights to a temporary file
            torch.save(model.state_dict(), 'best_model_weights.pth')
            print(f"🌟 Epoch {epoch+1}: New best validation loss! Saving weights.")
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

    # =========================
    # 3. METRICS & GRAPH
    # =========================
    print("\n✅ Training Complete! Calculating Final Metrics...")
    
    acc = accuracy_score(all_true, all_preds)
    prec = precision_score(all_true, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_true, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_true, all_preds, average='macro', zero_division=0)

    print("-" * 30)
    print("📊 FINAL RESULTS FOR PPT TABLE:")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("-" * 30)

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='orange', linewidth=2)
    
    plt.title('Model Loss Over Epochs', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig('loss_graph.png', dpi=300, bbox_inches='tight')
    print("📸 Graph saved as 'loss_graph.png' in your project folder!")

    model.load_state_dict(torch.load('best_model_weights.pth'))
    print("✅ Loaded the best weights (prevented overfitting).")

    return model

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # 1. Load the full dataset
    data_path = os.path.join("..", "data", "processed")
    full_dataset = ExerciseDataset(data_path)
    print(f"Total overall samples = {len(full_dataset)}")

    # 2. Calculate the split sizes (80% Train, 20% Validation)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # 3. Randomly split the dataset into two smaller datasets
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

    # 4. Create the TWO DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) # Validation doesn't need shuffling

    # 5. Setup the Model, Loss, and Optimizer
    input_size = 99
    num_classes = len(full_dataset.label_map)
    model = WorkoutClassifier(input_size, num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ==========================================
    # 6. LAUNCH THE NEW TRAINING LOOP
    # ==========================================
    print("Initiating training with validation...")
    
    # Call the graphing/metrics function we created!
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=loss_fn,
        optimizer=optimizer,
        num_epochs=EPOCHS
    )

    # 7. Save the model
    save_path = os.path.join("..", "models", "exercise_classifier.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(trained_model.state_dict(), save_path)
    print(f"Model saved successfully to {save_path}")
if __name__=="__main__":
    train()