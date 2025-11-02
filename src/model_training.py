import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from src.config import Config
import os

class ResumeClassifier(nn.Module):
    """Neural network classifier for resume categorization"""
    def __init__(self, input_dim, num_classes):
        super(ResumeClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

class ModelTrainer:
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
        self.label_encoder = LabelEncoder()
        self.model = None
        
    def prepare_data(self):
        """Prepare training data"""
        print("Preparing training data...")
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(self.labels)
        num_classes = len(self.label_encoder.classes_)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.embeddings, encoded_labels,
            test_size=0.2, random_state=42, stratify=encoded_labels
        )
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(Config.DEVICE)
        X_test = torch.FloatTensor(X_test).to(Config.DEVICE)
        y_train = torch.LongTensor(y_train).to(Config.DEVICE)
        y_test = torch.LongTensor(y_test).to(Config.DEVICE)
        
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"Number of classes: {num_classes}")
        
        return X_train, X_test, y_train, y_test, num_classes
    
    def train_model(self, X_train, y_train, X_test, y_test, num_classes):
        """Train classification model on GPU"""
        print(f"\nTraining model on {Config.DEVICE}...")
        
        # Initialize model
        input_dim = X_train.shape[1]
        self.model = ResumeClassifier(input_dim, num_classes).to(Config.DEVICE)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        
        # Training loop
        for epoch in range(Config.EPOCHS):
            self.model.train()
            
            # Forward pass
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Evaluation
            self.model.eval()
            with torch.no_grad():
                train_outputs = self.model(X_train)
                _, train_preds = torch.max(train_outputs, 1)
                train_acc = (train_preds == y_train).float().mean()
                
                test_outputs = self.model(X_test)
                _, test_preds = torch.max(test_outputs, 1)
                test_acc = (test_preds == y_test).float().mean()
            
            print(f"Epoch [{epoch+1}/{Config.EPOCHS}], Loss: {loss.item():.4f}, "
                  f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        # Final evaluation
        print("\nFinal Evaluation:")
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(X_test)
            _, test_preds = torch.max(test_outputs, 1)
            
        y_test_cpu = y_test.cpu().numpy()
        test_preds_cpu = test_preds.cpu().numpy()
        
        print(f"\nTest Accuracy: {accuracy_score(y_test_cpu, test_preds_cpu):.4f}")
        print("\nClassification Report:")
        print(classification_report(
            y_test_cpu, test_preds_cpu,
            target_names=self.label_encoder.classes_
        ))
    
    def save_model(self):
        """Save trained model"""
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        
        model_path = os.path.join(Config.MODEL_DIR, 'classifier_model.pth')
        encoder_path = os.path.join(Config.MODEL_DIR, 'label_encoder.npy')
        
        torch.save(self.model.state_dict(), model_path)
        np.save(encoder_path, self.label_encoder.classes_)
        
        print(f"\nModel saved to {model_path}")
        print(f"Label encoder saved to {encoder_path}")
