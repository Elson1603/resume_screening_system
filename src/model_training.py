import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import numpy as np
from src.config import Config
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
        self.training_history = {
            'epochs': [],
            'train_loss': [],
            'train_accuracy': [],
            'test_accuracy': []
        }
        
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
            
            # Store metrics
            self.training_history['epochs'].append(epoch + 1)
            self.training_history['train_loss'].append(loss.item())
            self.training_history['train_accuracy'].append(train_acc.item())
            self.training_history['test_accuracy'].append(test_acc.item())
            
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
        
        # Calculate detailed metrics
        final_accuracy = accuracy_score(y_test_cpu, test_preds_cpu)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test_cpu, test_preds_cpu, average='weighted'
        )
        
        print(f"\nTest Accuracy: {final_accuracy:.4f}")
        print(f"Weighted Precision: {precision:.4f}")
        print(f"Weighted Recall: {recall:.4f}")
        print(f"Weighted F1-Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(
            y_test_cpu, test_preds_cpu,
            target_names=self.label_encoder.classes_
        ))
        
        # Save metrics and generate visualizations
        self._save_metrics(
            y_test_cpu, test_preds_cpu, 
            final_accuracy, precision, recall, f1
        )
        self._generate_visualizations(y_test_cpu, test_preds_cpu)
    
    def save_model(self):
        """Save trained model"""
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        
        model_path = os.path.join(Config.MODEL_DIR, 'classifier_model.pth')
        encoder_path = os.path.join(Config.MODEL_DIR, 'label_encoder.npy')
        
        torch.save(self.model.state_dict(), model_path)
        np.save(encoder_path, self.label_encoder.classes_)
        
        print(f"\nModel saved to {model_path}")
        print(f"Label encoder saved to {encoder_path}")
    
    def _save_metrics(self, y_test, y_pred, accuracy, precision, recall, f1):
        """Save training metrics to JSON file"""
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        
        # Get confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Get per-class metrics
        class_report = classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        # Compile all metrics
        metrics = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_config': {
                'model_name': Config.MODEL_NAME,
                'device': str(Config.DEVICE),
                'batch_size': Config.BATCH_SIZE,
                'learning_rate': Config.LEARNING_RATE,
                'epochs': Config.EPOCHS,
                'embedding_dim': self.embeddings.shape[1],
                'num_classes': len(self.label_encoder.classes_),
                'total_samples': len(self.labels),
                'train_samples': int(len(self.labels) * 0.8),
                'test_samples': int(len(self.labels) * 0.2)
            },
            'training_history': self.training_history,
            'final_metrics': {
                'accuracy': float(accuracy),
                'weighted_precision': float(precision),
                'weighted_recall': float(recall),
                'weighted_f1_score': float(f1)
            },
            'per_class_metrics': class_report,
            'confusion_matrix': cm.tolist(),
            'class_names': self.label_encoder.classes_.tolist(),
            'scoring_weights': {
                'semantic': Config.SEMANTIC_WEIGHT,
                'keyword': Config.KEYWORD_WEIGHT,
                'experience': Config.EXPERIENCE_WEIGHT,
                'education': Config.EDUCATION_WEIGHT
            },
            'shortlist_threshold': Config.SHORTLIST_THRESHOLD
        }
        
        # Save to JSON
        metrics_path = os.path.join(Config.MODEL_DIR, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"\n✅ Training metrics saved to {metrics_path}")
        
        # Also save a readable text report
        report_path = os.path.join(Config.MODEL_DIR, 'training_report.txt')
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("RESUME SCREENING SYSTEM - TRAINING REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Training Date: {metrics['timestamp']}\n")
            f.write(f"Device: {Config.DEVICE}\n")
            f.write(f"Model: {Config.MODEL_NAME}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("DATASET INFORMATION\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Samples: {metrics['model_config']['total_samples']}\n")
            f.write(f"Training Samples: {metrics['model_config']['train_samples']}\n")
            f.write(f"Test Samples: {metrics['model_config']['test_samples']}\n")
            f.write(f"Number of Classes: {metrics['model_config']['num_classes']}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("TRAINING CONFIGURATION\n")
            f.write("-"*80 + "\n")
            f.write(f"Epochs: {Config.EPOCHS}\n")
            f.write(f"Batch Size: {Config.BATCH_SIZE}\n")
            f.write(f"Learning Rate: {Config.LEARNING_RATE}\n")
            f.write(f"Embedding Dimension: {metrics['model_config']['embedding_dim']}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("FINAL PERFORMANCE METRICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
            f.write(f"Weighted Precision: {precision:.4f}\n")
            f.write(f"Weighted Recall: {recall:.4f}\n")
            f.write(f"Weighted F1-Score: {f1:.4f}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("PER-CLASS PERFORMANCE\n")
            f.write("-"*80 + "\n")
            for class_name in self.label_encoder.classes_:
                if class_name in class_report:
                    cls_metrics = class_report[class_name]
                    f.write(f"\n{class_name}:\n")
                    f.write(f"  Precision: {cls_metrics['precision']:.4f}\n")
                    f.write(f"  Recall: {cls_metrics['recall']:.4f}\n")
                    f.write(f"  F1-Score: {cls_metrics['f1-score']:.4f}\n")
                    f.write(f"  Support: {cls_metrics['support']}\n")
            
            f.write("\n" + "-"*80 + "\n")
            f.write("RANKING CONFIGURATION\n")
            f.write("-"*80 + "\n")
            f.write(f"Semantic Weight: {Config.SEMANTIC_WEIGHT * 100}%\n")
            f.write(f"Keyword Weight: {Config.KEYWORD_WEIGHT * 100}%\n")
            f.write(f"Experience Weight: {Config.EXPERIENCE_WEIGHT * 100}%\n")
            f.write(f"Education Weight: {Config.EDUCATION_WEIGHT * 100}%\n")
            f.write(f"Shortlist Threshold: {Config.SHORTLIST_THRESHOLD * 100}%\n\n")
            
            f.write("="*80 + "\n")
        
        print(f"✅ Training report saved to {report_path}")
    
    def _generate_visualizations(self, y_test, y_pred):
        """Generate training visualization plots"""
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        plots_dir = os.path.join(Config.MODEL_DIR, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Training History - Accuracy and Loss Curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = self.training_history['epochs']
        
        # Loss curve
        ax1.plot(epochs, self.training_history['train_loss'], 'b-o', label='Training Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(epochs, self.training_history['train_accuracy'], 'b-o', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, self.training_history['test_accuracy'], 'r-s', label='Test Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training & Test Accuracy Curves', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Training curves saved to {plots_dir}/training_curves.png")
        
        # 2. Confusion Matrix Heatmap
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(12, 10))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_,
                    cbar_kws={'label': 'Percentage'})
        plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Category', fontsize=12, fontweight='bold')
        plt.ylabel('True Category', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Confusion matrix saved to {plots_dir}/confusion_matrix.png")
        
        # 3. Per-Class Performance Bar Chart
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None, labels=np.arange(len(self.label_encoder.classes_))
        )
        
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(self.label_encoder.classes_))
        width = 0.25
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
        bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71')
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')
        
        ax.set_xlabel('Resume Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(self.label_encoder.classes_, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'per_class_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Per-class metrics chart saved to {plots_dir}/per_class_metrics.png")
        
        # 4. Class Distribution
        unique, counts = np.unique(y_test, return_counts=True)
        class_names = [self.label_encoder.classes_[i] for i in unique]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
        ax1.bar(class_names, counts, color=colors)
        ax1.set_xlabel('Resume Category', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
        ax1.set_title('Test Set Class Distribution', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        for i, (name, count) in enumerate(zip(class_names, counts)):
            ax1.text(i, count + 1, str(count), ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        ax2.pie(counts, labels=class_names, autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Test Set Distribution (%)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Class distribution saved to {plots_dir}/class_distribution.png")
        
        # 5. Model Performance Summary Dashboard
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Overall metrics
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        final_acc = accuracy_score(y_test, y_pred)
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        summary_text = f"""MODEL PERFORMANCE SUMMARY
        
Test Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)
Weighted Precision: {weighted_precision:.4f}
Weighted Recall: {weighted_recall:.4f}
Weighted F1-Score: {weighted_f1:.4f}

Total Test Samples: {len(y_test)}
Number of Classes: {len(self.label_encoder.classes_)}
Device: {Config.DEVICE}
Model: {Config.MODEL_NAME}
        """
        ax1.text(0.5, 0.5, summary_text, transform=ax1.transAxes,
                fontsize=12, verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                family='monospace', fontweight='bold')
        
        # Mini confusion matrix
        ax2 = fig.add_subplot(gs[1:, :2])
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_,
                    ax=ax2, cbar_kws={'label': 'Percentage'})
        ax2.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Predicted', fontsize=10)
        ax2.set_ylabel('True', fontsize=10)
        ax2.tick_params(axis='x', rotation=45)
        
        # Training curves mini
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.plot(epochs, self.training_history['train_loss'], 'b-o', linewidth=2)
        ax3.set_title('Training Loss', fontsize=10, fontweight='bold')
        ax3.set_xlabel('Epoch', fontsize=9)
        ax3.set_ylabel('Loss', fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(gs[2, 2])
        ax4.plot(epochs, self.training_history['test_accuracy'], 'r-s', linewidth=2)
        ax4.set_title('Test Accuracy', fontsize=10, fontweight='bold')
        ax4.set_xlabel('Epoch', fontsize=9)
        ax4.set_ylabel('Accuracy', fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Resume Screening Model - Performance Dashboard', 
                    fontsize=18, fontweight='bold', y=0.995)
        plt.savefig(os.path.join(plots_dir, 'performance_dashboard.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Performance dashboard saved to {plots_dir}/performance_dashboard.png")
        
        print(f"\n🎉 All visualizations generated successfully in {plots_dir}/")
