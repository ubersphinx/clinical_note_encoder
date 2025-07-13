"""BERT training pipeline for toxicology ICD prediction"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import logging
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from config.settings import (
    BERT_MODEL_NAME, MAX_SEQUENCE_LENGTH, BATCH_SIZE, 
    LEARNING_RATE, NUM_EPOCHS, WARMUP_STEPS
)
from models.data_models import TrainingData, ModelMetrics, TrainingConfig
from models.text_processor import ToxicologyTextProcessor

logger = logging.getLogger(__name__)

class ToxicologyDataset(Dataset):
    """Dataset for toxicology notes and ICD codes"""
    
    def __init__(self, texts: List[str], labels: List[List[int]], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

class ToxicologyBERTTrainer:
    """BERT trainer for toxicology ICD prediction"""
    
    def __init__(self, config: TrainingConfig):
        """Initialize the BERT trainer"""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.label_encoder = {}
        self.reverse_label_encoder = {}
        self.text_processor = ToxicologyTextProcessor()
        
        logger.info(f"Using device: {self.device}")
    
    def prepare_data(self, data: List[TrainingData]) -> Tuple[List[str], List[List[int]], Dict[str, int]]:
        """Prepare data for training"""
        texts = []
        all_icd_codes = set()
        
        # Collect all unique ICD codes
        for item in data:
            texts.append(item.note)
            all_icd_codes.update(item.icd_codes)
        
        # Create label encoder
        icd_codes_list = sorted(list(all_icd_codes))
        self.label_encoder = {code: idx for idx, code in enumerate(icd_codes_list)}
        self.reverse_label_encoder = {idx: code for code, idx in self.label_encoder.items()}
        
        # Convert ICD codes to multi-hot labels
        labels = []
        for item in data:
            label = [0] * len(self.label_encoder)
            for icd_code in item.icd_codes:
                if icd_code in self.label_encoder:
                    label[self.label_encoder[icd_code]] = 1
            labels.append(label)
        
        logger.info(f"Prepared {len(texts)} samples with {len(self.label_encoder)} ICD codes")
        return texts, labels, self.label_encoder
    
    def load_model(self):
        """Load BERT model and tokenizer"""
        logger.info(f"Loading model: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=len(self.label_encoder),
            problem_type="multi_label_classification"
        )
        
        # Add special tokens for medical domain if needed
        special_tokens = {
            'additional_special_tokens': [
                '[SUBSTANCE]', '[SYMPTOM]', '[TREATMENT]', '[VITAL]',
                '[DOSE]', '[ROUTE]', '[TIMELINE]'
            ]
        }
        
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.model.to(self.device)
        logger.info("Model loaded successfully")
    
    def create_data_loaders(self, texts: List[str], labels: List[List[int]]) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation data loaders"""
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=None
        )
        
        # Create datasets
        train_dataset = ToxicologyDataset(train_texts, train_labels, self.tokenizer, self.config.max_length)
        val_dataset = ToxicologyDataset(val_texts, val_labels, self.tokenizer, self.config.max_length)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"Created data loaders: {len(train_dataset)} train, {len(val_dataset)} validation")
        return train_loader, val_loader
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """Train the BERT model"""
        logger.info("Starting model training")
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.config.warmup_steps, num_training_steps=total_steps
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                
                # Calculate accuracy
                predictions = torch.sigmoid(outputs.logits)
                predicted_labels = (predictions > 0.5).float()
                train_correct += (predicted_labels == labels).all(dim=1).sum().item()
                train_total += labels.size(0)
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    
                    val_loss += loss.item()
                    
                    # Calculate accuracy
                    predictions = torch.sigmoid(outputs.logits)
                    predicted_labels = (predictions > 0.5).float()
                    val_correct += (predicted_labels == labels).all(dim=1).sum().item()
                    val_total += labels.size(0)
            
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = val_correct / val_total
            
            # Update history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_accuracy'].append(train_accuracy)
            history['val_accuracy'].append(val_accuracy)
            
            logger.info(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
            logger.info(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model()
                logger.info("Saved best model")
        
        logger.info("Training completed")
        return history
    
    def evaluate_model(self, val_loader: DataLoader) -> ModelMetrics:
        """Evaluate the trained model"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.sigmoid(outputs.logits)
                predicted_labels = (predictions > 0.5).float()
                
                all_predictions.extend(predicted_labels.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Overall accuracy
        accuracy = (all_predictions == all_labels).all(axis=1).mean()
        
        # Per-label metrics
        precision = []
        recall = []
        f1_scores = []
        
        for i in range(all_labels.shape[1]):
            if all_labels[:, i].sum() > 0:  # Only calculate if label exists
                p = (all_predictions[:, i] * all_labels[:, i]).sum() / (all_predictions[:, i].sum() + 1e-8)
                r = (all_predictions[:, i] * all_labels[:, i]).sum() / (all_labels[:, i].sum() + 1e-8)
                f1 = 2 * p * r / (p + r + 1e-8)
                
                precision.append(p)
                recall.append(r)
                f1_scores.append(f1)
        
        avg_precision = np.mean(precision)
        avg_recall = np.mean(recall)
        avg_f1 = np.mean(f1_scores)
        
        # Per-class metrics
        per_class_metrics = {}
        for i, icd_code in self.reverse_label_encoder.items():
            if all_labels[:, i].sum() > 0:
                per_class_metrics[icd_code] = {
                    'precision': precision[i] if i < len(precision) else 0.0,
                    'recall': recall[i] if i < len(recall) else 0.0,
                    'f1_score': f1_scores[i] if i < len(f1_scores) else 0.0
                }
        
        metrics = ModelMetrics(
            accuracy=float(accuracy),
            precision=float(avg_precision),
            recall=float(avg_recall),
            f1_score=float(avg_f1),
            per_class_metrics=per_class_metrics
        )
        
        logger.info(f"Model Evaluation - Accuracy: {accuracy:.4f}, F1: {avg_f1:.4f}")
        return metrics
    
    def save_model(self, model_path: Optional[Path] = None):
        """Save the trained model"""
        if model_path is None:
            from config.settings import MODEL_FILES
            model_path = MODEL_FILES["bert_model"]
        
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        # Save label encoder
        encoder_path = model_path / "label_encoder.json"
        with open(encoder_path, 'w') as f:
            json.dump(self.label_encoder, f)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_trained_model(self, model_path: Optional[Path] = None):
        """Load a trained model"""
        if model_path is None:
            from config.settings import MODEL_FILES
            model_path = MODEL_FILES["bert_model"]
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            problem_type="multi_label_classification"
        )
        
        # Load label encoder
        encoder_path = model_path / "label_encoder.json"
        with open(encoder_path, 'r') as f:
            self.label_encoder = json.load(f)
        
        self.reverse_label_encoder = {int(v): k for k, v in self.label_encoder.items()}
        self.model.to(self.device)
        
        logger.info(f"Loaded trained model from {model_path}")
    
    def predict(self, text: str, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """Predict ICD codes for a given text"""
        self.model.eval()
        
        # Process text
        processed_note = self.text_processor.process_note(text)
        
        # Tokenize
        encoding = self.tokenizer(
            processed_note.expanded_text,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.sigmoid(outputs.logits)
        
        # Get predictions above threshold
        results = []
        for i, prob in enumerate(predictions[0]):
            if prob > threshold:
                icd_code = self.reverse_label_encoder.get(i, f"UNKNOWN_{i}")
                results.append((icd_code, float(prob)))
        
        # Sort by confidence
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def plot_training_history(self, history: Dict[str, List[float]], save_path: Optional[Path] = None):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(history['train_accuracy'], label='Train Accuracy')
        ax2.plot(history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_name': self.config.model_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_labels': len(self.label_encoder),
            'device': str(self.device),
            'label_encoder': self.label_encoder
        } 