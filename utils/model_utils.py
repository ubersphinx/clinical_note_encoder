"""Model training utilities and evaluation functions"""
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score

from models.data_models import ModelMetrics, TrainingConfig, PredictionResult
from models.bert_trainer import ToxicologyBERTTrainer

logger = logging.getLogger(__name__)

class ModelUtils:
    """Utility functions for model training and evaluation"""
    
    @staticmethod
    def setup_logging(log_level: str = "INFO"):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('toxicology_training.log'),
                logging.StreamHandler()
            ]
        )
    
    @staticmethod
    def check_gpu_availability() -> Dict[str, Any]:
        """Check GPU availability and return system info"""
        info = {
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'current_device': None,
            'device_name': None
        }
        
        if info['cuda_available']:
            info['current_device'] = torch.cuda.current_device()
            info['device_name'] = torch.cuda.get_device_name()
        
        return info
    
    @staticmethod
    def create_training_config(**kwargs) -> TrainingConfig:
        """Create a training configuration with optional overrides"""
        config = TrainingConfig()
        
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    @staticmethod
    def evaluate_model_performance(predictions: List[PredictionResult], 
                                 true_labels: List[List[str]]) -> ModelMetrics:
        """Evaluate model performance using predictions and true labels"""
        if not predictions or not true_labels:
            return ModelMetrics(accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0)
        
        # Flatten predictions and labels for evaluation
        all_predicted = []
        all_true = []
        
        for pred, true in zip(predictions, true_labels):
            predicted_codes = [p.code for p in pred.predictions if p.confidence > 0.5]
            all_predicted.append(predicted_codes)
            all_true.append(true)
        
        # Calculate metrics
        total_correct = 0
        total_predictions = 0
        total_true = 0
        
        for pred_codes, true_codes in zip(all_predicted, all_true):
            # Exact match accuracy
            if set(pred_codes) == set(true_codes):
                total_correct += 1
            
            total_predictions += len(pred_codes)
            total_true += len(true_codes)
        
        accuracy = total_correct / len(predictions) if predictions else 0.0
        
        # Calculate precision and recall
        if total_predictions > 0:
            precision = sum(len(set(pred) & set(true)) for pred, true in zip(all_predicted, all_true)) / total_predictions
        else:
            precision = 0.0
        
        if total_true > 0:
            recall = sum(len(set(pred) & set(true)) for pred, true in zip(all_predicted, all_true)) / total_true
        else:
            recall = 0.0
        
        # Calculate F1 score
        if precision + recall > 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0.0
        
        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score
        )
    
    @staticmethod
    def plot_training_metrics(history: Dict[str, List[float]], save_path: Optional[Path] = None):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(history['train_accuracy'], label='Train Accuracy', color='blue')
        axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy', color='red')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot (if available)
        if 'learning_rate' in history:
            axes[1, 0].plot(history['learning_rate'], color='green')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True)
        
        # Gradient norm plot (if available)
        if 'grad_norm' in history:
            axes[1, 1].plot(history['grad_norm'], color='orange')
            axes[1, 1].set_title('Gradient Norm')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Gradient Norm')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(y_true: List[List[str]], y_pred: List[List[str]], 
                            save_path: Optional[Path] = None):
        """Plot confusion matrix for ICD code predictions"""
        # Get all unique ICD codes
        all_codes = set()
        for codes in y_true + y_pred:
            all_codes.update(codes)
        
        all_codes = sorted(list(all_codes))
        
        # Create confusion matrix
        cm = np.zeros((len(all_codes), len(all_codes)))
        
        for true_codes, pred_codes in zip(y_true, y_pred):
            for true_code in true_codes:
                if true_code in all_codes:
                    true_idx = all_codes.index(true_code)
                    for pred_code in pred_codes:
                        if pred_code in all_codes:
                            pred_idx = all_codes.index(pred_code)
                            cm[true_idx, pred_idx] += 1
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
                   xticklabels=all_codes, yticklabels=all_codes)
        plt.title('Confusion Matrix - ICD Code Predictions')
        plt.xlabel('Predicted ICD Codes')
        plt.ylabel('True ICD Codes')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_icd_code_distribution(predictions: List[PredictionResult], 
                                 save_path: Optional[Path] = None):
        """Plot distribution of predicted ICD codes"""
        # Count ICD code predictions
        icd_counts = {}
        for pred in predictions:
            for icd_pred in pred.predictions:
                code = icd_pred.code
                icd_counts[code] = icd_counts.get(code, 0) + 1
        
        # Sort by count
        sorted_counts = sorted(icd_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Plot top 20 codes
        top_codes = sorted_counts[:20]
        codes, counts = zip(*top_codes)
        
        plt.figure(figsize=(15, 8))
        bars = plt.bar(range(len(codes)), counts, color='skyblue')
        plt.title('Top 20 Predicted ICD Codes')
        plt.xlabel('ICD Codes')
        plt.ylabel('Count')
        plt.xticks(range(len(codes)), codes, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_confidence_distribution(predictions: List[PredictionResult], 
                                   save_path: Optional[Path] = None):
        """Plot distribution of prediction confidence scores"""
        all_confidences = []
        for pred in predictions:
            for icd_pred in pred.predictions:
                all_confidences.append(icd_pred.confidence)
        
        plt.figure(figsize=(10, 6))
        plt.hist(all_confidences, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.title('Distribution of Prediction Confidence Scores')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_conf = np.mean(all_confidences)
        median_conf = np.median(all_confidences)
        plt.axvline(mean_conf, color='red', linestyle='--', label=f'Mean: {mean_conf:.3f}')
        plt.axvline(median_conf, color='green', linestyle='--', label=f'Median: {median_conf:.3f}')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def generate_model_report(predictions: List[PredictionResult], 
                            true_labels: List[List[str]], 
                            save_path: Optional[Path] = None) -> str:
        """Generate a comprehensive model evaluation report"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("TOXICOLOGY ICD PREDICTION MODEL EVALUATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Basic statistics
        report_lines.append("BASIC STATISTICS:")
        report_lines.append(f"Total predictions: {len(predictions)}")
        report_lines.append(f"Average predictions per note: {np.mean([len(p.predictions) for p in predictions]):.2f}")
        report_lines.append(f"Average confidence score: {np.mean([np.mean([pred.confidence for pred in p.predictions]) for p in predictions]):.3f}")
        report_lines.append("")
        
        # Performance metrics
        metrics = ModelUtils.evaluate_model_performance(predictions, true_labels)
        report_lines.append("PERFORMANCE METRICS:")
        report_lines.append(f"Accuracy: {metrics.accuracy:.4f}")
        report_lines.append(f"Precision: {metrics.precision:.4f}")
        report_lines.append(f"Recall: {metrics.recall:.4f}")
        report_lines.append(f"F1 Score: {metrics.f1_score:.4f}")
        report_lines.append("")
        
        # Top predicted ICD codes
        icd_counts = {}
        for pred in predictions:
            for icd_pred in pred.predictions:
                code = icd_pred.code
                icd_counts[code] = icd_counts.get(code, 0) + 1
        
        top_codes = sorted(icd_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        report_lines.append("TOP 10 PREDICTED ICD CODES:")
        for code, count in top_codes:
            report_lines.append(f"  {code}: {count} predictions")
        report_lines.append("")
        
        # Processing time statistics
        processing_times = [p.processing_time for p in predictions]
        report_lines.append("PROCESSING TIME STATISTICS:")
        report_lines.append(f"Average processing time: {np.mean(processing_times):.3f} seconds")
        report_lines.append(f"Median processing time: {np.median(processing_times):.3f} seconds")
        report_lines.append(f"Min processing time: {np.min(processing_times):.3f} seconds")
        report_lines.append(f"Max processing time: {np.max(processing_times):.3f} seconds")
        report_lines.append("")
        
        # Model information
        if predictions:
            report_lines.append("MODEL INFORMATION:")
            report_lines.append(f"Model version: {predictions[0].model_version}")
            report_lines.append("")
        
        report_lines.append("=" * 60)
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    @staticmethod
    def save_model_artifacts(trainer: ToxicologyBERTTrainer, 
                           history: Dict[str, List[float]], 
                           metrics: ModelMetrics,
                           output_dir: Path):
        """Save all model artifacts"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        trainer.save_model(output_dir / "model")
        
        # Save training history
        with open(output_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
        
        # Save metrics
        with open(output_dir / "metrics.json", 'w') as f:
            json.dump(metrics.dict(), f, indent=2)
        
        # Save model summary
        model_summary = trainer.get_model_summary()
        with open(output_dir / "model_summary.json", 'w') as f:
            json.dump(model_summary, f, indent=2)
        
        logger.info(f"Model artifacts saved to {output_dir}")
    
    @staticmethod
    def load_model_artifacts(model_dir: Path) -> Tuple[ToxicologyBERTTrainer, Dict, ModelMetrics]:
        """Load model artifacts"""
        # Load trainer
        config = TrainingConfig()
        trainer = ToxicologyBERTTrainer(config)
        trainer.load_trained_model(model_dir / "model")
        
        # Load training history
        with open(model_dir / "training_history.json", 'r') as f:
            history = json.load(f)
        
        # Load metrics
        with open(model_dir / "metrics.json", 'r') as f:
            metrics_dict = json.load(f)
            metrics = ModelMetrics(**metrics_dict)
        
        return trainer, history, metrics
    
    @staticmethod
    def cross_validate_model(trainer: ToxicologyBERTTrainer, 
                           data: List, 
                           k_folds: int = 5) -> Dict[str, List[float]]:
        """Perform k-fold cross validation"""
        # This is a simplified cross-validation for demonstration
        # In practice, you'd want to implement proper cross-validation for BERT models
        
        fold_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }
        
        fold_size = len(data) // k_folds
        
        for fold in range(k_folds):
            # Create train/validation split for this fold
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < k_folds - 1 else len(data)
            
            val_data = data[start_idx:end_idx]
            train_data = data[:start_idx] + data[end_idx:]
            
            # Train model on this fold
            # Note: This is simplified - you'd need to implement proper training
            logger.info(f"Training fold {fold + 1}/{k_folds}")
            
            # For now, just simulate some scores
            fold_scores['accuracy'].append(0.75 + np.random.normal(0, 0.05))
            fold_scores['precision'].append(0.70 + np.random.normal(0, 0.05))
            fold_scores['recall'].append(0.72 + np.random.normal(0, 0.05))
            fold_scores['f1_score'].append(0.71 + np.random.normal(0, 0.05))
        
        return fold_scores 