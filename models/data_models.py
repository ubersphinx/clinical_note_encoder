"""Pydantic data models for Toxicology ICD Prediction MVP"""
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
import json

class ToxicologyEntity(BaseModel):
    """Represents a clinical entity extracted from toxicology notes"""
    text: str = Field(..., description="The extracted text")
    label: str = Field(..., description="Entity type (SUBSTANCE, SYMPTOM, etc.)")
    start: int = Field(..., description="Start position in original text")
    end: int = Field(..., description="End position in original text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class VitalSigns(BaseModel):
    """Vital signs extracted from clinical notes"""
    gcs: Optional[str] = Field(None, description="Glasgow Coma Scale")
    blood_pressure: Optional[str] = Field(None, description="Blood pressure")
    heart_rate: Optional[str] = Field(None, description="Heart rate")
    respiratory_rate: Optional[str] = Field(None, description="Respiratory rate")
    temperature: Optional[str] = Field(None, description="Temperature")
    oxygen_saturation: Optional[str] = Field(None, description="Oxygen saturation")

class ToxicologyNote(BaseModel):
    """Represents a processed toxicology clinical note"""
    text: str = Field(..., description="Original note text")
    expanded_text: str = Field(..., description="Text after abbreviation expansion")
    chief_complaint: Optional[str] = Field(None, description="Chief complaint")
    substance: Optional[str] = Field(None, description="Primary substance involved")
    route: Optional[str] = Field(None, description="Route of exposure")
    dose: Optional[str] = Field(None, description="Dose/quantity information")
    timeline: Optional[str] = Field(None, description="Timeline of events")
    symptoms: List[str] = Field(default_factory=list, description="List of symptoms")
    vitals: VitalSigns = Field(default_factory=VitalSigns, description="Vital signs")
    entities: List[ToxicologyEntity] = Field(default_factory=list, description="Extracted entities")
    severity_score: Optional[float] = Field(None, ge=0.0, le=10.0, description="Clinical severity score")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ICDPrediction(BaseModel):
    """Represents an ICD code prediction with metadata"""
    code: str = Field(..., description="ICD-10 code")
    description: str = Field(..., description="ICD code description")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    category: str = Field(..., description="ICD category (T-code, R-code, Z-code)")
    reasoning: str = Field(..., description="Reasoning for prediction")
    severity: Optional[str] = Field(None, description="Severity level if applicable")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class PredictionResult(BaseModel):
    """Complete prediction result for a toxicology note"""
    note: ToxicologyNote = Field(..., description="Processed note")
    predictions: List[ICDPrediction] = Field(..., description="ICD predictions")
    top_prediction: Optional[ICDPrediction] = Field(None, description="Highest confidence prediction")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_version: str = Field(..., description="Model version used")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class TrainingData(BaseModel):
    """Training data structure"""
    note: str = Field(..., description="Original note text")
    icd_codes: List[str] = Field(..., description="Associated ICD codes")
    processed_note: Optional[ToxicologyNote] = Field(None, description="Processed note")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ModelMetrics(BaseModel):
    """Model performance metrics"""
    accuracy: float = Field(..., ge=0.0, le=1.0)
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1_score: float = Field(..., ge=0.0, le=1.0)
    auc_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    confusion_matrix: Optional[List[List[int]]] = Field(None)
    per_class_metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class TrainingConfig(BaseModel):
    """Training configuration parameters"""
    model_name: str = Field(default="emilyalsentzer/Bio_ClinicalBERT")
    max_length: int = Field(default=512)
    batch_size: int = Field(default=8)
    learning_rate: float = Field(default=2e-5)
    num_epochs: int = Field(default=3)
    warmup_steps: int = Field(default=100)
    train_split: float = Field(default=0.8, ge=0.1, le=0.9)
    validation_split: float = Field(default=0.1, ge=0.05, le=0.3)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class BatchPredictionResult(BaseModel):
    """Results from batch prediction"""
    results: List[PredictionResult] = Field(..., description="Individual prediction results")
    summary: Dict[str, Any] = Field(..., description="Summary statistics")
    processing_time: float = Field(..., description="Total processing time")
    success_count: int = Field(..., description="Number of successful predictions")
    error_count: int = Field(..., description="Number of failed predictions")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class AbbreviationMapping(BaseModel):
    """Medical abbreviation mapping"""
    abbreviation: str = Field(..., description="Medical abbreviation")
    expansion: str = Field(..., description="Full expansion")
    context: Optional[str] = Field(None, description="Context where this applies")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# Utility functions for data models
def note_to_dict(note: ToxicologyNote) -> Dict[str, Any]:
    """Convert ToxicologyNote to dictionary"""
    return note.dict()

def prediction_to_dict(prediction: ICDPrediction) -> Dict[str, Any]:
    """Convert ICDPrediction to dictionary"""
    return prediction.dict()

def result_to_json(result: PredictionResult) -> str:
    """Convert PredictionResult to JSON string"""
    return result.json()

def batch_result_to_csv(batch_result: BatchPredictionResult) -> str:
    """Convert BatchPredictionResult to CSV string"""
    import pandas as pd
    
    rows = []
    for result in batch_result.results:
        row = {
            "original_text": result.note.text,
            "expanded_text": result.note.expanded_text,
            "substance": result.note.substance,
            "route": result.note.route,
            "top_icd_code": result.top_prediction.code if result.top_prediction else "",
            "top_confidence": result.top_prediction.confidence if result.top_prediction else 0.0,
            "all_predictions": "; ".join([f"{p.code}({p.confidence:.3f})" for p in result.predictions]),
            "processing_time": result.processing_time
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df.to_csv(index=False) 