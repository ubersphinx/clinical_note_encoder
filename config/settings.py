"""
Configuration settings for Toxicology ICD Prediction MVP
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "trained_models"
BERT_MODEL_DIR = MODELS_DIR / "bert_toxicology"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
BERT_MODEL_DIR.mkdir(exist_ok=True)

# Model settings
BERT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WARMUP_STEPS = 100

# Toxicology-specific settings
TOXICOLOGY_ICD_CATEGORIES = {
    "T36-T50": "Poisoning by drugs and medications",
    "T51-T65": "Toxic effects of substances",
    "R40": "Somnolence, stupor and coma",
    "R56": "Convulsions",
    "Z91.5": "Personal history of self-harm"
}

# Substance categories for classification
SUBSTANCE_CATEGORIES = {
    "sedatives": ["alprazolam", "zolpidem", "diazepam", "lorazepam", "clonazepam"],
    "analgesics": ["acetaminophen", "aspirin", "ibuprofen", "naproxen", "oxycodone"],
    "stimulants": ["cocaine", "methamphetamine", "amphetamine", "methylphenidate"],
    "hallucinogens": ["LSD", "psilocybin", "mescaline", "DMT"],
    "opioids": ["heroin", "fentanyl", "morphine", "codeine", "hydrocodone"],
    "antidepressants": ["sertraline", "fluoxetine", "venlafaxine", "bupropion"],
    "antipsychotics": ["risperidone", "olanzapine", "quetiapine", "haloperidol"]
}

# Clinical entities to extract
CLINICAL_ENTITIES = [
    "SUBSTANCE", "SYMPTOM", "VITAL_SIGN", "TREATMENT", "ROUTE", 
    "DOSE", "TIMELINE", "LAB_VALUE", "PROCEDURE"
]

# Vital signs patterns
VITAL_SIGNS_PATTERNS = {
    "GCS": r"GCS\s*[:=]\s*(\d+)",
    "BP": r"BP\s*[:=]\s*(\d+/\d+)",
    "HR": r"HR\s*[:=]\s*(\d+)",
    "RR": r"RR\s*[:=]\s*(\d+)",
    "TEMP": r"TEMP\s*[:=]\s*(\d+\.?\d*)",
    "O2_SAT": r"O2\s*[:=]\s*(\d+)"
}

# UI Settings
STREAMLIT_CONFIG = {
    "page_title": "Toxicology ICD Predictor",
    "page_icon": "🏥",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Color scheme for medical/toxicology theme
COLORS = {
    "primary": "#2E86AB",      # Medical blue
    "secondary": "#A23B72",    # Medical purple
    "accent": "#F18F01",       # Warning orange
    "success": "#C73E1D",      # Medical red
    "background": "#F8F9FA",   # Light gray
    "text": "#2C3E50"          # Dark gray
}

# File paths
DATA_FILES = {
    "clean_notes": DATA_DIR / "toxicology_notes_dataset.csv",
    "messy_notes": DATA_DIR / "toxicology_messy_notes_dataset.csv",
    "abbreviations": DATA_DIR / "tox_abbreviation_dictionary.csv",
    "icd_codes": DATA_DIR / "toxicology_icd_codes.json"
}

# Model file paths
MODEL_FILES = {
    "bert_model": BERT_MODEL_DIR,
    "icd_classifier": MODELS_DIR / "icd_classifier.pkl",
    "text_processor": MODELS_DIR / "text_processor.pkl"
}

# Cache settings
CACHE_TTL = 3600  # 1 hour
MAX_CACHE_SIZE = 100

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s" 