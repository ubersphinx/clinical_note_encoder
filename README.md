# Toxicology ICD Code Prediction System

A comprehensive Streamlit application for predicting ICD-10 codes from toxicology clinical notes. This system processes both clean and abbreviated medical notes to predict appropriate toxicology-related ICD codes.

## 🏥 Features

### Core Functionality
- **Multi-tab Interface**: Data upload, training, single note prediction, batch processing, and model performance
- **BERT-based Prediction**: Fine-tuned clinical BERT model for accurate ICD code prediction
- **Abbreviation Expansion**: Automatic expansion of medical abbreviations in clinical notes
- **Entity Extraction**: Toxicology-specific NER for substances, symptoms, vital signs, and treatments
- **Multi-label Classification**: Predicts multiple ICD codes per note with confidence scores

### Toxicology-Specific Features
- **Substance Detection**: Identifies drugs, medications, and toxic substances
- **Route Classification**: Determines exposure routes (oral, IV, inhalation, etc.)
- **Severity Assessment**: Calculates clinical severity scores based on symptoms and vitals
- **Treatment Recognition**: Identifies antidotes and treatments mentioned in notes

## 📁 Project Structure

```
toxicology_icd_mvp/
├── app.py                           # Main Streamlit application
├── requirements.txt                 # Dependencies
├── models/
│   ├── __init__.py
│   ├── text_processor.py            # Toxicology text preprocessing and NER
│   ├── abbreviation_expander.py     # Expand medical abbreviations
│   ├── bert_trainer.py              # BERT model training pipeline
│   ├── icd_predictor.py             # Toxicology ICD code prediction
│   └── data_models.py               # Pydantic models for data structures
├── data/
│   ├── sample_toxicology_data.csv   # Sample training data
│   └── __init__.py
├── utils/
│   ├── __init__.py
│   ├── csv_handler.py               # CSV data ingestion and processing
│   ├── model_utils.py               # Model training utilities
│   └── helpers.py                   # Utility functions
├── config/
│   ├── __init__.py
│   └── settings.py                  # Configuration settings
└── trained_models/                  # Saved model artifacts
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd toxicology_icd_mvp

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (optional, for enhanced NER)
python -m spacy download en_core_web_sm
```

### 2. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📊 Usage Guide

### Data Upload & Training Tab

1. **Upload Training Data**
   - Upload a CSV file with `Note` and `ICD_Codes` columns
   - The system will validate the file format and display statistics
   - Use the "Generate Sample Training Data" button to create sample data

2. **Train Model**
   - Configure training parameters (BERT model, batch size, learning rate, etc.)
   - Click "Start Training" to begin model training
   - Monitor training progress and metrics

3. **Training Metrics**
   - View training history plots (loss, accuracy)
   - Examine per-class performance metrics
   - Download training artifacts

### Single Note Prediction Tab

1. **Enter Clinical Note**
   - Type or paste a toxicology clinical note
   - Use example notes for testing
   - The system will process the note and expand abbreviations

2. **View Predictions**
   - See top ICD code predictions with confidence scores
   - Review reasoning for each prediction
   - Examine extracted entities (substances, symptoms, treatments)

3. **Note Analysis**
   - Compare original vs. expanded text
   - View extracted clinical entities
   - See processing time and model information

### Batch Processing Tab

1. **Upload CSV File**
   - Upload a CSV file with a `Note` column
   - The system will process all notes in the file

2. **View Results**
   - See summary statistics (total notes, success rate, processing time)
   - Review results table with predictions
   - Download results as CSV file

### Model Performance Tab

1. **Model Information**
   - View model status and configuration
   - See system information and device details

2. **Performance Metrics**
   - Review training metrics (accuracy, precision, recall, F1)
   - Examine per-class performance

3. **Test Predictions**
   - Run test predictions on sample notes
   - Evaluate model performance

## 📋 Data Format

### Training Data CSV Format

```csv
Note,ICD_Codes
"CC: Altered mental status. HPI: 27-year-old male found unresponsive at home. Empty bottle of alprazolam found. GCS 6. BP 90/60, HR 45, RR 8. Given naloxone with minimal response.","T42.4X1A, R40.20"
"pt found unresponsive @ home, bottle of zolpidem empty. GCS 8. BP 110/70, HR 52, RR 10. Flumazenil given with improvement.","T42.6X1A, R40.20"
```

### Batch Processing CSV Format

```csv
Note
"CC: Altered mental status. HPI: 27-year-old male found unresponsive at home. Empty bottle of alprazolam found. GCS 6. BP 90/60, HR 45, RR 8. Given naloxone with minimal response."
"pt found unresponsive @ home, bottle of zolpidem empty. GCS 8. BP 110/70, HR 52, RR 10. Flumazenil given with improvement."
```

## 🧠 Model Architecture

### BERT Training Pipeline
- **Base Model**: `emilyalsentzer/Bio_ClinicalBERT` (clinical domain pre-trained)
- **Task**: Multi-label classification for ICD codes
- **Input**: Clinical notes (with abbreviation expansion)
- **Output**: Multiple ICD codes with confidence scores

### Text Processing Pipeline
1. **Abbreviation Expansion**: Expand medical abbreviations using dictionary
2. **Entity Extraction**: Extract substances, symptoms, vital signs, treatments
3. **Severity Assessment**: Calculate clinical severity score
4. **BERT Prediction**: Generate ICD code predictions

### Supported ICD Code Categories
- **T-codes (T36-T50)**: Poisoning by drugs and medications
- **T-codes (T51-T65)**: Toxic effects of substances
- **R-codes (R40)**: Somnolence, stupor and coma
- **R-codes (R56)**: Convulsions
- **Z-codes (Z91.5)**: Personal history of self-harm

## 🔧 Configuration

### Model Settings (`config/settings.py`)
```python
BERT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
```

### Training Configuration
- **Model Selection**: Choose between different BERT models
- **Hyperparameters**: Adjust batch size, learning rate, epochs
- **Data Split**: Configure train/validation/test splits
- **Threshold**: Set confidence threshold for predictions

## 📈 Performance Metrics

The system provides comprehensive evaluation metrics:
- **Accuracy**: Overall prediction accuracy
- **Precision**: Precision for each ICD code
- **Recall**: Recall for each ICD code
- **F1 Score**: Harmonic mean of precision and recall
- **Per-class Metrics**: Detailed performance for each ICD code

## 🛠️ Technical Requirements

### Dependencies
- **Streamlit**: Web application framework
- **Transformers**: BERT model implementation
- **PyTorch**: Deep learning framework
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning utilities
- **spaCy**: Natural language processing
- **Pydantic**: Data validation

### System Requirements
- **Python**: 3.8+
- **RAM**: 8GB+ (16GB recommended for training)
- **GPU**: Optional but recommended for training
- **Storage**: 2GB+ for models and data

## 🚨 Important Notes

### Medical Disclaimer
This application is for educational and research purposes only. It should not be used for actual clinical decision-making without proper validation and medical oversight.

### Data Privacy
- Ensure all clinical data is properly anonymized
- Follow HIPAA and other relevant privacy regulations
- Do not upload sensitive patient information

### Model Limitations
- The model requires training on domain-specific data
- Performance depends on quality and quantity of training data
- Regular retraining may be necessary for optimal performance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed information

## 🔮 Future Enhancements

- **Real-time Training**: Live model updates during training
- **Advanced NER**: More sophisticated entity extraction
- **Multi-language Support**: Support for other languages
- **API Integration**: REST API for external applications
- **Mobile App**: Native mobile application
- **Cloud Deployment**: AWS/Azure deployment options