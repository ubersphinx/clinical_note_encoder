"""Toxicology ICD Code Prediction - Streamlit Application"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time
from typing import List, Dict, Optional

# Import our modules
from config.settings import DATA_FILES, MODEL_FILES, COLORS
from models.icd_predictor import ToxicologyICDPredictor
from models.bert_trainer import ToxicologyBERTTrainer, TrainingConfig
from models.text_processor import ToxicologyTextProcessor
from models.abbreviation_expander import AbbreviationExpander
from utils.csv_handler import CSVHandler
from utils.model_utils import ModelUtils
from utils.helpers import (
    StreamlitHelpers, DataHelpers, FileHelpers, 
    ValidationHelpers, LoggingHelpers
)

# Setup logging
LoggingHelpers.setup_logging()

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'training_data' not in st.session_state:
    st.session_state.training_data = None

def main():
    """Main application function"""
    # Setup page configuration
    StreamlitHelpers.setup_page_config()
    
    # Create sidebar
    page, confidence_threshold, max_predictions = StreamlitHelpers.create_sidebar()
    
    # Display header
    StreamlitHelpers.display_header()
    
    # Initialize predictor if not already done
    if st.session_state.predictor is None:
        with st.spinner("Initializing toxicology predictor..."):
            st.session_state.predictor = ToxicologyICDPredictor()
            st.session_state.model_loaded = st.session_state.predictor.load_model()
    
    # Route to appropriate page
    if page == "Data Upload & Training":
        data_upload_training_page(confidence_threshold, max_predictions)
    elif page == "Single Note Prediction":
        single_note_prediction_page(confidence_threshold, max_predictions)
    elif page == "Batch Processing":
        batch_processing_page(confidence_threshold, max_predictions)
    elif page == "Model Performance":
        model_performance_page()

def data_upload_training_page(confidence_threshold: float, max_predictions: int):
    """Data Upload & Training page"""
    st.header("📊 Data Upload & Training")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📁 Upload Data", "🤖 Train Model", "📈 Training Metrics"])
    
    with tab1:
        st.subheader("Upload Training Data")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file with training data",
            type=['csv'],
            help="CSV should have 'Note' and 'ICD_Codes' columns"
        )
        
        if uploaded_file is not None:
            # Save uploaded file
            csv_handler = CSVHandler()
            file_path = FileHelpers.save_uploaded_file(uploaded_file, Path("data"))
            
            # Validate file
            is_valid, errors = csv_handler.validate_csv_format(file_path)
            
            if is_valid:
                st.success("✅ File uploaded and validated successfully!")
                
                # Load and display data
                training_data = csv_handler.load_training_data(file_path)
                st.session_state.training_data = training_data
                
                # Display data statistics
                stats = csv_handler.get_data_statistics(file_path)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Samples", stats.get('total_samples', 0))
                with col2:
                    st.metric("Unique ICD Codes", stats.get('unique_icd_codes', 0))
                with col3:
                    st.metric("Avg ICD Codes/Note", f"{stats.get('avg_icd_codes_per_note', 0):.2f}")
                
                # Display sample data
                if training_data:
                    st.subheader("Sample Data")
                    sample_df = pd.DataFrame([
                        {
                            'Note': item.note[:100] + "..." if len(item.note) > 100 else item.note,
                            'ICD_Codes': ', '.join(item.icd_codes),
                            'Processed': 'Yes' if item.processed_note else 'No'
                        }
                        for item in training_data[:5]
                    ])
                    st.dataframe(sample_df, use_container_width=True)
                
            else:
                st.error("❌ File validation failed:")
                for error in errors:
                    st.error(f"- {error}")
        
        # Create sample data option
        st.subheader("Create Sample Data")
        if st.button("Generate Sample Training Data"):
            csv_handler = CSVHandler()
            sample_path = Path("data/sample_toxicology_data.csv")
            csv_handler.create_sample_data(sample_path, num_samples=100)
            st.success(f"✅ Sample data created at {sample_path}")
    
    with tab2:
        st.subheader("Train BERT Model")
        
        if st.session_state.training_data is None:
            st.warning("⚠️ Please upload training data first.")
            return
        
        # Training configuration
        st.subheader("Training Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_name = st.selectbox(
                "BERT Model",
                ["emilyalsentzer/Bio_ClinicalBERT", "bert-base-uncased"],
                index=0
            )
            
            max_length = st.slider("Max Sequence Length", 256, 512, 512, 64)
            batch_size = st.selectbox("Batch Size", [4, 8, 16, 32], index=1)
        
        with col2:
            learning_rate = st.selectbox(
                "Learning Rate",
                [1e-5, 2e-5, 3e-5, 5e-5],
                index=1,
                format_func=lambda x: f"{x:.0e}"
            )
            
            num_epochs = st.slider("Number of Epochs", 1, 10, 3)
            warmup_steps = st.slider("Warmup Steps", 50, 200, 100, 10)
        
        # Training button
        if st.button("🚀 Start Training", type="primary"):
            with st.spinner("Training model..."):
                try:
                    # Create training configuration
                    config = TrainingConfig(
                        model_name=model_name,
                        max_length=max_length,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        num_epochs=num_epochs,
                        warmup_steps=warmup_steps
                    )
                    
                    # Initialize trainer
                    trainer = ToxicologyBERTTrainer(config)
                    
                    # Prepare data
                    texts, labels, label_encoder = trainer.prepare_data(st.session_state.training_data)
                    
                    # Load model
                    trainer.load_model()
                    
                    # Create data loaders
                    train_loader, val_loader = trainer.create_data_loaders(texts, labels)
                    
                    # Train model
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    history = trainer.train_model(train_loader, val_loader)
                    
                    # Evaluate model
                    metrics = trainer.evaluate_model(val_loader)
                    
                    # Save model
                    trainer.save_model()
                    
                    st.success("✅ Model training completed successfully!")
                    
                    # Update session state
                    st.session_state.predictor.bert_trainer = trainer
                    st.session_state.model_loaded = True
                    
                    # Store results
                    st.session_state.training_history = history
                    st.session_state.training_metrics = metrics
                    
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
    
    with tab3:
        st.subheader("Training Metrics")
        
        if 'training_metrics' in st.session_state:
            StreamlitHelpers.display_training_metrics(st.session_state.training_metrics)
            
            if 'training_history' in st.session_state:
                st.subheader("Training History")
                
                # Plot training curves
                try:
                    import matplotlib.pyplot as plt
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                    
                    # Loss plot
                    ax1.plot(st.session_state.training_history['train_loss'], label='Train Loss')
                    ax1.plot(st.session_state.training_history['val_loss'], label='Validation Loss')
                    ax1.set_title('Training and Validation Loss')
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Loss')
                    ax1.legend()
                    ax1.grid(True)
                    
                    # Accuracy plot
                    ax2.plot(st.session_state.training_history['train_accuracy'], label='Train Accuracy')
                    ax2.plot(st.session_state.training_history['val_accuracy'], label='Validation Accuracy')
                    ax2.set_title('Training and Validation Accuracy')
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('Accuracy')
                    ax2.legend()
                    ax2.grid(True)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error plotting training history: {e}")
        else:
            st.info("📊 Training metrics will appear here after model training.")

def single_note_prediction_page(confidence_threshold: float, max_predictions: int):
    """Single Note Prediction page"""
    st.header("🔍 Single Note Prediction")
    
    # Check if model is loaded
    if not st.session_state.model_loaded:
        st.warning("⚠️ Model not loaded. Please train a model first.")
        return
    
    # Input section
    st.subheader("Enter Toxicology Note")
    
    # Text input
    note_text = st.text_area(
        "Clinical Note",
        height=200,
        placeholder="Enter the toxicology clinical note here...",
        help="Enter the clinical note in either clean or abbreviated format"
    )
    
    # Example notes
    with st.expander("📋 Example Notes"):
        examples = [
            "CC: Altered mental status. HPI: 27-year-old male found unresponsive at home. Empty bottle of alprazolam found. GCS 6. BP 90/60, HR 45, RR 8. Given naloxone with minimal response.",
            "pt found unresponsive @ home, bottle of zolpidem empty. GCS 8. BP 110/70, HR 52, RR 10. Flumazenil given with improvement.",
            "CC: Overdose. HPI: 19-year-old female ingested 30 acetaminophen tablets. Nausea and vomiting. LFTs elevated. Given N-acetylcysteine.",
            "pt found with needle in arm, heroin overdose suspected. Unresponsive, pinpoint pupils. Given naloxone, improved."
        ]
        
        for i, example in enumerate(examples, 1):
            if st.button(f"Example {i}", key=f"example_{i}"):
                st.session_state.note_text = example
                st.rerun()
    
    # Prediction button
    if st.button("🔮 Predict ICD Codes", type="primary") and note_text:
        # Validate input
        is_valid, message = DataHelpers.validate_note_text(note_text)
        
        if not is_valid:
            st.error(f"❌ {message}")
            return
        
        # Clean text
        cleaned_text = DataHelpers.clean_note_text(note_text)
        
        # Make prediction
        with st.spinner("Processing note and predicting ICD codes..."):
            try:
                result = st.session_state.predictor.predict_single_note(
                    cleaned_text, 
                    threshold=confidence_threshold
                )
                
                # Log result
                LoggingHelpers.log_prediction_result(result)
                
                # Display results
                StreamlitHelpers.display_prediction_results(result, max_predictions)
                
                # Display note analysis
                StreamlitHelpers.display_note_analysis(
                    result.note.text,
                    result.note.expanded_text,
                    result.note.entities
                )
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
    
    elif not note_text:
        st.info("ℹ️ Please enter a clinical note to get predictions.")

def batch_processing_page(confidence_threshold: float, max_predictions: int):
    """Batch Processing page"""
    st.header("📦 Batch Processing")
    
    # Check if model is loaded
    if not st.session_state.model_loaded:
        st.warning("⚠️ Model not loaded. Please train a model first.")
        return
    
    # File upload
    st.subheader("Upload CSV File")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with notes",
        type=['csv'],
        help="CSV should have a 'Note' column with clinical notes"
    )
    
    if uploaded_file is not None:
        # Save uploaded file
        file_path = FileHelpers.save_uploaded_file(uploaded_file, Path("temp"))
        
        # Process file
        try:
            df = pd.read_csv(file_path)
            
            if 'Note' not in df.columns:
                st.error("❌ CSV file must have a 'Note' column")
                return
            
            st.success(f"✅ Loaded {len(df)} notes from file")
            
            # Display sample
            st.subheader("Sample Data")
            st.dataframe(df.head(), use_container_width=True)
            
            # Process button
            if st.button("🔄 Process All Notes", type="primary"):
                with st.spinner("Processing notes..."):
                    try:
                        # Get notes
                        notes = df['Note'].dropna().tolist()
                        
                        # Make predictions
                        batch_result = st.session_state.predictor.predict_batch(
                            notes, 
                            threshold=confidence_threshold
                        )
                        
                        # Log results
                        LoggingHelpers.log_batch_result(batch_result)
                        
                        # Display results
                        StreamlitHelpers.display_batch_results(batch_result)
                        
                        # Export results
                        st.subheader("📥 Export Results")
                        
                        # Create download link
                        csv_handler = CSVHandler()
                        output_path = Path("temp/batch_results.csv")
                        csv_handler.export_batch_results(batch_result, output_path)
                        
                        with open(output_path, 'r') as f:
                            csv_data = f.read()
                        
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv_data,
                            file_name="toxicology_predictions.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Batch processing failed: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")

def model_performance_page():
    """Model Performance page"""
    st.header("📊 Model Performance")
    
    # Model information
    st.subheader("🤖 Model Information")
    
    if st.session_state.predictor:
        model_info = st.session_state.predictor.get_model_info()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Loaded", "Yes" if model_info['model_loaded'] else "No")
            st.metric("Text Processor", "Available" if model_info['text_processor_available'] else "Not Available")
        
        with col2:
            st.metric("Total ICD Codes", model_info.get('total_icd_codes', 0))
            st.metric("Substance Mappings", model_info.get('substance_mappings', 0))
        
        with col3:
            if 'num_labels' in model_info:
                st.metric("BERT Labels", model_info['num_labels'])
            if 'device' in model_info:
                st.metric("Device", model_info['device'])
    
    # Performance metrics
    if 'training_metrics' in st.session_state:
        st.subheader("📈 Performance Metrics")
        StreamlitHelpers.display_training_metrics(st.session_state.training_metrics)
    
    # Model evaluation
    st.subheader("🧪 Model Evaluation")
    
    # Test with sample notes
    test_notes = [
        "CC: Altered mental status. HPI: 27-year-old male found unresponsive at home. Empty bottle of alprazolam found. GCS 6. BP 90/60, HR 45, RR 8. Given naloxone with minimal response.",
        "pt found unresponsive @ home, bottle of zolpidem empty. GCS 8. BP 110/70, HR 52, RR 10. Flumazenil given with improvement.",
        "CC: Overdose. HPI: 19-year-old female ingested 30 acetaminophen tablets. Nausea and vomiting. LFTs elevated. Given N-acetylcysteine.",
        "pt found with needle in arm, heroin overdose suspected. Unresponsive, pinpoint pupils. Given naloxone, improved."
    ]
    
    if st.button("🧪 Run Test Predictions"):
        with st.spinner("Running test predictions..."):
            try:
                results = []
                for note in test_notes:
                    result = st.session_state.predictor.predict_single_note(note)
                    results.append(result)
                
                # Display test results
                st.subheader("Test Results")
                
                for i, result in enumerate(results, 1):
                    with st.expander(f"Test Note {i}"):
                        st.text_area("Note", value=result.note.text, height=100, disabled=True)
                        
                        if result.predictions:
                            st.markdown("**Predictions:**")
                            for pred in result.predictions[:3]:
                                st.markdown(f"- {pred.code}: {pred.description} (confidence: {pred.confidence:.3f})")
                        else:
                            st.warning("No predictions generated")
                
            except Exception as e:
                st.error(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    main() 