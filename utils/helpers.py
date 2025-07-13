"""Helper functions for toxicology ICD prediction application"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import json
import logging
from datetime import datetime

from config.settings import COLORS
from models.data_models import PredictionResult, ICDPrediction

logger = logging.getLogger(__name__)

class StreamlitHelpers:
    """Helper functions for Streamlit UI components"""
    
    @staticmethod
    def setup_page_config():
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="Toxicology ICD Predictor",
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    @staticmethod
    def create_sidebar():
        """Create the sidebar with navigation and settings"""
        with st.sidebar:
            st.title("🏥 Toxicology ICD Predictor")
            st.markdown("---")
            
            # Navigation
            st.subheader("Navigation")
            page = st.selectbox(
                "Choose a page:",
                ["Data Upload & Training", "Single Note Prediction", "Batch Processing", "Model Performance"]
            )
            
            st.markdown("---")
            
            # Settings
            st.subheader("Settings")
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Minimum confidence score for ICD predictions"
            )
            
            max_predictions = st.slider(
                "Max Predictions",
                min_value=1,
                max_value=10,
                value=5,
                help="Maximum number of ICD codes to predict"
            )
            
            st.markdown("---")
            
            # Model Status
            st.subheader("Model Status")
            model_loaded = st.checkbox("BERT Model Loaded", value=False, disabled=True)
            
            if model_loaded:
                st.success("✅ Model Ready")
            else:
                st.warning("⚠️ Model Not Loaded")
            
            st.markdown("---")
            
            # System Info
            st.subheader("System Info")
            st.text(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
            st.text(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        
        return page, confidence_threshold, max_predictions
    
    @staticmethod
    def display_header():
        """Display the main header"""
        st.title("🏥 Toxicology ICD Code Prediction System")
        st.markdown("""
        This application processes toxicology clinical notes and predicts appropriate ICD-10 codes 
        for poisoning and toxic effects. The system handles both clean and abbreviated note formats.
        """)
        st.markdown("---")
    
    @staticmethod
    def display_prediction_results(result: PredictionResult, max_predictions: int = 5):
        """Display prediction results in a formatted way"""
        if not result.predictions:
            st.warning("No predictions generated for this note.")
            return
        
        # Display top prediction prominently
        if result.top_prediction:
            st.subheader("🎯 Top Prediction")
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"**ICD Code:** {result.top_prediction.code}")
                st.markdown(f"**Description:** {result.top_prediction.description}")
            
            with col2:
                confidence_color = "green" if result.top_prediction.confidence > 0.8 else "orange" if result.top_prediction.confidence > 0.6 else "red"
                st.markdown(f"**Confidence:** :{confidence_color}[{result.top_prediction.confidence:.3f}]")
            
            with col3:
                st.markdown(f"**Category:** {result.top_prediction.category}")
            
            st.markdown(f"**Reasoning:** {result.top_prediction.reasoning}")
            st.markdown("---")
        
        # Display all predictions
        st.subheader(f"📋 All Predictions (Top {min(max_predictions, len(result.predictions))})")
        
        for i, prediction in enumerate(result.predictions[:max_predictions]):
            with st.expander(f"{prediction.code} - {prediction.description}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Description:** {prediction.description}")
                    st.markdown(f"**Category:** {prediction.category}")
                    st.markdown(f"**Reasoning:** {prediction.reasoning}")
                
                with col2:
                    confidence_color = "green" if prediction.confidence > 0.8 else "orange" if prediction.confidence > 0.6 else "red"
                    st.markdown(f"**Confidence:** :{confidence_color}[{prediction.confidence:.3f}]")
        
        # Display processing information
        st.markdown("---")
        st.subheader("⚙️ Processing Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Processing Time", f"{result.processing_time:.3f}s")
        
        with col2:
            st.metric("Model Version", result.model_version)
        
        with col3:
            st.metric("Total Predictions", len(result.predictions))
    
    @staticmethod
    def display_note_analysis(note: str, expanded_note: str, entities: List):
        """Display note analysis with entity highlighting"""
        st.subheader("📝 Note Analysis")
        
        # Original vs Expanded
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Note:**")
            st.text_area("", value=note, height=150, disabled=True)
        
        with col2:
            st.markdown("**Expanded Note:**")
            st.text_area("", value=expanded_note, height=150, disabled=True)
        
        # Entity highlighting
        if entities:
            st.markdown("**🔍 Extracted Entities:**")
            
            entity_types = {}
            for entity in entities:
                if entity.label not in entity_types:
                    entity_types[entity.label] = []
                entity_types[entity.label].append(entity)
            
            for entity_type, type_entities in entity_types.items():
                with st.expander(f"{entity_type} ({len(type_entities)})"):
                    for entity in type_entities:
                        st.markdown(f"- **{entity.text}** (confidence: {entity.confidence:.3f})")
    
    @staticmethod
    def display_batch_results(batch_result):
        """Display batch processing results"""
        st.subheader("📊 Batch Processing Results")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Notes", batch_result.summary['total_notes'])
        
        with col2:
            st.metric("Successful", batch_result.summary['successful_predictions'])
        
        with col3:
            st.metric("Failed", batch_result.summary['failed_predictions'])
        
        with col4:
            st.metric("Processing Time", f"{batch_result.processing_time:.2f}s")
        
        # Results table
        st.subheader("📋 Results Table")
        
        results_data = []
        for result in batch_result.results:
            row = {
                'Original Text': result.note.text[:100] + "..." if len(result.note.text) > 100 else result.note.text,
                'Substance': result.note.substance or "N/A",
                'Top ICD Code': result.top_prediction.code if result.top_prediction else "N/A",
                'Top Confidence': f"{result.top_prediction.confidence:.3f}" if result.top_prediction else "N/A",
                'Processing Time': f"{result.processing_time:.3f}s"
            }
            results_data.append(row)
        
        df = pd.DataFrame(results_data)
        st.dataframe(df, use_container_width=True)
    
    @staticmethod
    def display_training_metrics(metrics):
        """Display training metrics"""
        st.subheader("📈 Training Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics.accuracy:.4f}")
        
        with col2:
            st.metric("Precision", f"{metrics.precision:.4f}")
        
        with col3:
            st.metric("Recall", f"{metrics.recall:.4f}")
        
        with col4:
            st.metric("F1 Score", f"{metrics.f1_score:.4f}")
        
        # Per-class metrics if available
        if metrics.per_class_metrics:
            st.subheader("📊 Per-Class Metrics")
            
            class_data = []
            for icd_code, class_metrics in metrics.per_class_metrics.items():
                class_data.append({
                    'ICD Code': icd_code,
                    'Precision': f"{class_metrics['precision']:.4f}",
                    'Recall': f"{class_metrics['recall']:.4f}",
                    'F1 Score': f"{class_metrics['f1_score']:.4f}"
                })
            
            df = pd.DataFrame(class_data)
            st.dataframe(df, use_container_width=True)

class DataHelpers:
    """Helper functions for data processing"""
    
    @staticmethod
    def validate_note_text(text: str) -> Tuple[bool, str]:
        """Validate note text input"""
        if not text or not text.strip():
            return False, "Note text cannot be empty"
        
        if len(text.strip()) < 10:
            return False, "Note text must be at least 10 characters long"
        
        if len(text) > 10000:
            return False, "Note text is too long (max 10,000 characters)"
        
        return True, "Valid note text"
    
    @staticmethod
    def clean_note_text(text: str) -> str:
        """Clean and normalize note text"""
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove special characters that might cause issues
        text = text.replace('\x00', '')
        
        return text.strip()
    
    @staticmethod
    def format_icd_code(code: str) -> str:
        """Format ICD code for display"""
        # Ensure proper formatting (e.g., T42.4X1A)
        code = code.upper().strip()
        
        # Add missing parts if needed
        if code.startswith('T') and len(code) >= 4:
            if not code.endswith('A'):
                code += 'A'  # Add encounter type
        
        return code
    
    @staticmethod
    def get_icd_category_color(category: str) -> str:
        """Get color for ICD category"""
        color_map = {
            "T-code (Poisoning/Toxic effects)": "red",
            "R-code (Symptoms/Signs)": "orange",
            "Z-code (Factors influencing health)": "blue",
            "Other": "gray"
        }
        return color_map.get(category, "gray")

class FileHelpers:
    """Helper functions for file operations"""
    
    @staticmethod
    def save_uploaded_file(uploaded_file, directory: Path) -> Path:
        """Save uploaded file to directory"""
        directory.mkdir(parents=True, exist_ok=True)
        file_path = directory / uploaded_file.name
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    
    @staticmethod
    def get_file_extension(file_path: Path) -> str:
        """Get file extension"""
        return file_path.suffix.lower()
    
    @staticmethod
    def is_csv_file(file_path: Path) -> bool:
        """Check if file is CSV"""
        return file_path.suffix.lower() == '.csv'
    
    @staticmethod
    def create_download_link(data: str, filename: str, text: str):
        """Create download link for Streamlit"""
        import base64
        
        b64 = base64.b64encode(data.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
        return href

class ValidationHelpers:
    """Helper functions for data validation"""
    
    @staticmethod
    def validate_csv_columns(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
        """Validate CSV columns"""
        errors = []
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_icd_codes(icd_codes: List[str]) -> Tuple[bool, List[str]]:
        """Validate ICD code format"""
        errors = []
        
        for code in icd_codes:
            if not code or not code.strip():
                continue
            
            code = code.strip().upper()
            
            # Basic ICD-10 validation
            if not (code.startswith('T') or code.startswith('R') or code.startswith('Z')):
                errors.append(f"Invalid ICD code format: {code}")
            
            if len(code) < 3:
                errors.append(f"ICD code too short: {code}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_training_data(data: List) -> Tuple[bool, List[str]]:
        """Validate training data"""
        errors = []
        
        if not data:
            errors.append("No training data provided")
            return False, errors
        
        for i, item in enumerate(data):
            if not hasattr(item, 'note') or not item.note:
                errors.append(f"Item {i}: Missing or empty note")
            
            if not hasattr(item, 'icd_codes') or not item.icd_codes:
                errors.append(f"Item {i}: Missing ICD codes")
        
        return len(errors) == 0, errors

class LoggingHelpers:
    """Helper functions for logging"""
    
    @staticmethod
    def setup_logging(log_file: Path = None):
        """Setup logging configuration"""
        if log_file is None:
            log_file = Path("toxicology_app.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    @staticmethod
    def log_prediction_result(result: PredictionResult):
        """Log prediction result"""
        logger.info(f"Prediction completed in {result.processing_time:.3f}s")
        logger.info(f"Top prediction: {result.top_prediction.code if result.top_prediction else 'None'}")
        logger.info(f"Total predictions: {len(result.predictions)}")
    
    @staticmethod
    def log_batch_result(batch_result):
        """Log batch processing result"""
        logger.info(f"Batch processing completed")
        logger.info(f"Total notes: {batch_result.summary['total_notes']}")
        logger.info(f"Successful: {batch_result.summary['successful_predictions']}")
        logger.info(f"Failed: {batch_result.summary['failed_predictions']}")
        logger.info(f"Processing time: {batch_result.processing_time:.2f}s") 