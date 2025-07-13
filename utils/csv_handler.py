"""CSV data handling for toxicology notes"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import logging
import json

from models.data_models import TrainingData, PredictionResult, BatchPredictionResult
from models.text_processor import ToxicologyTextProcessor

logger = logging.getLogger(__name__)

class CSVHandler:
    """Handles CSV data ingestion and processing for toxicology notes"""
    
    def __init__(self):
        """Initialize the CSV handler"""
        self.text_processor = ToxicologyTextProcessor()
    
    def load_training_data(self, file_path: Path) -> List[TrainingData]:
        """Load training data from CSV file"""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} rows from {file_path}")
            
            training_data = []
            for _, row in df.iterrows():
                note = row.get('Note', row.get('note', ''))
                icd_codes_str = row.get('ICD_Codes', row.get('icd_codes', ''))
                
                # Parse ICD codes (comma-separated)
                icd_codes = []
                if icd_codes_str:
                    icd_codes = [code.strip() for code in str(icd_codes_str).split(',') if code.strip()]
                
                # Process the note
                processed_note = self.text_processor.process_note(note)
                
                training_data.append(TrainingData(
                    note=note,
                    icd_codes=icd_codes,
                    processed_note=processed_note
                ))
            
            logger.info(f"Processed {len(training_data)} training samples")
            return training_data
            
        except Exception as e:
            logger.error(f"Error loading training data from {file_path}: {e}")
            return []
    
    def validate_csv_format(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Validate CSV file format for training data"""
        errors = []
        
        try:
            df = pd.read_csv(file_path)
            
            # Check required columns
            required_columns = ['Note', 'ICD_Codes']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                errors.append(f"Missing required columns: {missing_columns}")
            
            # Check for empty notes
            if 'Note' in df.columns:
                empty_notes = df['Note'].isna().sum()
                if empty_notes > 0:
                    errors.append(f"Found {empty_notes} empty notes")
            
            # Check for empty ICD codes
            if 'ICD_Codes' in df.columns:
                empty_icd = df['ICD_Codes'].isna().sum()
                if empty_icd > 0:
                    errors.append(f"Found {empty_icd} rows without ICD codes")
            
            # Check data types
            if 'Note' in df.columns and not df['Note'].dtype == 'object':
                errors.append("Note column should contain text data")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Error reading CSV file: {e}")
            return False, errors
    
    def create_sample_data(self, output_path: Path, num_samples: int = 100):
        """Create sample training data"""
        sample_data = []
        
        # Sample toxicology notes with ICD codes
        samples = [
            {
                'note': 'CC: Altered mental status. HPI: 27-year-old male found unresponsive at home. Empty bottle of alprazolam found. GCS 6. BP 90/60, HR 45, RR 8. Given naloxone with minimal response.',
                'icd_codes': 'T42.4X1A, R40.20'
            },
            {
                'note': 'pt found unresponsive @ home, bottle of zolpidem empty. GCS 8. BP 110/70, HR 52, RR 10. Flumazenil given with improvement.',
                'icd_codes': 'T42.6X1A, R40.20'
            },
            {
                'note': 'CC: Overdose. HPI: 19-year-old female ingested 30 acetaminophen tablets. Nausea and vomiting. LFTs elevated. Given N-acetylcysteine.',
                'icd_codes': 'T39.1X1A, R11.0'
            },
            {
                'note': 'pt found with needle in arm, heroin overdose suspected. Unresponsive, pinpoint pupils. Given naloxone, improved.',
                'icd_codes': 'T40.1X1A, R40.20'
            },
            {
                'note': 'CC: Seizure. HPI: 45-year-old male with cocaine use, found seizing. BP 180/110, HR 120. Given benzodiazepines.',
                'icd_codes': 'T40.5X1A, R56.9'
            },
            {
                'note': 'pt ingested unknown pills, altered mental status. GCS 10. BP 100/65, HR 85. Activated charcoal given.',
                'icd_codes': 'T50.9X1A, R40.20'
            },
            {
                'note': 'CC: Overdose. HPI: 32-year-old female took 20 sertraline tablets. Tremors, confusion. Given IV fluids.',
                'icd_codes': 'T43.2X1A, R40.20'
            },
            {
                'note': 'pt found unconscious, empty bottle of oxycodone. GCS 3. BP 80/50, HR 40. Intubated, given naloxone.',
                'icd_codes': 'T40.2X1A, R40.20'
            },
            {
                'note': 'CC: Drug overdose. HPI: 28-year-old male ingested LSD, hallucinating. BP 140/90, HR 110. Calm environment.',
                'icd_codes': 'T40.8X1A, R44.3'
            },
            {
                'note': 'pt took 50 aspirin tablets, tinnitus, nausea. BP 120/80, HR 90. Given activated charcoal.',
                'icd_codes': 'T39.0X1A, H93.19'
            }
        ]
        
        # Generate additional samples
        for i in range(num_samples):
            sample_idx = i % len(samples)
            sample = samples[sample_idx].copy()
            
            # Add some variation
            if i > len(samples) - 1:
                sample['note'] = f"Sample {i+1}: {sample['note']}"
            
            sample_data.append(sample)
        
        # Create DataFrame and save
        df = pd.DataFrame(sample_data)
        df.to_csv(output_path, index=False)
        logger.info(f"Created sample data with {len(sample_data)} samples at {output_path}")
    
    def export_predictions(self, predictions: List[PredictionResult], output_path: Path):
        """Export predictions to CSV"""
        try:
            rows = []
            
            for result in predictions:
                row = {
                    'original_text': result.note.text,
                    'expanded_text': result.note.expanded_text,
                    'substance': result.note.substance,
                    'route': result.note.route,
                    'dose': result.note.dose,
                    'symptoms': '; '.join(result.note.symptoms),
                    'severity_score': result.note.severity_score,
                    'top_icd_code': result.top_prediction.code if result.top_prediction else '',
                    'top_confidence': result.top_prediction.confidence if result.top_prediction else 0.0,
                    'all_predictions': '; '.join([f"{p.code}({p.confidence:.3f})" for p in result.predictions]),
                    'processing_time': result.processing_time
                }
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(rows)} predictions to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting predictions: {e}")
    
    def export_batch_results(self, batch_result: BatchPredictionResult, output_path: Path):
        """Export batch prediction results to CSV"""
        try:
            rows = []
            
            for result in batch_result.results:
                row = {
                    'original_text': result.note.text,
                    'expanded_text': result.note.expanded_text,
                    'substance': result.note.substance,
                    'route': result.note.route,
                    'dose': result.note.dose,
                    'symptoms': '; '.join(result.note.symptoms),
                    'severity_score': result.note.severity_score,
                    'top_icd_code': result.top_prediction.code if result.top_prediction else '',
                    'top_confidence': result.top_prediction.confidence if result.top_prediction else 0.0,
                    'all_predictions': '; '.join([f"{p.code}({p.confidence:.3f})" for p in result.predictions]),
                    'processing_time': result.processing_time
                }
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            logger.info(f"Exported batch results with {len(rows)} predictions to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting batch results: {e}")
    
    def get_data_statistics(self, file_path: Path) -> Dict[str, Any]:
        """Get statistics about the training data"""
        try:
            df = pd.read_csv(file_path)
            
            stats = {
                'total_samples': len(df),
                'columns': list(df.columns),
                'missing_values': df.isnull().sum().to_dict(),
                'data_types': df.dtypes.to_dict()
            }
            
            # ICD code statistics
            if 'ICD_Codes' in df.columns:
                all_icd_codes = []
                for icd_str in df['ICD_Codes'].dropna():
                    codes = [code.strip() for code in str(icd_str).split(',') if code.strip()]
                    all_icd_codes.extend(codes)
                
                icd_counts = pd.Series(all_icd_codes).value_counts()
                stats['icd_code_counts'] = icd_counts.head(20).to_dict()
                stats['unique_icd_codes'] = len(icd_counts)
                stats['avg_icd_codes_per_note'] = len(all_icd_codes) / len(df)
            
            # Note length statistics
            if 'Note' in df.columns:
                note_lengths = df['Note'].str.len()
                stats['note_length_stats'] = {
                    'mean': note_lengths.mean(),
                    'median': note_lengths.median(),
                    'min': note_lengths.min(),
                    'max': note_lengths.max(),
                    'std': note_lengths.std()
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting data statistics: {e}")
            return {}
    
    def split_data(self, file_path: Path, train_ratio: float = 0.8, 
                   val_ratio: float = 0.1, test_ratio: float = 0.1) -> Tuple[Path, Path, Path]:
        """Split data into train/validation/test sets"""
        try:
            df = pd.read_csv(file_path)
            
            # Shuffle the data
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Calculate split indices
            n = len(df)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            # Split the data
            train_df = df[:train_end]
            val_df = df[train_end:val_end]
            test_df = df[val_end:]
            
            # Save splits
            base_path = file_path.parent
            train_path = base_path / f"{file_path.stem}_train.csv"
            val_path = base_path / f"{file_path.stem}_val.csv"
            test_path = base_path / f"{file_path.stem}_test.csv"
            
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            logger.info(f"Split data: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
            
            return train_path, val_path, test_path
            
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            return None, None, None
    
    def merge_csv_files(self, file_paths: List[Path], output_path: Path):
        """Merge multiple CSV files into one"""
        try:
            dfs = []
            for file_path in file_paths:
                if file_path.exists():
                    df = pd.read_csv(file_path)
                    dfs.append(df)
                    logger.info(f"Loaded {len(df)} rows from {file_path}")
            
            if dfs:
                merged_df = pd.concat(dfs, ignore_index=True)
                merged_df.to_csv(output_path, index=False)
                logger.info(f"Merged {len(merged_df)} total rows to {output_path}")
            else:
                logger.warning("No valid CSV files to merge")
                
        except Exception as e:
            logger.error(f"Error merging CSV files: {e}")
    
    def clean_data(self, file_path: Path, output_path: Path):
        """Clean and preprocess the data"""
        try:
            df = pd.read_csv(file_path)
            
            # Remove duplicates
            initial_count = len(df)
            df = df.drop_duplicates()
            logger.info(f"Removed {initial_count - len(df)} duplicate rows")
            
            # Remove rows with empty notes
            df = df.dropna(subset=['Note'])
            logger.info(f"Removed rows with empty notes, {len(df)} remaining")
            
            # Clean ICD codes
            if 'ICD_Codes' in df.columns:
                df['ICD_Codes'] = df['ICD_Codes'].fillna('')
                # Remove invalid ICD codes (basic validation)
                df['ICD_Codes'] = df['ICD_Codes'].apply(
                    lambda x: ','.join([code.strip() for code in str(x).split(',') 
                                      if code.strip() and len(code.strip()) >= 3])
                )
            
            # Save cleaned data
            df.to_csv(output_path, index=False)
            logger.info(f"Cleaned data saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error cleaning data: {e}") 