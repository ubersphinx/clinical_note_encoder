"""ICD code prediction for toxicology notes"""
import time
import logging
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import json

from models.data_models import (
    ToxicologyNote, ICDPrediction, PredictionResult, 
    BatchPredictionResult, TrainingData
)
from models.text_processor import ToxicologyTextProcessor
from models.bert_trainer import ToxicologyBERTTrainer
from models.abbreviation_expander import AbbreviationExpander
from config.settings import TOXICOLOGY_ICD_CATEGORIES, SUBSTANCE_CATEGORIES

logger = logging.getLogger(__name__)

class ToxicologyICDPredictor:
    """Main predictor for toxicology ICD codes"""
    
    def __init__(self, model_path: Optional[Path] = None):
        """Initialize the ICD predictor"""
        self.text_processor = ToxicologyTextProcessor()
        self.abbreviation_expander = AbbreviationExpander()
        self.bert_trainer = None
        self.model_loaded = False
        self.model_path = model_path
        
        # ICD code descriptions and categories
        self.icd_descriptions = self._load_icd_descriptions()
        self.substance_to_icd_mapping = self._create_substance_icd_mapping()
        
        logger.info("Toxicology ICD Predictor initialized")
    
    def _load_icd_descriptions(self) -> Dict[str, str]:
        """Load ICD code descriptions"""
        # Default toxicology ICD codes with descriptions
        default_icd_codes = {
            # T-codes (Poisoning by drugs and medications)
            'T36.0X1A': 'Poisoning by penicillins, accidental (unintentional), initial encounter',
            'T36.1X1A': 'Poisoning by cephalosporins and other beta-lactam antibiotics, accidental (unintentional), initial encounter',
            'T36.2X1A': 'Poisoning by other systemic antibiotics, accidental (unintentional), initial encounter',
            'T36.3X1A': 'Poisoning by sulfonamides, accidental (unintentional), initial encounter',
            'T36.4X1A': 'Poisoning by other anti-infectives, accidental (unintentional), initial encounter',
            'T36.5X1A': 'Poisoning by antimycobacterial drugs, accidental (unintentional), initial encounter',
            'T36.6X1A': 'Poisoning by antiviral drugs, accidental (unintentional), initial encounter',
            'T36.8X1A': 'Poisoning by other systemic anti-infectives and antiparasitics, accidental (unintentional), initial encounter',
            'T36.9X1A': 'Poisoning by unspecified systemic anti-infective and antiparasitic, accidental (unintentional), initial encounter',
            
            'T37.0X1A': 'Poisoning by sulfonamides, accidental (unintentional), initial encounter',
            'T37.1X1A': 'Poisoning by antimycobacterial drugs, accidental (unintentional), initial encounter',
            'T37.2X1A': 'Poisoning by other systemic anti-infectives and antiparasitics, accidental (unintentional), initial encounter',
            'T37.3X1A': 'Poisoning by other systemic anti-infectives and antiparasitics, accidental (unintentional), initial encounter',
            'T37.4X1A': 'Poisoning by antiprotozoal drugs, accidental (unintentional), initial encounter',
            'T37.5X1A': 'Poisoning by antiviral drugs, accidental (unintentional), initial encounter',
            'T37.8X1A': 'Poisoning by other systemic anti-infectives and antiparasitics, accidental (unintentional), initial encounter',
            'T37.9X1A': 'Poisoning by unspecified systemic anti-infective and antiparasitic, accidental (unintentional), initial encounter',
            
            'T38.0X1A': 'Poisoning by glucocorticoids and synthetic analogues, accidental (unintentional), initial encounter',
            'T38.1X1A': 'Poisoning by thyroid hormones and substitutes, accidental (unintentional), initial encounter',
            'T38.2X1A': 'Poisoning by antithyroid drugs, accidental (unintentional), initial encounter',
            'T38.3X1A': 'Poisoning by insulin and oral hypoglycemic drugs, accidental (unintentional), initial encounter',
            'T38.4X1A': 'Poisoning by oral contraceptives, accidental (unintentional), initial encounter',
            'T38.5X1A': 'Poisoning by other estrogens and progestogens, accidental (unintentional), initial encounter',
            'T38.6X1A': 'Poisoning by antigonadotrophins, antiestrogens, antiandrogens, not elsewhere classified, accidental (unintentional), initial encounter',
            'T38.7X1A': 'Poisoning by androgens and anabolic congeners, accidental (unintentional), initial encounter',
            'T38.8X1A': 'Poisoning by other hormones and their synthetic substitutes, accidental (unintentional), initial encounter',
            'T38.9X1A': 'Poisoning by unspecified hormone and its synthetic substitutes, accidental (unintentional), initial encounter',
            
            'T39.0X1A': 'Poisoning by salicylates, accidental (unintentional), initial encounter',
            'T39.1X1A': 'Poisoning by 4-Aminophenol derivatives, accidental (unintentional), initial encounter',
            'T39.2X1A': 'Poisoning by pyrazolone derivatives, accidental (unintentional), initial encounter',
            'T39.3X1A': 'Poisoning by other nonsteroidal anti-inflammatory drugs, accidental (unintentional), initial encounter',
            'T39.4X1A': 'Poisoning by antirheumatics, not elsewhere classified, accidental (unintentional), initial encounter',
            'T39.8X1A': 'Poisoning by other nonopioid analgesics and antipyretics, not elsewhere classified, accidental (unintentional), initial encounter',
            'T39.9X1A': 'Poisoning by unspecified nonopioid analgesic and antipyretic, accidental (unintentional), initial encounter',
            
            'T40.0X1A': 'Poisoning by opium, accidental (unintentional), initial encounter',
            'T40.1X1A': 'Poisoning by heroin, accidental (unintentional), initial encounter',
            'T40.2X1A': 'Poisoning by other opioids, accidental (unintentional), initial encounter',
            'T40.3X1A': 'Poisoning by methadone, accidental (unintentional), initial encounter',
            'T40.4X1A': 'Poisoning by other synthetic narcotics, accidental (unintentional), initial encounter',
            'T40.5X1A': 'Poisoning by cocaine, accidental (unintentional), initial encounter',
            'T40.6X1A': 'Poisoning by other and unspecified narcotics, accidental (unintentional), initial encounter',
            'T40.7X1A': 'Poisoning by cannabis (derivatives), accidental (unintentional), initial encounter',
            'T40.8X1A': 'Poisoning by lysergide (LSD), accidental (unintentional), initial encounter',
            'T40.9X1A': 'Poisoning by other and unspecified psychodysleptics (hallucinogens), accidental (unintentional), initial encounter',
            
            'T41.0X1A': 'Poisoning by inhaled anesthetics, accidental (unintentional), initial encounter',
            'T41.1X1A': 'Poisoning by intravenous anesthetics, accidental (unintentional), initial encounter',
            'T41.2X1A': 'Poisoning by other and unspecified general anesthetics, accidental (unintentional), initial encounter',
            'T41.3X1A': 'Poisoning by local anesthetics, accidental (unintentional), initial encounter',
            'T41.4X1A': 'Poisoning by unspecified anesthetic, accidental (unintentional), initial encounter',
            'T41.5X1A': 'Poisoning by therapeutic gases, accidental (unintentional), initial encounter',
            
            'T42.0X1A': 'Poisoning by hydantoin derivatives, accidental (unintentional), initial encounter',
            'T42.1X1A': 'Poisoning by iminostilbenes, accidental (unintentional), initial encounter',
            'T42.2X1A': 'Poisoning by succinimides and oxazolidinediones, accidental (unintentional), initial encounter',
            'T42.3X1A': 'Poisoning by barbiturates, accidental (unintentional), initial encounter',
            'T42.4X1A': 'Poisoning by benzodiazepines, accidental (unintentional), initial encounter',
            'T42.5X1A': 'Poisoning by mixed antiepileptics, accidental (unintentional), initial encounter',
            'T42.6X1A': 'Poisoning by other antiepileptic and sedative-hypnotic drugs, accidental (unintentional), initial encounter',
            'T42.7X1A': 'Poisoning by antiparkinsonism drugs, accidental (unintentional), initial encounter',
            'T42.8X1A': 'Poisoning by other antiparkinsonism drugs and other central muscle-tone depressants, accidental (unintentional), initial encounter',
            
            'T43.0X1A': 'Poisoning by tricyclic and tetracyclic antidepressants, accidental (unintentional), initial encounter',
            'T43.1X1A': 'Poisoning by monoamine-oxidase-inhibitor antidepressants, accidental (unintentional), initial encounter',
            'T43.2X1A': 'Poisoning by other and unspecified antidepressants, accidental (unintentional), initial encounter',
            'T43.3X1A': 'Poisoning by phenothiazine antipsychotics and neuroleptics, accidental (unintentional), initial encounter',
            'T43.4X1A': 'Poisoning by butyrophenone and thioxanthene neuroleptics, accidental (unintentional), initial encounter',
            'T43.5X1A': 'Poisoning by other and unspecified antipsychotics and neuroleptics, accidental (unintentional), initial encounter',
            'T43.6X1A': 'Poisoning by psychostimulants with abuse potential, accidental (unintentional), initial encounter',
            'T43.8X1A': 'Poisoning by other psychotropic drugs, not elsewhere classified, accidental (unintentional), initial encounter',
            'T43.9X1A': 'Poisoning by unspecified psychotropic drug, accidental (unintentional), initial encounter',
            
            'T44.0X1A': 'Poisoning by anticholinesterase agents, accidental (unintentional), initial encounter',
            'T44.1X1A': 'Poisoning by other parasympathomimetics (cholinergics), accidental (unintentional), initial encounter',
            'T44.2X1A': 'Poisoning by ganglionic blocking drugs, accidental (unintentional), initial encounter',
            'T44.3X1A': 'Poisoning by other parasympatholytics (anticholinergics and antimuscarinics) and spasmolytics, accidental (unintentional), initial encounter',
            'T44.4X1A': 'Poisoning by predominantly alpha-adrenoreceptor agonists, accidental (unintentional), initial encounter',
            'T44.5X1A': 'Poisoning by predominantly beta-adrenoreceptor agonists, accidental (unintentional), initial encounter',
            'T44.6X1A': 'Poisoning by alpha-adrenoreceptor antagonists, accidental (unintentional), initial encounter',
            'T44.7X1A': 'Poisoning by beta-adrenoreceptor antagonists, accidental (unintentional), initial encounter',
            'T44.8X1A': 'Poisoning by centrally-acting and adrenergic-neuron-blocking agents, accidental (unintentional), initial encounter',
            'T44.9X1A': 'Poisoning by other and unspecified drugs primarily affecting the autonomic nervous system, accidental (unintentional), initial encounter',
            
            'T45.0X1A': 'Poisoning by antiallergic and antiemetic drugs, accidental (unintentional), initial encounter',
            'T45.1X1A': 'Poisoning by antineoplastic and immunosuppressive drugs, accidental (unintentional), initial encounter',
            'T45.2X1A': 'Poisoning by vitamins, accidental (unintentional), initial encounter',
            'T45.3X1A': 'Poisoning by enzymes, accidental (unintentional), initial encounter',
            'T45.4X1A': 'Poisoning by iron and its compounds, accidental (unintentional), initial encounter',
            'T45.5X1A': 'Poisoning by anticoagulants, accidental (unintentional), initial encounter',
            'T45.6X1A': 'Poisoning by antithrombotic drugs, accidental (unintentional), initial encounter',
            'T45.7X1A': 'Poisoning by thrombolytic drugs, accidental (unintentional), initial encounter',
            'T45.8X1A': 'Poisoning by other primarily systemic and hematological agents, accidental (unintentional), initial encounter',
            'T45.9X1A': 'Poisoning by unspecified primarily systemic and hematological agent, accidental (unintentional), initial encounter',
            
            'T46.0X1A': 'Poisoning by cardiac-stimulant glycosides and drugs of similar action, accidental (unintentional), initial encounter',
            'T46.1X1A': 'Poisoning by calcium-channel blockers, accidental (unintentional), initial encounter',
            'T46.2X1A': 'Poisoning by other antidysrhythmic drugs, accidental (unintentional), initial encounter',
            'T46.3X1A': 'Poisoning by coronary vasodilators, accidental (unintentional), initial encounter',
            'T46.4X1A': 'Poisoning by angiotensin-converting-enzyme inhibitors, accidental (unintentional), initial encounter',
            'T46.5X1A': 'Poisoning by other antihypertensive drugs, accidental (unintentional), initial encounter',
            'T46.6X1A': 'Poisoning by antihyperlipidemic and antiarteriosclerotic drugs, accidental (unintentional), initial encounter',
            'T46.7X1A': 'Poisoning by peripheral vasodilators, accidental (unintentional), initial encounter',
            'T46.8X1A': 'Poisoning by antivaricose drugs, including sclerosing agents, accidental (unintentional), initial encounter',
            'T46.9X1A': 'Poisoning by other and unspecified agents primarily affecting the cardiovascular system, accidental (unintentional), initial encounter',
            
            'T47.0X1A': 'Poisoning by histamine H2-receptor blockers, accidental (unintentional), initial encounter',
            'T47.1X1A': 'Poisoning by other antacids and anti-gastric-secretion drugs, accidental (unintentional), initial encounter',
            'T47.2X1A': 'Poisoning by stimulant laxatives, accidental (unintentional), initial encounter',
            'T47.3X1A': 'Poisoning by saline and osmotic laxatives, accidental (unintentional), initial encounter',
            'T47.4X1A': 'Poisoning by other laxatives, accidental (unintentional), initial encounter',
            'T47.5X1A': 'Poisoning by digestants, accidental (unintentional), initial encounter',
            'T47.6X1A': 'Poisoning by antidiarrheal drugs, accidental (unintentional), initial encounter',
            'T47.7X1A': 'Poisoning by emetics, accidental (unintentional), initial encounter',
            'T47.8X1A': 'Poisoning by other agents primarily affecting the gastrointestinal system, accidental (unintentional), initial encounter',
            'T47.9X1A': 'Poisoning by unspecified agent primarily affecting the gastrointestinal system, accidental (unintentional), initial encounter',
            
            'T48.0X1A': 'Poisoning by oxytocic drugs, accidental (unintentional), initial encounter',
            'T48.1X1A': 'Poisoning by skeletal muscle relaxants (neuromuscular blocking agents), accidental (unintentional), initial encounter',
            'T48.2X1A': 'Poisoning by other and unspecified drugs acting on muscles, accidental (unintentional), initial encounter',
            'T48.3X1A': 'Poisoning by antitussives, accidental (unintentional), initial encounter',
            'T48.4X1A': 'Poisoning by expectorants, accidental (unintentional), initial encounter',
            'T48.5X1A': 'Poisoning by other anti-common-cold drugs, accidental (unintentional), initial encounter',
            'T48.6X1A': 'Poisoning by antiasthmatics, accidental (unintentional), initial encounter',
            'T48.8X1A': 'Poisoning by other agents primarily acting on the smooth and skeletal muscles and the respiratory system, accidental (unintentional), initial encounter',
            'T48.9X1A': 'Poisoning by unspecified agent primarily acting on the smooth and skeletal muscles and the respiratory system, accidental (unintentional), initial encounter',
            
            'T49.0X1A': 'Poisoning by local antifungal, anti-infective and anti-inflammatory drugs, accidental (unintentional), initial encounter',
            'T49.1X1A': 'Poisoning by antipruritics, accidental (unintentional), initial encounter',
            'T49.2X1A': 'Poisoning by local astringents and local detergents, accidental (unintentional), initial encounter',
            'T49.3X1A': 'Poisoning by emollients, demulcents and protectants, accidental (unintentional), initial encounter',
            'T49.4X1A': 'Poisoning by keratolytics, keratoplastics, and other hair treatment drugs and preparations, accidental (unintentional), initial encounter',
            'T49.5X1A': 'Poisoning by ophthalmological drugs and preparations, accidental (unintentional), initial encounter',
            'T49.6X1A': 'Poisoning by otorhinolaryngological drugs and preparations, accidental (unintentional), initial encounter',
            'T49.7X1A': 'Poisoning by dental drugs, accidental (unintentional), initial encounter',
            'T49.8X1A': 'Poisoning by other topical agents, accidental (unintentional), initial encounter',
            'T49.9X1A': 'Poisoning by unspecified topical agent, accidental (unintentional), initial encounter',
            
            'T50.0X1A': 'Poisoning by mineralocorticoids and their antagonists, accidental (unintentional), initial encounter',
            'T50.1X1A': 'Poisoning by carbonic-anhydrase inhibitors, benzothiadiazides and other diuretics, accidental (unintentional), initial encounter',
            'T50.2X1A': 'Poisoning by other diuretics, accidental (unintentional), initial encounter',
            'T50.3X1A': 'Poisoning by electrolytic, caloric and water-balance agents, accidental (unintentional), initial encounter',
            'T50.4X1A': 'Poisoning by drugs affecting uric acid metabolism, accidental (unintentional), initial encounter',
            'T50.5X1A': 'Poisoning by appetite depressants, accidental (unintentional), initial encounter',
            'T50.6X1A': 'Poisoning by antacids and anti-gastric-secretion drugs, accidental (unintentional), initial encounter',
            'T50.7X1A': 'Poisoning by emetics, accidental (unintentional), initial encounter',
            'T50.8X1A': 'Poisoning by diagnostic agents, accidental (unintentional), initial encounter',
            'T50.9X1A': 'Poisoning by other and unspecified drugs, medicaments and biological substances, accidental (unintentional), initial encounter',
            
            # R-codes (Symptoms and signs)
            'R40.0': 'Somnolence',
            'R40.1': 'Stupor',
            'R40.2': 'Coma',
            'R40.20': 'Coma, unspecified',
            'R40.21': 'Coma scale, eyes open',
            'R40.22': 'Coma scale, best verbal response',
            'R40.23': 'Coma scale, best motor response',
            'R40.24': 'Coma scale, total',
            'R40.3': 'Persistent vegetative state',
            'R40.4': 'Transient alteration of awareness',
            
            'R56.0': 'Febrile convulsions',
            'R56.00': 'Simple febrile convulsions',
            'R56.01': 'Complex febrile convulsions',
            'R56.1': 'Post traumatic convulsions',
            'R56.8': 'Other and unspecified convulsions',
            'R56.9': 'Unspecified convulsions',
            
            # Z-codes (Factors influencing health status)
            'Z91.5': 'Personal history of self-harm',
            'Z91.50': 'Personal history of self-harm, unspecified',
            'Z91.51': 'Personal history of self-harm, self-poisoning',
            'Z91.52': 'Personal history of self-harm, self-injury',
            'Z91.59': 'Personal history of self-harm, other',
            
            'Z79.01': 'Long term (current) use of anticoagulants',
            'Z79.02': 'Long term (current) use of antithrombotics/antiplatelets',
            'Z79.1': 'Long term (current) use of non-steroidal anti-inflammatories (NSAID)',
            'Z79.2': 'Long term (current) use of antibiotics',
            'Z79.3': 'Long term (current) use of hormone replacement therapy',
            'Z79.4': 'Long term (current) use of insulin',
            'Z79.5': 'Long term (current) use of steroids',
            'Z79.8': 'Other long term (current) drug therapy',
            'Z79.9': 'Long term (current) drug therapy, unspecified'
        }
        
        return default_icd_codes
    
    def _create_substance_icd_mapping(self) -> Dict[str, List[str]]:
        """Create mapping from substances to likely ICD codes"""
        mapping = {}
        
        # Opioids
        opioid_substances = ['heroin', 'fentanyl', 'morphine', 'oxycodone', 'hydrocodone', 'codeine']
        for substance in opioid_substances:
            mapping[substance] = ['T40.1X1A', 'T40.2X1A', 'T40.3X1A', 'T40.4X1A', 'R40.20']
        
        # Benzodiazepines
        benzo_substances = ['alprazolam', 'diazepam', 'lorazepam', 'clonazepam', 'zolpidem']
        for substance in benzo_substances:
            mapping[substance] = ['T42.4X1A', 'R40.20']
        
        # Stimulants
        stimulant_substances = ['cocaine', 'methamphetamine', 'amphetamine', 'methylphenidate']
        for substance in stimulant_substances:
            mapping[substance] = ['T40.5X1A', 'T43.6X1A', 'R40.20']
        
        # Antidepressants
        antidepressant_substances = ['sertraline', 'fluoxetine', 'venlafaxine', 'bupropion']
        for substance in antidepressant_substances:
            mapping[substance] = ['T43.0X1A', 'T43.1X1A', 'T43.2X1A', 'R40.20']
        
        # Antipsychotics
        antipsychotic_substances = ['risperidone', 'olanzapine', 'quetiapine', 'haloperidol']
        for substance in antipsychotic_substances:
            mapping[substance] = ['T43.3X1A', 'T43.4X1A', 'T43.5X1A', 'R40.20']
        
        # Analgesics
        analgesic_substances = ['acetaminophen', 'aspirin', 'ibuprofen']
        for substance in analgesic_substances:
            mapping[substance] = ['T39.0X1A', 'T39.1X1A', 'T39.2X1A', 'T39.3X1A']
        
        # Hallucinogens
        hallucinogen_substances = ['lsd', 'psilocybin', 'mdma', 'ketamine']
        for substance in hallucinogen_substances:
            mapping[substance] = ['T40.8X1A', 'T40.9X1A', 'R40.20']
        
        return mapping
    
    def load_model(self) -> bool:
        """Load the trained BERT model"""
        try:
            from models.bert_trainer import TrainingConfig
            config = TrainingConfig()
            self.bert_trainer = ToxicologyBERTTrainer(config)
            self.bert_trainer.load_trained_model(self.model_path)
            self.model_loaded = True
            logger.info("BERT model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}")
            self.model_loaded = False
            return False
    
    def predict_single_note(self, text: str, threshold: float = 0.5) -> PredictionResult:
        """Predict ICD codes for a single toxicology note"""
        start_time = time.time()
        
        # Process the note
        processed_note = self.text_processor.process_note(text)
        
        # Get BERT predictions if model is loaded
        bert_predictions = []
        if self.model_loaded and self.bert_trainer:
            try:
                bert_predictions = self.bert_trainer.predict(text, threshold)
            except Exception as e:
                logger.warning(f"BERT prediction failed: {e}")
        
        # Get rule-based predictions
        rule_predictions = self._get_rule_based_predictions(processed_note)
        
        # Combine predictions
        all_predictions = self._combine_predictions(bert_predictions, rule_predictions)
        
        # Create ICD prediction objects
        icd_predictions = []
        for icd_code, confidence, reasoning in all_predictions:
            description = self.icd_descriptions.get(icd_code, f"ICD code {icd_code}")
            category = self._get_icd_category(icd_code)
            
            icd_predictions.append(ICDPrediction(
                code=icd_code,
                description=description,
                confidence=confidence,
                category=category,
                reasoning=reasoning
            ))
        
        # Sort by confidence
        icd_predictions.sort(key=lambda x: x.confidence, reverse=True)
        
        # Get top prediction
        top_prediction = icd_predictions[0] if icd_predictions else None
        
        processing_time = time.time() - start_time
        
        return PredictionResult(
            note=processed_note,
            predictions=icd_predictions,
            top_prediction=top_prediction,
            processing_time=processing_time,
            model_version="1.0.0"
        )
    
    def _get_rule_based_predictions(self, processed_note: ToxicologyNote) -> List[Tuple[str, float, str]]:
        """Get rule-based ICD predictions"""
        predictions = []
        
        # Substance-based predictions
        if processed_note.substance:
            substance_lower = processed_note.substance.lower()
            
            # Check substance mapping
            if substance_lower in self.substance_to_icd_mapping:
                for icd_code in self.substance_to_icd_mapping[substance_lower]:
                    predictions.append((
                        icd_code,
                        0.8,
                        f"Substance detected: {processed_note.substance}"
                    ))
            
            # Generic substance predictions
            if 'opioid' in substance_lower or any(op in substance_lower for op in ['heroin', 'fentanyl', 'morphine']):
                predictions.append(('T40.2X1A', 0.7, f"Opioid substance: {processed_note.substance}"))
            
            if 'benzodiazepine' in substance_lower or any(bz in substance_lower for bz in ['alprazolam', 'diazepam', 'zolpidem']):
                predictions.append(('T42.4X1A', 0.7, f"Benzodiazepine substance: {processed_note.substance}"))
        
        # Symptom-based predictions
        symptoms = processed_note.symptoms
        if 'altered mental status' in symptoms or 'ams' in symptoms:
            predictions.append(('R40.20', 0.8, "Altered mental status detected"))
        
        if 'seizure' in symptoms or 'convulsion' in symptoms:
            predictions.append(('R56.9', 0.8, "Seizure activity detected"))
        
        if 'unresponsive' in symptoms or 'comatose' in symptoms:
            predictions.append(('R40.20', 0.9, "Comatose state detected"))
        
        # Route-based predictions
        if processed_note.route:
            if processed_note.route == 'intravenous':
                predictions.append(('T50.9X1A', 0.6, "Intravenous route of exposure"))
            elif processed_note.route == 'oral':
                predictions.append(('T50.9X1A', 0.6, "Oral route of exposure"))
        
        # Severity-based predictions
        if processed_note.severity_score and processed_note.severity_score > 7:
            predictions.append(('R40.20', 0.7, "High severity score indicating altered consciousness"))
        
        return predictions
    
    def _combine_predictions(self, bert_predictions: List[Tuple[str, float]], 
                           rule_predictions: List[Tuple[str, float, str]]) -> List[Tuple[str, float, str]]:
        """Combine BERT and rule-based predictions"""
        combined = {}
        
        # Add BERT predictions
        for icd_code, confidence in bert_predictions:
            combined[icd_code] = (confidence, f"BERT model prediction")
        
        # Add rule-based predictions
        for icd_code, confidence, reasoning in rule_predictions:
            if icd_code in combined:
                # Average confidence if both sources predict same code
                existing_conf, existing_reason = combined[icd_code]
                avg_confidence = (existing_conf + confidence) / 2
                combined[icd_code] = (avg_confidence, f"{existing_reason}; {reasoning}")
            else:
                combined[icd_code] = (confidence, reasoning)
        
        # Convert to list and sort
        result = [(code, conf, reason) for code, (conf, reason) in combined.items()]
        result.sort(key=lambda x: x[1], reverse=True)
        
        return result
    
    def _get_icd_category(self, icd_code: str) -> str:
        """Get the category of an ICD code"""
        if icd_code.startswith('T'):
            return "T-code (Poisoning/Toxic effects)"
        elif icd_code.startswith('R'):
            return "R-code (Symptoms/Signs)"
        elif icd_code.startswith('Z'):
            return "Z-code (Factors influencing health)"
        else:
            return "Other"
    
    def predict_batch(self, texts: List[str], threshold: float = 0.5) -> BatchPredictionResult:
        """Predict ICD codes for multiple notes"""
        start_time = time.time()
        results = []
        success_count = 0
        error_count = 0
        
        for i, text in enumerate(texts):
            try:
                result = self.predict_single_note(text, threshold)
                results.append(result)
                success_count += 1
            except Exception as e:
                logger.error(f"Error processing note {i}: {e}")
                error_count += 1
        
        processing_time = time.time() - start_time
        
        # Calculate summary statistics
        summary = {
            'total_notes': len(texts),
            'successful_predictions': success_count,
            'failed_predictions': error_count,
            'avg_processing_time': processing_time / len(texts) if texts else 0,
            'avg_predictions_per_note': sum(len(r.predictions) for r in results) / len(results) if results else 0
        }
        
        return BatchPredictionResult(
            results=results,
            summary=summary,
            processing_time=processing_time,
            success_count=success_count,
            error_count=error_count
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        info = {
            'model_loaded': self.model_loaded,
            'text_processor_available': True,
            'abbreviation_expander_available': True,
            'total_icd_codes': len(self.icd_descriptions),
            'substance_mappings': len(self.substance_to_icd_mapping)
        }
        
        if self.bert_trainer:
            info.update(self.bert_trainer.get_model_summary())
        
        return info 