"""Toxicology-specific text processing and NER"""
import re
import spacy
from typing import List, Dict, Optional, Tuple, Any
import logging
from pathlib import Path
import json

from config.settings import CLINICAL_ENTITIES, VITAL_SIGNS_PATTERNS, SUBSTANCE_CATEGORIES
from models.data_models import ToxicologyNote, ToxicologyEntity, VitalSigns
from models.abbreviation_expander import AbbreviationExpander

logger = logging.getLogger(__name__)

class ToxicologyTextProcessor:
    """Processes toxicology clinical notes for entity extraction and analysis"""
    
    def __init__(self, use_spacy: bool = True):
        """Initialize the text processor"""
        self.use_spacy = use_spacy
        self.nlp = None
        self.abbreviation_expander = AbbreviationExpander()
        
        if use_spacy:
            try:
                # Try to load clinical spacy model
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model for NER")
            except OSError:
                logger.warning("spaCy model not found, using rule-based extraction")
                self.use_spacy = False
        
        # Initialize substance patterns
        self._init_substance_patterns()
        self._init_symptom_patterns()
        self._init_treatment_patterns()
        self._init_route_patterns()
        self._init_dose_patterns()
    
    def _init_substance_patterns(self):
        """Initialize substance detection patterns"""
        self.substance_patterns = {}
        
        # Drug categories with common names and slang
        for category, substances in SUBSTANCE_CATEGORIES.items():
            for substance in substances:
                # Add variations and common misspellings
                variations = [
                    substance.lower(),
                    substance.title(),
                    substance.upper(),
                    substance.replace('-', ' '),
                    substance.replace(' ', ''),
                ]
                
                # Add common slang terms
                slang_map = {
                    'alprazolam': ['xanax', 'bars', 'xannies'],
                    'diazepam': ['valium', 'vals'],
                    'lorazepam': ['ativan'],
                    'clonazepam': ['klonopin', 'k-pins'],
                    'zolpidem': ['ambien'],
                    'acetaminophen': ['tylenol', 'paracetamol'],
                    'ibuprofen': ['advil', 'motrin'],
                    'aspirin': ['asa', 'acetylsalicylic acid'],
                    'cocaine': ['coke', 'blow', 'snow', 'crack'],
                    'methamphetamine': ['meth', 'crystal', 'ice', 'crank'],
                    'heroin': ['dope', 'smack', 'h'],
                    'fentanyl': ['fent', 'china white'],
                    'morphine': ['morph'],
                    'oxycodone': ['oxy', 'oxycontin', 'percocet'],
                    'hydrocodone': ['vicodin', 'norco'],
                    'codeine': ['lean', 'purple drank'],
                    'amphetamine': ['speed', 'adderall'],
                    'methylphenidate': ['ritalin', 'concerta'],
                    'fluoxetine': ['prozac'],
                    'sertraline': ['zoloft'],
                    'paroxetine': ['paxil'],
                    'citalopram': ['celexa'],
                    'escitalopram': ['lexapro'],
                    'bupropion': ['wellbutrin'],
                    'venlafaxine': ['effexor'],
                    'duloxetine': ['cymbalta'],
                    'olanzapine': ['zyprexa'],
                    'quetiapine': ['seroquel'],
                    'risperidone': ['risperdal'],
                    'aripiprazole': ['abilify'],
                    'ziprasidone': ['geodon'],
                    'haloperidol': ['haldol'],
                    'chlorpromazine': ['thorazine'],
                    'lithium': ['lithium carbonate'],
                    'divalproex': ['depakote'],
                    'carbamazepine': ['tegretol'],
                    'lamotrigine': ['lamictal'],
                    'topiramate': ['topamax'],
                    'levetiracetam': ['keppra'],
                    'phenytoin': ['dilantin'],
                    'phenobarbital': ['phenobarb'],
                    'naloxone': ['narcan'],
                    'flumazenil': ['romazicon'],
                    'acetylcysteine': ['mucomyst', 'n-acetylcysteine'],
                    'marijuana': ['weed', 'pot', 'cannabis', 'ganja', 'mary jane'],
                    'lsd': ['acid', 'lysergic acid diethylamide'],
                    'psilocybin': ['shrooms', 'mushrooms', 'magic mushrooms'],
                    'mdma': ['ecstasy', 'molly', 'e', 'x'],
                    'ketamine': ['k', 'special k', 'vitamin k'],
                    'pcp': ['angel dust', 'phencyclidine'],
                    'ghb': ['gamma-hydroxybutyrate', 'liquid ecstasy', 'g'],
                    'flunitrazepam': ['rohypnol', 'roofies', 'roofie'],
                    'synthetic cannabinoids': ['k2', 'spice', 'fake weed'],
                    'synthetic cathinones': ['bath salts', 'mephedrone', 'methylone']
                }
                
                if substance.lower() in slang_map:
                    variations.extend(slang_map[substance.lower()])
                
                for variation in variations:
                    if variation:
                        self.substance_patterns[variation.lower()] = {
                            'substance': substance,
                            'category': category,
                            'confidence': 0.9
                        }
        
        # Add regex patterns for substance detection
        self.substance_regex_patterns = [
            (r'\b(\d+)\s*(mg|mcg|g|ml|l)\s*(of\s+)?([a-zA-Z\s]+)\b', 'dose_substance'),
            (r'\b([a-zA-Z\s]+)\s*(\d+)\s*(mg|mcg|g|ml|l)\b', 'substance_dose'),
            (r'\b(bottle|vial|pill|tablet|capsule|tab|cap)\s+(of\s+)?([a-zA-Z\s]+)\b', 'container_substance'),
            (r'\b([a-zA-Z\s]+)\s+(bottle|vial|pill|tablet|capsule|tab|cap)\b', 'substance_container'),
        ]
    
    def _init_symptom_patterns(self):
        """Initialize symptom detection patterns"""
        self.symptom_patterns = {
            # Mental status symptoms
            'altered_mental_status': [
                'altered mental status', 'ams', 'confusion', 'disorientation',
                'delirium', 'agitation', 'combative', 'unresponsive',
                'lethargic', 'stuporous', 'comatose', 'unconscious'
            ],
            'seizure': [
                'seizure', 'convulsion', 'tonic-clonic', 'grand mal',
                'petit mal', 'absence', 'myoclonic', 'focal'
            ],
            'respiratory': [
                'shortness of breath', 'sob', 'dyspnea', 'respiratory distress',
                'apnea', 'hypoventilation', 'hyperventilation', 'tachypnea',
                'bradypnea', 'respiratory depression', 'respiratory arrest'
            ],
            'cardiovascular': [
                'chest pain', 'cp', 'palpitations', 'tachycardia',
                'bradycardia', 'arrhythmia', 'hypertension', 'hypotension',
                'cardiac arrest', 'myocardial infarction', 'mi', 'heart attack'
            ],
            'gastrointestinal': [
                'nausea', 'vomiting', 'n/v', 'diarrhea', 'abdominal pain',
                'abd pain', 'dysphagia', 'odynophagia', 'dyspepsia'
            ],
            'neurological': [
                'headache', 'migraine', 'dizziness', 'vertigo', 'syncope',
                'paresthesia', 'numbness', 'tingling', 'weakness', 'paralysis',
                'tremor', 'ataxia', 'dysarthria', 'aphasia'
            ],
            'ophthalmological': [
                'blurred vision', 'diplopia', 'photophobia', 'mydriasis',
                'miosis', 'nystagmus', 'ptosis', 'conjunctival injection'
            ],
            'dermatological': [
                'rash', 'urticaria', 'pruritus', 'erythema', 'cyanosis',
                'pallor', 'diaphoresis', 'sweating', 'flushing'
            ],
            'renal': [
                'oliguria', 'anuria', 'polyuria', 'hematuria', 'proteinuria',
                'acute kidney injury', 'aki', 'renal failure'
            ],
            'hepatic': [
                'jaundice', 'hepatomegaly', 'splenomegaly', 'hepatitis',
                'liver failure', 'hepatic encephalopathy'
            ]
        }
    
    def _init_treatment_patterns(self):
        """Initialize treatment detection patterns"""
        self.treatment_patterns = {
            'antidotes': [
                'naloxone', 'narcan', 'flumazenil', 'romazicon',
                'acetylcysteine', 'mucomyst', 'n-acetylcysteine',
                'atropine', 'pralidoxime', '2-pam', 'calcium gluconate',
                'sodium bicarbonate', 'bicarb', 'deferoxamine', 'desferal',
                'dimercaprol', 'bal', 'edetate calcium disodium', 'edta',
                'penicillamine', 'succimer', 'chemet'
            ],
            'decontamination': [
                'activated charcoal', 'charcoal', 'gastric lavage',
                'stomach pump', 'whole bowel irrigation', 'wbi',
                'cathartics', 'sorbitol', 'polyethylene glycol', 'peg'
            ],
            'supportive_care': [
                'intubation', 'mechanical ventilation', 'ventilator',
                'oxygen', 'o2', 'supplemental oxygen', 'cpap', 'bipap',
                'iv fluids', 'intravenous fluids', 'normal saline',
                'lactated ringers', 'lr', 'dextrose', 'd5w', 'd10w'
            ],
            'monitoring': [
                'cardiac monitor', 'telemetry', 'pulse oximetry',
                'blood pressure monitoring', 'bp monitoring',
                'temperature monitoring', 'temp monitoring',
                'neurological checks', 'neuro checks', 'gcs monitoring'
            ],
            'procedures': [
                'central line', 'central venous catheter', 'cvc',
                'arterial line', 'a-line', 'swan-ganz catheter',
                'pulmonary artery catheter', 'pac', 'dialysis',
                'hemodialysis', 'hd', 'cvvh', 'ecmo'
            ]
        }
    
    def _init_route_patterns(self):
        """Initialize route of exposure patterns"""
        self.route_patterns = {
            'oral': ['po', 'by mouth', 'orally', 'swallowed', 'ingested'],
            'intravenous': ['iv', 'intravenous', 'intravenously'],
            'intramuscular': ['im', 'intramuscular', 'intramuscularly'],
            'subcutaneous': ['sc', 'subcutaneous', 'subcutaneously'],
            'inhalation': ['inhaled', 'inhalation', 'smoked', 'snorted'],
            'dermal': ['dermal', 'topical', 'skin', 'transdermal'],
            'rectal': ['pr', 'rectal', 'per rectum'],
            'nasal': ['intranasal', 'nasal', 'snorted'],
            'ocular': ['ocular', 'eye', 'ophthalmic'],
            'aural': ['aural', 'ear', 'otic']
        }
    
    def _init_dose_patterns(self):
        """Initialize dose/quantity patterns"""
        self.dose_patterns = [
            (r'\b(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|l|tablets?|tabs?|capsules?|caps?)\b', 'quantity_unit'),
            (r'\b(\d+(?:\.\d+)?)\s*(milligrams?|micrograms?|grams?|milliliters?|liters?)\b', 'quantity_unit_full'),
            (r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+(tablets?|tabs?|capsules?|caps?|pills?)\b', 'word_quantity'),
            (r'\b(\d+)\s*(bottles?|vials?|packs?|bags?)\b', 'container_quantity'),
            (r'\b(empty|full|half|quarter)\s+(bottle|vial|pack|bag)\b', 'descriptive_quantity'),
            (r'\b(\d+(?:\.\d+)?)\s*(times?|doses?|administrations?)\b', 'frequency'),
            (r'\b(\d+(?:\.\d+)?)\s*(hours?|hrs?|minutes?|mins?|days?)\b', 'time_quantity')
        ]
    
    def process_note(self, text: str) -> ToxicologyNote:
        """Process a toxicology note and extract entities"""
        if not text:
            return ToxicologyNote(
                text="",
                expanded_text="",
                entities=[],
                symptoms=[],
                vitals=VitalSigns()
            )
        
        # Expand abbreviations
        expanded_text = self.abbreviation_expander.expand_text(text)
        
        # Extract entities
        entities = self.extract_entities(expanded_text)
        
        # Extract specific information
        substance = self.extract_substance(expanded_text)
        route = self.extract_route(expanded_text)
        dose = self.extract_dose(expanded_text)
        symptoms = self.extract_symptoms(expanded_text)
        vitals = self.extract_vital_signs(expanded_text)
        chief_complaint = self.extract_chief_complaint(expanded_text)
        timeline = self.extract_timeline(expanded_text)
        
        # Calculate severity score
        severity_score = self.calculate_severity_score(expanded_text, symptoms, vitals)
        
        return ToxicologyNote(
            text=text,
            expanded_text=expanded_text,
            chief_complaint=chief_complaint,
            substance=substance,
            route=route,
            dose=dose,
            timeline=timeline,
            symptoms=symptoms,
            vitals=vitals,
            entities=entities,
            severity_score=severity_score
        )
    
    def extract_entities(self, text: str) -> List[ToxicologyEntity]:
        """Extract clinical entities from text"""
        entities = []
        
        # Extract substances
        substance_entities = self._extract_substance_entities(text)
        entities.extend(substance_entities)
        
        # Extract symptoms
        symptom_entities = self._extract_symptom_entities(text)
        entities.extend(symptom_entities)
        
        # Extract treatments
        treatment_entities = self._extract_treatment_entities(text)
        entities.extend(treatment_entities)
        
        # Extract vital signs
        vital_entities = self._extract_vital_entities(text)
        entities.extend(vital_entities)
        
        # Extract routes
        route_entities = self._extract_route_entities(text)
        entities.extend(route_entities)
        
        # Extract doses
        dose_entities = self._extract_dose_entities(text)
        entities.extend(dose_entities)
        
        # Use spaCy for additional NER if available
        if self.use_spacy and self.nlp:
            spacy_entities = self._extract_spacy_entities(text)
            entities.extend(spacy_entities)
        
        return entities
    
    def _extract_substance_entities(self, text: str) -> List[ToxicologyEntity]:
        """Extract substance entities"""
        entities = []
        text_lower = text.lower()
        
        for substance, info in self.substance_patterns.items():
            if substance in text_lower:
                # Find all occurrences
                pattern = rf'\b{re.escape(substance)}\b'
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    entities.append(ToxicologyEntity(
                        text=match.group(),
                        label="SUBSTANCE",
                        start=match.start(),
                        end=match.end(),
                        confidence=info['confidence']
                    ))
        
        return entities
    
    def _extract_symptom_entities(self, text: str) -> List[ToxicologyEntity]:
        """Extract symptom entities"""
        entities = []
        text_lower = text.lower()
        
        for category, symptoms in self.symptom_patterns.items():
            for symptom in symptoms:
                if symptom in text_lower:
                    pattern = rf'\b{re.escape(symptom)}\b'
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    
                    for match in matches:
                        entities.append(ToxicologyEntity(
                            text=match.group(),
                            label="SYMPTOM",
                            start=match.start(),
                            end=match.end(),
                            confidence=0.8
                        ))
        
        return entities
    
    def _extract_treatment_entities(self, text: str) -> List[ToxicologyEntity]:
        """Extract treatment entities"""
        entities = []
        text_lower = text.lower()
        
        for category, treatments in self.treatment_patterns.items():
            for treatment in treatments:
                if treatment in text_lower:
                    pattern = rf'\b{re.escape(treatment)}\b'
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    
                    for match in matches:
                        entities.append(ToxicologyEntity(
                            text=match.group(),
                            label="TREATMENT",
                            start=match.start(),
                            end=match.end(),
                            confidence=0.8
                        ))
        
        return entities
    
    def _extract_vital_entities(self, text: str) -> List[ToxicologyEntity]:
        """Extract vital sign entities"""
        entities = []
        
        for vital_type, pattern in VITAL_SIGNS_PATTERNS.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                entities.append(ToxicologyEntity(
                    text=match.group(),
                    label="VITAL_SIGN",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9
                ))
        
        return entities
    
    def _extract_route_entities(self, text: str) -> List[ToxicologyEntity]:
        """Extract route of exposure entities"""
        entities = []
        text_lower = text.lower()
        
        for route_type, routes in self.route_patterns.items():
            for route in routes:
                if route in text_lower:
                    pattern = rf'\b{re.escape(route)}\b'
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    
                    for match in matches:
                        entities.append(ToxicologyEntity(
                            text=match.group(),
                            label="ROUTE",
                            start=match.start(),
                            end=match.end(),
                            confidence=0.8
                        ))
        
        return entities
    
    def _extract_dose_entities(self, text: str) -> List[ToxicologyEntity]:
        """Extract dose/quantity entities"""
        entities = []
        
        for pattern, label in self.dose_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                entities.append(ToxicologyEntity(
                    text=match.group(),
                    label="DOSE",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.7
                ))
        
        return entities
    
    def _extract_spacy_entities(self, text: str) -> List[ToxicologyEntity]:
        """Extract entities using spaCy"""
        entities = []
        doc = self.nlp(text)
        
        for ent in doc.ents:
            # Map spaCy entity types to our labels
            label_map = {
                'PERSON': 'PERSON',
                'ORG': 'ORGANIZATION',
                'GPE': 'LOCATION',
                'DATE': 'TIMELINE',
                'TIME': 'TIMELINE',
                'QUANTITY': 'DOSE',
                'CARDINAL': 'DOSE'
            }
            
            label = label_map.get(ent.label_, ent.label_)
            
            entities.append(ToxicologyEntity(
                text=ent.text,
                label=label,
                start=ent.start_char,
                end=ent.end_char,
                confidence=0.6
            ))
        
        return entities
    
    def extract_substance(self, text: str) -> Optional[str]:
        """Extract primary substance from text"""
        text_lower = text.lower()
        
        # Find substances with highest confidence
        found_substances = []
        for substance, info in self.substance_patterns.items():
            if substance in text_lower:
                found_substances.append((substance, info['confidence']))
        
        if found_substances:
            # Return the substance with highest confidence
            found_substances.sort(key=lambda x: x[1], reverse=True)
            return found_substances[0][0]
        
        return None
    
    def extract_route(self, text: str) -> Optional[str]:
        """Extract route of exposure"""
        text_lower = text.lower()
        
        for route_type, routes in self.route_patterns.items():
            for route in routes:
                if route in text_lower:
                    return route_type
        
        return None
    
    def extract_dose(self, text: str) -> Optional[str]:
        """Extract dose/quantity information"""
        dose_matches = []
        
        for pattern, label in self.dose_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                dose_matches.append(match.group())
        
        if dose_matches:
            return '; '.join(dose_matches)
        
        return None
    
    def extract_symptoms(self, text: str) -> List[str]:
        """Extract symptoms from text"""
        symptoms = []
        text_lower = text.lower()
        
        for category, symptom_list in self.symptom_patterns.items():
            for symptom in symptom_list:
                if symptom in text_lower:
                    symptoms.append(symptom)
        
        return list(set(symptoms))  # Remove duplicates
    
    def extract_vital_signs(self, text: str) -> VitalSigns:
        """Extract vital signs from text"""
        vitals = VitalSigns()
        
        # Extract GCS
        gcs_match = re.search(VITAL_SIGNS_PATTERNS['GCS'], text, re.IGNORECASE)
        if gcs_match:
            vitals.gcs = gcs_match.group(1)
        
        # Extract BP
        bp_match = re.search(VITAL_SIGNS_PATTERNS['BP'], text, re.IGNORECASE)
        if bp_match:
            vitals.blood_pressure = bp_match.group(1)
        
        # Extract HR
        hr_match = re.search(VITAL_SIGNS_PATTERNS['HR'], text, re.IGNORECASE)
        if hr_match:
            vitals.heart_rate = hr_match.group(1)
        
        # Extract RR
        rr_match = re.search(VITAL_SIGNS_PATTERNS['RR'], text, re.IGNORECASE)
        if rr_match:
            vitals.respiratory_rate = rr_match.group(1)
        
        # Extract temperature
        temp_match = re.search(VITAL_SIGNS_PATTERNS['TEMP'], text, re.IGNORECASE)
        if temp_match:
            vitals.temperature = temp_match.group(1)
        
        # Extract O2 saturation
        o2_match = re.search(VITAL_SIGNS_PATTERNS['O2_SAT'], text, re.IGNORECASE)
        if o2_match:
            vitals.oxygen_saturation = o2_match.group(1)
        
        return vitals
    
    def extract_chief_complaint(self, text: str) -> Optional[str]:
        """Extract chief complaint"""
        # Look for common chief complaint patterns
        cc_patterns = [
            r'cc\s*[:=]\s*([^.\n]+)',
            r'chief\s+complaint\s*[:=]\s*([^.\n]+)',
            r'presenting\s+problem\s*[:=]\s*([^.\n]+)'
        ]
        
        for pattern in cc_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def extract_timeline(self, text: str) -> Optional[str]:
        """Extract timeline information"""
        timeline_patterns = [
            r'(\d+\s+(?:hours?|hrs?|minutes?|mins?|days?)\s+(?:ago|before|prior))',
            r'(found\s+\d+\s+(?:hours?|hrs?|minutes?|mins?|days?)\s+(?:ago|before|prior))',
            r'(last\s+(?:seen|found)\s+\d+\s+(?:hours?|hrs?|minutes?|mins?|days?)\s+(?:ago|before|prior))'
        ]
        
        for pattern in timeline_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def calculate_severity_score(self, text: str, symptoms: List[str], vitals: VitalSigns) -> float:
        """Calculate clinical severity score (0-10)"""
        score = 0.0
        
        # Base score from symptoms
        severe_symptoms = [
            'unresponsive', 'comatose', 'cardiac arrest', 'respiratory arrest',
            'seizure', 'convulsion', 'myocardial infarction', 'heart attack'
        ]
        
        moderate_symptoms = [
            'altered mental status', 'ams', 'confusion', 'delirium',
            'respiratory distress', 'dyspnea', 'shortness of breath',
            'chest pain', 'hypertension', 'hypotension', 'tachycardia',
            'bradycardia', 'arrhythmia'
        ]
        
        mild_symptoms = [
            'nausea', 'vomiting', 'dizziness', 'headache', 'rash',
            'pruritus', 'diaphoresis', 'sweating'
        ]
        
        for symptom in symptoms:
            if symptom in severe_symptoms:
                score += 3.0
            elif symptom in moderate_symptoms:
                score += 2.0
            elif symptom in mild_symptoms:
                score += 1.0
        
        # Vital signs scoring
        if vitals.gcs:
            try:
                gcs_value = int(vitals.gcs)
                if gcs_value <= 8:
                    score += 3.0
                elif gcs_value <= 12:
                    score += 2.0
                elif gcs_value <= 14:
                    score += 1.0
            except ValueError:
                pass
        
        if vitals.heart_rate:
            try:
                hr_value = int(vitals.heart_rate)
                if hr_value < 50 or hr_value > 120:
                    score += 1.5
            except ValueError:
                pass
        
        if vitals.respiratory_rate:
            try:
                rr_value = int(vitals.respiratory_rate)
                if rr_value < 12 or rr_value > 20:
                    score += 1.5
            except ValueError:
                pass
        
        # Cap score at 10
        return min(score, 10.0) 