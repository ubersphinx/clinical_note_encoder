"""Medical abbreviation expansion for toxicology notes"""
import re
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from config.settings import DATA_FILES
from models.data_models import AbbreviationMapping

logger = logging.getLogger(__name__)

class AbbreviationExpander:
    """Handles medical abbreviation expansion for toxicology notes"""
    
    def __init__(self, abbreviation_file: Optional[Path] = None):
        """Initialize the abbreviation expander"""
        self.abbreviation_file = abbreviation_file or DATA_FILES["abbreviations"]
        self.abbreviation_dict = {}
        self.context_patterns = {}
        self.load_abbreviations()
        
    def load_abbreviations(self):
        """Load abbreviation dictionary from CSV file"""
        try:
            if self.abbreviation_file.exists():
                df = pd.read_csv(self.abbreviation_file)
                for _, row in df.iterrows():
                    abbr = row.get('abbreviation', '').strip().lower()
                    expansion = row.get('expansion', '').strip()
                    context = row.get('context', '').strip()
                    
                    if abbr and expansion:
                        self.abbreviation_dict[abbr] = {
                            'expansion': expansion,
                            'context': context if context else None
                        }
                        
                        # Create context patterns for context-aware expansion
                        if context:
                            pattern = re.compile(rf'\b{re.escape(abbr)}\b(?=.*{re.escape(context)})', re.IGNORECASE)
                            self.context_patterns[abbr] = pattern
                            
                logger.info(f"Loaded {len(self.abbreviation_dict)} abbreviations")
            else:
                logger.warning(f"Abbreviation file not found: {self.abbreviation_file}")
                self._load_default_abbreviations()
                
        except Exception as e:
            logger.error(f"Error loading abbreviations: {e}")
            self._load_default_abbreviations()
    
    def _load_default_abbreviations(self):
        """Load default toxicology abbreviations"""
        default_abbreviations = {
            'cc': 'chief complaint',
            'hpi': 'history of present illness',
            'pt': 'patient',
            'tx': 'treatment',
            'rx': 'prescription',
            'po': 'by mouth',
            'iv': 'intravenous',
            'im': 'intramuscular',
            'sc': 'subcutaneous',
            'pr': 'per rectum',
            'inh': 'inhalation',
            'derm': 'dermal',
            'gcs': 'glasgow coma scale',
            'bp': 'blood pressure',
            'hr': 'heart rate',
            'rr': 'respiratory rate',
            'temp': 'temperature',
            'o2': 'oxygen',
            'sat': 'saturation',
            'ams': 'altered mental status',
            'loc': 'loss of consciousness',
            'sob': 'shortness of breath',
            'cp': 'chest pain',
            'n/v': 'nausea and vomiting',
            'n/v/d': 'nausea, vomiting, and diarrhea',
            'abd': 'abdomen',
            'c/o': 'complains of',
            'w/': 'with',
            'w/o': 'without',
            'h/o': 'history of',
            'fh': 'family history',
            'sh': 'social history',
            'pmh': 'past medical history',
            'meds': 'medications',
            'tab': 'tablet',
            'tabs': 'tablets',
            'cap': 'capsule',
            'caps': 'capsules',
            'mg': 'milligrams',
            'mcg': 'micrograms',
            'g': 'grams',
            'ml': 'milliliters',
            'l': 'liters',
            'hr': 'hour',
            'hrs': 'hours',
            'min': 'minute',
            'mins': 'minutes',
            'sec': 'second',
            'secs': 'seconds',
            'qd': 'once daily',
            'bid': 'twice daily',
            'tid': 'three times daily',
            'qid': 'four times daily',
            'qhs': 'at bedtime',
            'prn': 'as needed',
            'stat': 'immediately',
            'asap': 'as soon as possible',
            'etoh': 'alcohol',
            'cocaine': 'cocaine',
            'heroin': 'heroin',
            'meth': 'methamphetamine',
            'weed': 'marijuana',
            'pot': 'marijuana',
            'lsd': 'lysergic acid diethylamide',
            'mdma': '3,4-methylenedioxymethamphetamine',
            'ecstasy': '3,4-methylenedioxymethamphetamine',
            'molly': '3,4-methylenedioxymethamphetamine',
            'xanax': 'alprazolam',
            'valium': 'diazepam',
            'ativan': 'lorazepam',
            'klonopin': 'clonazepam',
            'ambien': 'zolpidem',
            'tylenol': 'acetaminophen',
            'advil': 'ibuprofen',
            'motrin': 'ibuprofen',
            'aspirin': 'acetylsalicylic acid',
            'vicodin': 'hydrocodone/acetaminophen',
            'percocet': 'oxycodone/acetaminophen',
            'oxy': 'oxycodone',
            'oxycontin': 'oxycodone',
            'fentanyl': 'fentanyl',
            'morphine': 'morphine',
            'codeine': 'codeine',
            'adderall': 'amphetamine/dextroamphetamine',
            'ritalin': 'methylphenidate',
            'prozac': 'fluoxetine',
            'zoloft': 'sertraline',
            'paxil': 'paroxetine',
            'celexa': 'citalopram',
            'lexapro': 'escitalopram',
            'wellbutrin': 'bupropion',
            'effexor': 'venlafaxine',
            'cymbalta': 'duloxetine',
            'zyprexa': 'olanzapine',
            'seroquel': 'quetiapine',
            'risperdal': 'risperidone',
            'abilify': 'aripiprazole',
            'geodon': 'ziprasidone',
            'haldol': 'haloperidol',
            'thorazine': 'chlorpromazine',
            'lithium': 'lithium carbonate',
            'depakote': 'divalproex sodium',
            'tegretol': 'carbamazepine',
            'lamictal': 'lamotrigine',
            'topamax': 'topiramate',
            'keppra': 'levetiracetam',
            'dilantin': 'phenytoin',
            'phenobarb': 'phenobarbital',
            'ativan': 'lorazepam',
            'klonopin': 'clonazepam',
            'xanax': 'alprazolam',
            'valium': 'diazepam',
            'librium': 'chlordiazepoxide',
            'restoril': 'temazepam',
            'dalmane': 'flurazepam',
            'halcion': 'triazolam',
            'versed': 'midazolam',
            'rohypnol': 'flunitrazepam',
            'roofies': 'flunitrazepam',
            'ghb': 'gamma-hydroxybutyrate',
            'ketamine': 'ketamine',
            'pcp': 'phencyclidine',
            'angel dust': 'phencyclidine',
            'mescaline': 'mescaline',
            'psilocybin': 'psilocybin',
            'mushrooms': 'psilocybin',
            'dmt': 'dimethyltryptamine',
            'ayahuasca': 'dimethyltryptamine',
            'salvia': 'salvia divinorum',
            'k2': 'synthetic cannabinoids',
            'spice': 'synthetic cannabinoids',
            'bath salts': 'synthetic cathinones',
            'mephedrone': '4-methylmethcathinone',
            'methylone': '3,4-methylenedioxy-N-methylcathinone',
            'molly': '3,4-methylenedioxymethamphetamine',
            'ecstasy': '3,4-methylenedioxymethamphetamine',
            'mdma': '3,4-methylenedioxymethamphetamine',
            'crystal': 'methamphetamine',
            'ice': 'methamphetamine',
            'crank': 'methamphetamine',
            'speed': 'amphetamine',
            'dexedrine': 'dextroamphetamine',
            'adderall': 'amphetamine/dextroamphetamine',
            'ritalin': 'methylphenidate',
            'concerta': 'methylphenidate',
            'focalin': 'dexmethylphenidate',
            'strattera': 'atomoxetine',
            'intuniv': 'guanfacine',
            'kapvay': 'clonidine',
            'tenex': 'guanfacine',
            'catapres': 'clonidine',
            'narcan': 'naloxone',
            'naloxone': 'naloxone',
            'flumazenil': 'flumazenil',
            'romazicon': 'flumazenil',
            'acetylcysteine': 'N-acetylcysteine',
            'mucomyst': 'N-acetylcysteine',
            'acetylcysteine': 'N-acetylcysteine',
            'charcoal': 'activated charcoal',
            'gastric lavage': 'gastric lavage',
            'pumping stomach': 'gastric lavage',
            'stomach pump': 'gastric lavage',
            'dialysis': 'hemodialysis',
            'hd': 'hemodialysis',
            'cvvh': 'continuous veno-venous hemofiltration',
            'ecmo': 'extracorporeal membrane oxygenation',
            'ventilator': 'mechanical ventilation',
            'vent': 'mechanical ventilation',
            'intubation': 'endotracheal intubation',
            'ett': 'endotracheal tube',
            'trach': 'tracheostomy',
            'tracheostomy': 'tracheostomy',
            'central line': 'central venous catheter',
            'cvp': 'central venous pressure',
            'art line': 'arterial line',
            'a-line': 'arterial line',
            'swan': 'swan-ganz catheter',
            'pulmonary artery catheter': 'swan-ganz catheter',
            'pac': 'pulmonary artery catheter',
            'icp': 'intracranial pressure',
            'cpap': 'continuous positive airway pressure',
            'bipap': 'bilevel positive airway pressure',
            'peep': 'positive end-expiratory pressure',
            'tidal volume': 'tidal volume',
            'tv': 'tidal volume',
            'minute ventilation': 'minute ventilation',
            'mv': 'minute ventilation',
            'fio2': 'fraction of inspired oxygen',
            'spo2': 'oxygen saturation',
            'sa02': 'oxygen saturation',
            'pao2': 'partial pressure of oxygen',
            'paco2': 'partial pressure of carbon dioxide',
            'ph': 'hydrogen ion concentration',
            'hco3': 'bicarbonate',
            'bicarb': 'bicarbonate',
            'base excess': 'base excess',
            'be': 'base excess',
            'anion gap': 'anion gap',
            'ag': 'anion gap',
            'lactate': 'lactate',
            'lactic acid': 'lactate',
            'creatinine': 'creatinine',
            'cr': 'creatinine',
            'bun': 'blood urea nitrogen',
            'urea': 'blood urea nitrogen',
            'glucose': 'glucose',
            'glu': 'glucose',
            'sodium': 'sodium',
            'na': 'sodium',
            'potassium': 'potassium',
            'k': 'potassium',
            'chloride': 'chloride',
            'cl': 'chloride',
            'calcium': 'calcium',
            'ca': 'calcium',
            'magnesium': 'magnesium',
            'mg': 'magnesium',
            'phosphate': 'phosphate',
            'po4': 'phosphate',
            'albumin': 'albumin',
            'alb': 'albumin',
            'total protein': 'total protein',
            'tp': 'total protein',
            'bilirubin': 'bilirubin',
            'bili': 'bilirubin',
            'alt': 'alanine aminotransferase',
            'sgot': 'aspartate aminotransferase',
            'ast': 'aspartate aminotransferase',
            'alk phos': 'alkaline phosphatase',
            'alp': 'alkaline phosphatase',
            'ggt': 'gamma-glutamyl transferase',
            'ggtp': 'gamma-glutamyl transferase',
            'pt': 'prothrombin time',
            'inr': 'international normalized ratio',
            'aptt': 'activated partial thromboplastin time',
            'ptt': 'partial thromboplastin time',
            'fibrinogen': 'fibrinogen',
            'd-dimer': 'D-dimer',
            'ddimer': 'D-dimer',
            'troponin': 'troponin',
            'trop': 'troponin',
            'ck': 'creatine kinase',
            'cpk': 'creatine phosphokinase',
            'mb': 'creatine kinase-MB',
            'ck-mb': 'creatine kinase-MB',
            'bnp': 'brain natriuretic peptide',
            'nt-probnp': 'N-terminal pro-brain natriuretic peptide',
            'cbc': 'complete blood count',
            'wbc': 'white blood cell count',
            'rbc': 'red blood cell count',
            'hgb': 'hemoglobin',
            'hct': 'hematocrit',
            'plt': 'platelet count',
            'platelets': 'platelet count',
            'diff': 'differential',
            'seg': 'segmented neutrophils',
            'bands': 'band neutrophils',
            'lymph': 'lymphocytes',
            'mono': 'monocytes',
            'eos': 'eosinophils',
            'baso': 'basophils',
            'urinalysis': 'urinalysis',
            'ua': 'urinalysis',
            'urine': 'urinalysis',
            'protein': 'protein',
            'glucose': 'glucose',
            'ketones': 'ketones',
            'blood': 'blood',
            'leukocytes': 'leukocytes',
            'nitrites': 'nitrites',
            'specific gravity': 'specific gravity',
            'sg': 'specific gravity',
            'ph': 'hydrogen ion concentration',
            'culture': 'culture',
            'sensitivity': 'sensitivity',
            'c&s': 'culture and sensitivity',
            'gram stain': 'gram stain',
            'afb': 'acid-fast bacilli',
            'pcr': 'polymerase chain reaction',
            'rapid test': 'rapid test',
            'drug screen': 'drug screen',
            'tox screen': 'toxicology screen',
            'urine drug screen': 'urine drug screen',
            'uds': 'urine drug screen',
            'blood alcohol': 'blood alcohol',
            'bac': 'blood alcohol concentration',
            'breath alcohol': 'breath alcohol',
            'brac': 'breath alcohol concentration',
            'saliva test': 'saliva test',
            'hair test': 'hair test',
            'sweat patch': 'sweat patch',
            'meconium': 'meconium',
            'amniotic fluid': 'amniotic fluid',
            'placenta': 'placenta',
            'umbilical cord': 'umbilical cord',
            'cord blood': 'umbilical cord blood',
            'meconium': 'meconium',
            'amniotic fluid': 'amniotic fluid',
            'placenta': 'placenta',
            'umbilical cord': 'umbilical cord',
            'cord blood': 'umbilical cord blood'
        }
        
        for abbr, expansion in default_abbreviations.items():
            self.abbreviation_dict[abbr] = {
                'expansion': expansion,
                'context': None
            }
        
        logger.info(f"Loaded {len(self.abbreviation_dict)} default abbreviations")
    
    def expand_text(self, text: str, context_aware: bool = True) -> str:
        """Expand abbreviations in text"""
        if not text:
            return text
            
        expanded_text = text
        
        # Sort abbreviations by length (longest first) to avoid partial matches
        sorted_abbreviations = sorted(
            self.abbreviation_dict.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        
        for abbr, info in sorted_abbreviations:
            expansion = info['expansion']
            context = info.get('context')
            
            if context_aware and context:
                # Context-aware replacement
                pattern = rf'\b{re.escape(abbr)}\b(?=.*{re.escape(context)})'
                expanded_text = re.sub(pattern, expansion, expanded_text, flags=re.IGNORECASE)
            else:
                # Simple word boundary replacement
                pattern = rf'\b{re.escape(abbr)}\b'
                expanded_text = re.sub(pattern, expansion, expanded_text, flags=re.IGNORECASE)
        
        return expanded_text
    
    def get_expansion_preview(self, text: str) -> List[Tuple[str, str, str]]:
        """Get preview of abbreviations that would be expanded"""
        previews = []
        
        for abbr, info in self.abbreviation_dict.items():
            pattern = rf'\b{re.escape(abbr)}\b'
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                previews.append((
                    abbr,
                    info['expansion'],
                    f"Position {match.start()}-{match.end()}"
                ))
        
        return previews
    
    def add_abbreviation(self, abbreviation: str, expansion: str, context: Optional[str] = None):
        """Add a new abbreviation to the dictionary"""
        abbr = abbreviation.strip().lower()
        exp = expansion.strip()
        
        if abbr and exp:
            self.abbreviation_dict[abbr] = {
                'expansion': exp,
                'context': context
            }
            logger.info(f"Added abbreviation: {abbr} -> {exp}")
    
    def save_abbreviations(self):
        """Save abbreviations to CSV file"""
        try:
            rows = []
            for abbr, info in self.abbreviation_dict.items():
                rows.append({
                    'abbreviation': abbr,
                    'expansion': info['expansion'],
                    'context': info.get('context', '')
                })
            
            df = pd.DataFrame(rows)
            df.to_csv(self.abbreviation_file, index=False)
            logger.info(f"Saved {len(rows)} abbreviations to {self.abbreviation_file}")
            
        except Exception as e:
            logger.error(f"Error saving abbreviations: {e}")
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about loaded abbreviations"""
        return {
            'total_abbreviations': len(self.abbreviation_dict),
            'with_context': len([info for info in self.abbreviation_dict.values() if info.get('context')]),
            'without_context': len([info for info in self.abbreviation_dict.values() if not info.get('context')])
        } 