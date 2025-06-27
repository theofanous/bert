import torch
from transformers import BertTokenizer, BertModel
import re
from typing import List, Dict, Tuple
import string

class AxiomExtractor:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        
        # common english stop words for filtering
        self.stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'shall', 'of', 'in', 'on',
            'at', 'to', 'for', 'with', 'by', 'from', 'about', 'as', 'or', 'and'
        }
        
        # Common nouns that might appear in logical statements
        self.common_nouns = {
            'mammal', 'animal', 'person', 'human', 'dog', 'cat', 'bird', 'fish'
        }
        
        # pattern matching rules for different axiom types
        self.patterns = {
            'subclass': [
                r'(\w+)\s+is\s+a\s+(\w+)',
                r'(\w+)\s+are\s+(\w+)',
                r'(\w+)\s+is\s+an?\s+(\w+)'
            ],
            'complex_definition': [
                r'A?\s*(\w+)\s+is\s+a\s+(\w+)\s+which\s+(.*)',
                r'(\w+)\s+are\s+(\w+)\s+that\s+(.*)'
            ]
        }
        
        # property extraction patterns
        self.property_patterns = {
            'sound': r'(barks?|meows?|roars?)',
            'legs': r'has\s+(\d+)\s+legs?',
            'action': r'(runs?|flies?|walks?)'
        }
    
    def encode_sentence(self, sentence: str) -> torch.Tensor:
        #encode sentence using BERT
        inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)

        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)
    
    def extract_entities(self, sentence: str) -> Dict[str, List[str]]:
        #extract entities using BERT tokenizer and basic NLP rules
        tokens = self.tokenizer.tokenize(sentence.lower())
        
        entities = {
            'nouns': [],
            'numbers': [],
            'actions': []
        }
        
        # clean sentence for basic processing
        clean_sentence = sentence.lower().translate(str.maketrans('', '', string.punctuation))
        words = clean_sentence.split()
        
        # extract numbers
        for word in words:
            if word.isdigit():
                entities['numbers'].append(word)
        
        # extract potential nouns (basic heuristics)
        for word in words:
            if (word not in self.stop_words and 
                len(word) > 2 and 
                word.isalpha() and
                not word.endswith('ing') and  # skips words ending in -ing
                not word.endswith('ed')):     # skips words ending in -ed
                entities['nouns'].append(word)
        
        # extract action verbs (basic patterns)
        action_patterns = ['bark', 'meow', 'walk']
        for word in words:
            for pattern in action_patterns:
                if pattern in word:
                    entities['actions'].append(pattern)
        
        return entities
    
    def match_subclass_pattern(self, sentence: str) -> Tuple[str, str, str]:
        sentence_lower = sentence.lower().strip()
        
        for pattern in self.patterns['subclass']:
            match = re.search(pattern, sentence_lower)
            if match:
                subclass = match.group(1).capitalize()
                superclass = match.group(2).capitalize()
                axiom = f"{subclass} SubClassOf {superclass}"
                return subclass, superclass, axiom
        
        return None, None, None
    
    def extract_properties(self, description: str) -> Dict[str, str]:
        properties = {}
        
        # extract sound properties
        sound_match = re.search(self.property_patterns['sound'], description.lower())
        if sound_match:
            sound = sound_match.group(1)
            if sound.startswith('bark'):
                properties['sound'] = 'Bark'
            elif sound.startswith('meow'):
                properties['sound'] = 'Meow'
            # add more sound mappings as needed
        
        # extract leg count
        legs_match = re.search(self.property_patterns['legs'], description.lower())
        if legs_match:
            leg_count = legs_match.group(1)
            properties['legs'] = f"has_{{={leg_count}}} Legs"
        
        return properties
    
    def match_complex_definition(self, sentence: str) -> str:
        sentence_lower = sentence.lower().strip()
        
        for pattern in self.patterns['complex_definition']:
            match = re.search(pattern, sentence_lower)
            if match:
                entity = match.group(1).capitalize()
                base_class = match.group(2).capitalize()
                description = match.group(3)
                
                properties = self.extract_properties(description)
                
                # building axiom
                axiom_parts = [base_class]
                
                if 'sound' in properties:
                    axiom_parts.append(f"makesSound some {properties['sound']}")
                
                if 'legs' in properties:
                    axiom_parts.append(properties['legs'])
                
                if len(axiom_parts) > 1:
                    axiom = f"{' and '.join(axiom_parts)} = {entity}"
                else:
                    axiom = f"{entity} SubClassOf {base_class}"
                
                return axiom
        
        return None
    
    def sentence_to_axiom(self, sentence: str) -> str:
        sentence = sentence.strip()
        
        # simple subclass pattern first
        subclass, superclass, axiom = self.match_subclass_pattern(sentence)
        if axiom:
            return axiom
        
        # complex definition pattern
        complex_axiom = self.match_complex_definition(sentence)
        if complex_axiom:
            return complex_axiom
        
        # If no pattern matches, use BERT embeddings for similarity matching
        return self.fallback_axiom_extraction(sentence)
    
    def fallback_axiom_extraction(self, sentence: str) -> str:
        entities = self.extract_entities(sentence)
        
        if len(entities['nouns']) >= 2:
            #first noun is subclass, second is superclass
            subclass = entities['nouns'][0].capitalize()
            superclass = entities['nouns'][1].capitalize()
            return f"{subclass} SubClassOf {superclass}"
        
        return f"Unknown axiom pattern for: {sentence}"
    
    def process_sentences(self, sentences: List[str]) -> List[str]:
        axioms = []
        
        print("LOADING . . .")
        
        for i, sentence in enumerate(sentences, 1):
            print(f"\nSentence {i}: {sentence}")
            
            # Get BERT embedding (for potential future use)
            embedding = self.encode_sentence(sentence)
            
            # extract axiom
            axiom = self.sentence_to_axiom(sentence)
            axioms.append(axiom)
            
            print(f"Axiom {i}: {axiom}")
        
        return axioms

def main():
    extractor = AxiomExtractor()
    
    # Input sentences
    sentences = [
        "Dog is a mammal",
        "A person is a human", 
        "A dog is a mammal which barks and has 4 legs"
    ]
    
    axioms = extractor.process_sentences(sentences)
    
    print("\n")
    print("SUMMARY - Generated Axioms:")
    
    for i, (sentence, axiom) in enumerate(zip(sentences, axioms), 1):
        print(f"{i}. '{sentence}' → {axiom}")
    
    
    expected_axioms = [
        "Dog SubClassOf Mammal",
        "Person SubClassOf Human", 
        "Mammal and makesSound some Bark and has_{=4} Legs = Dog"
    ]
    
    print("\n")
    print("EXPECTED vs GENERATED:")
    
    for i, (expected, generated) in enumerate(zip(expected_axioms, axioms), 1):
        match = "CORRECT" if expected == generated else "WRONG"
        print(f"{i}. {match} Expected: {expected}")
        print(f"   Generated: {generated}")
        print()

if __name__ == "__main__":
    try:
        import transformers
        import torch
    except ImportError as e:
        print("Missing dependencies. Please install:")
        print("pip install transformers torch")
        exit(1)
    
    main()