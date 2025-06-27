import re
from transformers import BertTokenizer, BertModel
import torch

# loads pre-trained model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

def extract_axiom(sentence: str) -> str:
    sentence = sentence.strip().lower()

    # rule 1 "<X> is a <Y>" → X SubClassOf Y
    match = re.match(r"(a |an |the )?(?P<X>\w+)\s+is\s+(a |an |the )?(?P<Y>\w+)", sentence)
    if match:
        X, Y = match.group("X").capitalize(), match.group("Y").capitalize()
        return f"{X} SubClassOf {Y}"

    # rule 2 "<X> is a <Y> which <...>" → X SubClassOf Y and <R>
    match = re.match(r"(a |an |the )?(?P<X>\w+)\s+is\s+(a |an |the )?(?P<Y>\w+)\s+which\s+(?P<props>.+)", sentence)
    if match:
        X, Y, props = match.group("X").capitalize(), match.group("Y").capitalize(), match.group("props")
        prop_axioms = []

        if "bark" in props or "barks" in props:
            prop_axioms.append("makesSound some Bark")
        if re.search(r"has\s+4\s+legs?", props):
            prop_axioms.append("has_{=4} Legs")

        prop_axiom = " and ".join(prop_axioms)
        return f"{X} SubClassOf {Y} and {prop_axiom}"

    return f"# Could not work out: {sentence}"

if __name__ == "__main__":

    #sentences to be used as data set
    sentences = [
        "Dog is a mammal",
        "A person is a human",
        "A dog is a mammal which barks and has 4 legs"
    ]

    print("Ontology Axioms Created:\n")
    for sentence in sentences:
        axiom = extract_axiom(sentence)
        print(f"{sentence} → {axiom}")
