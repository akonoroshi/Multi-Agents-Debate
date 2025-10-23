import os
import json
from comet import download_model, load_from_checkpoint

# Choose your model from Hugging Face Hub
model_path = download_model("Unbabel/XCOMET-XL")
# or for example:
# model_path = download_model("Unbabel/wmt22-comet-da")

# Load the model checkpoint:
model = load_from_checkpoint(model_path)

debate_path = "MAD_Debate_Process"
files = os.listdir(debate_path)
for file in files:
    print('Processing', file)
    with open(os.path.join(debate_path, file), 'r') as f:
        debate = json.load(f)
    
    data = [
        {
            "src": debate['source'],
            "mt": debate['debate translation'],
            "ref": debate['correct reference']
        }
    ]

    # Call predict method:
    model_output = model.predict(data, batch_size=8, gpus=1)

    # Sentence-level scores (list)
    debate['comet score'] = model_output.scores[0]

    with open(os.path.join(debate_path, file), 'w') as f:
        json.dump(debate, f, indent=4)

"""
# Data must be in the following format:
data = [
    {
        "src": "10 到 15 分钟可以送到吗",
        "mt": "Can I receive my food in 10 to 15 minutes?",
        "ref": "Can it be delivered between 10 to 15 minutes?"
    },
    {
        "src": "Pode ser entregue dentro de 10 a 15 minutos?",
        "mt": "Can you send it for 10 to 15 minutes?",
        "ref": "Can it be delivered between 10 to 15 minutes?"
    }
]
# Call predict method:
model_output = model.predict(data, batch_size=8, gpus=1)

# Sentence-level scores (list)
print(model_output.scores)
#[0.9822099208831787, 0.9599897861480713]

# System-level score (float)
print(model_output.system_score)
#0.971099853515625

# Detected error spans (list of list of dicts)
print(model_output.metadata.error_spans)

[
  [{'confidence': 0.4160953164100647,
   'end': 21,
   'severity': 'minor',
   'start': 13,
   'text': 'my food'}],
  [{'confidence': 0.40004390478134155,
   'end': 19,
   'severity': 'minor',
   'start': 3,
   'text': 'you send it for'}]
]
"""