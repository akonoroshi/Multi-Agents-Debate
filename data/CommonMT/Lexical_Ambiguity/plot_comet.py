import os
import json
import matplotlib.pyplot as plt

debate_path = "MAD_Debate_Process"
files = os.listdir(debate_path)
data = []
for file in files:
    print('Processing', file)
    with open(os.path.join(debate_path, file), 'r') as f:
        debate = json.load(f)
    data.append(debate['comet score'])
plt.hist(data, bins=20, edgecolor='black')
plt.savefig('comet_score_distribution.png')