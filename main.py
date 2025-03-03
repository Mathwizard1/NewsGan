import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained LLM model and tokenizer (for paraphrasing)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
generator = GPT2LMHeadModel.from_pretrained('gpt2')

# Simple Discriminators for toxicity and factuality checks
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Binary classification: toxic/non-toxic or factual/non-factual
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)

# Load discriminators
toxicity_discriminator = Discriminator(768)  # Assuming embedding size 768
factual_discriminator = Discriminator(768)

# Optimizers for generator and discriminators
gen_optimizer = optim.Adam(generator.parameters(), lr=1e-5)
tox_opt = optim.Adam(toxicity_discriminator.parameters(), lr=1e-4)
fact_opt = optim.Adam(factual_discriminator.parameters(), lr=1e-4)

# Loss function (Binary Cross Entropy for discriminator outputs)
criterion = nn.BCELoss()

# Thresholds
toxicity_threshold = 0.5  # Toxicity threshold
factual_threshold = 0.5  # Factual score threshold

# Paraphrasing function using LLM
def paraphrase_sentence(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = generator.generate(inputs['input_ids'], max_length=50, num_beams=5, num_return_sequences=1)
    paraphrased_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return paraphrased_sentence

# A simple feature extractor to get sentence embeddings (or use pre-trained like BERT embeddings)
def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = generator.transformer.wte(inputs['input_ids']).mean(dim=1)  # Use token embeddings as feature
    return outputs

# Training loop
def train_step(sentence):
    generator.train()
    toxicity_discriminator.train()
    factual_discriminator.train()

    # 1. Paraphrase the sentence using the generator (LLM)
    paraphrased_sentence = paraphrase_sentence(sentence)
    
    # 2. Get embeddings for the paraphrased sentence
    paraphrased_embedding = get_sentence_embedding(paraphrased_sentence)

    # 3. Pass through the toxicity and factual discriminators
    tox_pred = toxicity_discriminator(paraphrased_embedding)
    fact_pred = factual_discriminator(paraphrased_embedding)
    
    # Assuming we know the ground truth for toxicity (non-toxic: 0) and factuality (factual: 1)
    target_tox = torch.tensor([0.0])  # Non-toxic as target
    target_fact = torch.tensor([1.0])  # Factual as target

    # 4. Calculate losses for the discriminators
    tox_loss = criterion(tox_pred, target_tox)
    fact_loss = criterion(fact_pred, target_fact)

    # 5. If the paraphrased sentence crosses thresholds, we backpropagate
    if tox_pred.item() > toxicity_threshold or fact_pred.item() < factual_threshold:
        # 6. Backpropagate on generator and discriminators
        gen_optimizer.zero_grad()
        tox_opt.zero_grad()
        fact_opt.zero_grad()
        
        # Combine the losses for training
        combined_loss = tox_loss + fact_loss
        combined_loss.backward()  # Backpropagation
        
        # Optimizer steps
        gen_optimizer.step()
        tox_opt.step()
        fact_opt.step()

    return paraphrased_sentence, tox_pred.item(), fact_pred.item()

# Example usage
sentence = "The weather today is great."
for epoch in range(10):
    paraphrased, tox_score, fact_score = train_step(sentence)
    print(f"Epoch {epoch+1}:")
    print(f"Paraphrased: {paraphrased}")
    print(f"Toxicity Score: {tox_score}, Factual Score: {fact_score}\n")
