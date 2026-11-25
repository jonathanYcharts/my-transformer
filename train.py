# Run python3 train.py
# 1. We always start with a dataset to train on. (tiny shakespeare dataset)
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# print("Length of dataset in characters: ", len(text))

# let's look at the first 1000 characters
# print(text[:1000])

# 2. Here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(''.join(chars))
# print(vocab_size)

# 3. Create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# print(encode("hii there"))
# print(decode(encode("hii there")))

import torch
data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:1000]) # the 1000 characters we looked at earier will to the GPT look like this


# 4. Let's now split up the data into train and validation sets
n = int(0.9*len(data)) # first 90% will be train
# We keep the last 10% of the text hidden from the model. The model learns from the first 90%,
# and every so often we check how well it does on the 10% it has never seen.
train_data = data[:n]
val_data = data[n:]

block_size = 8
train_data[:block_size+1]

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    # print(f"when input is {context} the target: {target}")

# makes the "random" picks the same on every run, so output is reproducible.
torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
# print('inputs:')
# print(xb.shape)
# print(xb)
# print('targets:')
# print(yb.shape)
# print(yb)

# print('----')

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        # print(f"when input is {context.tolist()} the target: {target}")

# print(xb) # our input to the transformer

# =================== Simplest possible "language model" before any training =======================

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337) # fix the randomness

class BigramLanguageModel(nn.Module):
    # "Bigram" means pairs. This model looks only at the current character and guesses the next one. Seeing q, it 
    # should learn that u usually follows. It has no memory of anything before the current character. 

    def __init__(self, vocab_size):
        super().__init__()
        # This creates a 65 × 65 table of numbers, starting out random.
        #   - Each row belongs to one current character.
        #   - Each column belongs to a possible next character.
        #   - The number in a cell is a score for "how likely is this next character?"
        # Each token directly reads off the logits for the next token from a lookup table
        # Training means adjusting the numbers in this table.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # Making predictions and measuring the mistakes
        # idx and targets are both (B,T) tensor of integers
        # For every character in the input, the model looks up its row in the table and gets 65 scores (Logits).
        logits = self.token_embedding_table(idx) # (Batch, Time, Channels)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # flatten the 4×8 positions into one list of 32, because cross_entropy expects the data in that form
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # Check each of the 32 positions: "how much score did the model give the character that actually came next?"
            loss = F.cross_entropy(logits, targets)

        # loss is a single number for how wrong the guesses were. Lower is better.
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # keep only the last position, because we only care about what comes next after the end of the text so far.
            logits = logits[:, -1, :] # becomes (B, C)
            # turn the scores into probabilities that add up to 100%. (e.g. e = 12%, a = 8%, z = 0.1% and so on)
            probs = F.softmax(logits, dim=-1) # (B, C)
            # pick one character at random, weighted by those probabilities. Like a loaded dice
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # add the chosen character to the end of the text
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

# torch.zeros((1, 1)) is the starting text: a single character with number 0, which is the newline character
# generates 100 more characters. [0] takes the first (and only) snippet, .tolist() converts it to a normal 
# Python list, and decode turns the numbers back into letters.
# The output will be gibberish like pYCXxfRkRZd,lL..., because the table is still random.
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))


# ==================================== Train the model ========================================
# The model practices on the text over and over and gets a little less wrong each time

# Training repeats one simple routine:
#   1. Let the model make guesses.
#   2. Measure how wrong it was (the loss).
#   3. Work out which way to turn each knob so it would be a bit less wrong.
#   4. Turn every knob a tiny bit in that direction.
#   5. Repeat many times.

# That's all "learning" means. Large models like ChatGPT do the same thing, just with billions of knobs.

# create a PyTorch optimizer
# lr = learning rate. Too big, and the model overshoots and becomes unstable. Too small, and learning is painfully slow.
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
for steps in range(1000): # increase number of steps for good results

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))
# Why it's still not great
# This model only looks at one character to guess the next. After an h, it can't tell whether it's in "the", "she" or "what". That's the limit of a bigram
# model. The rest of building a transformer is about letting characters look back at earlier characters (this is called attention), which is where real improvement
# comes from. See self_attention_math.py
