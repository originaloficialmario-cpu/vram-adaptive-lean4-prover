import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class DifferentiableVRAMTokenizer(nn.Module):
    """
    Optimiert die Token-Auswahl über Gumbel-Softmax.
    Liefert eine Maske, die direkt vom PyTorch Transformer verstanden wird.
    """
    def __init__(self, embed_dim: int, tau: float = 1.0):
        super().__init__()
        self.tau = tau  
        self.mask_predictor = nn.Linear(embed_dim, 2)

    def forward(self, x: torch.Tensor):
        # x Shape: [Batch_Size, Sequence_Length, Embed_Dim]
        logits = self.mask_predictor(x) # Shape: [B, N, 2]
        
        if self.training:
            # hard=True liefert im Forward-Pass ein One-Hot-Szenario,
            # behält im Backward-Pass aber die weichen Gradienten.
            soft_mask = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)
            binary_mask = soft_mask[:, :, 1] # 1 = Keep, 0 = Drop
        else:
            binary_mask = torch.argmax(logits, dim=-1).float()
        
        probs = F.softmax(logits, dim=-1)[:, :, 1]
        return binary_mask, probs

class TheoremProverPolicy(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_actions: int, num_heads: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.tokenizer_opt = DifferentiableVRAMTokenizer(embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.policy_head = nn.Linear(embed_dim, num_actions)
        self.value_head = nn.Linear(embed_dim, 1)

    def forward(self, state_tokens: torch.Tensor):
        x = self.embedding(state_tokens)
        
        # Maske berechnen (1 = behalten, 0 = droppen)
        mask, keep_probs = self.tokenizer_opt(x)
        
        # WICHTIG: PyTorch Transformer erwartet für src_key_padding_mask ein Byte-/Bool-Tensor,
        # bei dem TRUE bedeutet, dass das Token IGNORIERT (gedroppt) werden soll.
        # Daher invertieren wir die Maske: 1 (Keep) -> False, 0 (Drop) -> True
        src_key_padding_mask = (mask == 0)
        
        # Inferenz-Optimierung (Optional):
        # Wenn wir nicht im Training sind, könnten wir hier die Token physisch per Indexing 
        # herausschneiden, um echte VRAM/Zeit-Einsparungen im RAM zu erzielen.
        # Im Training nutzen wir das src_key_padding_mask-Feature des Transformers:
        transformer_out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Globales Pooling (Nur über valide Token)
        valid_tokens = mask.unsqueeze(-1).clamp(min=1e-9)
        pooled = (transformer_out * valid_tokens).sum(dim=1) / valid_tokens.sum(dim=1)
        
        action_logits = self.policy_head(pooled)
        state_value = self.value_head(pooled)
        
        return action_logits, state_value, keep_probs, mask

# --- KORRIGIERTER TRAININGSLOOP ---

VOCAB_SIZE = 1000
EMBED_DIM = 64
NUM_ACTIONS = 10
MAX_SEQ_LEN = 20
LAMBDA_VRAM = 0.05  
LAMBDA_VALUE = 0.5 # Gewichtung für den Kritiker-Verlust
LR = 1e-3

model = TheoremProverPolicy(VOCAB_SIZE, EMBED_DIM, NUM_ACTIONS)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# Simulierter Environment-Step bleibt gleich
from main import LeanEnvironmentSimulator # Falls in separater Datei, hier importiert
env = LeanEnvironmentSimulator(VOCAB_SIZE, MAX_SEQ_LEN)

state = torch.randint(0, VOCAB_SIZE, (1, MAX_SEQ_LEN))

log_probs = []
vram_losses = []
values = []
rewards = []

for t in range(5):
    logits, value, keep_probs, mask = model(state)
    
    dist = Categorical(logits=logits)
    action = dist.sample()
    
    log_probs.append(dist.log_prob(action))
    values.append(value.squeeze())
    
    # Schnitt der Beibehaltungs-Wahrscheinlichkeiten
    vram_losses.append(keep_probs.mean())
    
    next_state, reward, done = env.step(action.item())
    rewards.append(reward)
    
    if done:
        break
    state = next_state

# REINFORCE + Kritiker-Loss-Berechnung
returns = []
discounted_sum = 0
for r in reversed(rewards):
    discounted_sum = r + 0.99 * discounted_sum
    returns.insert(0, discounted_sum)

returns = torch.tensor(returns, dtype=torch.float32)
log_probs = torch.stack(log_probs)
vram_losses = torch.stack(vram_losses)
values = torch.stack(values)

# Advantage-Berechnung
advantages = returns - values.detach()

# 1. RL (Policy) Loss
l_rl = - (log_probs * advantages).mean()

# 2. VRAM Sparsamkeits-Loss
l_vram = vram_losses.mean()

# 3. KORREKTUR: Value (Kritiker) Loss berechnen und optimieren
l_value = F.mse_loss(values, returns)

# Gesamter Loss inklusive Kritiker-Update
l_total = l_rl + (LAMBDA_VRAM * l_vram) + (LAMBDA_VALUE * l_value)

optimizer.zero_grad()
l_total.backward()
optimizer.step()

print(f"RL Loss: {l_rl.item():.4f} | VRAM Loss: {l_vram.item():.4f} | Value Loss: {l_value.item():.4f} | Total Loss: {l_total.item():.4f}")
