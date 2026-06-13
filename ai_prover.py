import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F

# =====================================================================
# 1. MOCK LEAN 4 COMPILER ENVIRONMENT
# =====================================================================
class Lean4Environment:
    def __init__(self, project_dir="math_project"):
        self.project_dir = project_dir

    def execute_proof_step(self, theorem_name, proof_script):
        # Simuliertes Feedback, damit das RL-Modell ohne installierten Compiler trainiert
        if random.random() > 0.3:
            reward = 1.0
            done = True
            info = "Beweis akzeptiert! (Simuliert)"
        else:
            done = False
            info = "remaining goals: 1"
            reward = 0.1
        return reward, done, info

# =====================================================================
# 2. DYNAMIC DIFF-TOKENIZER & NEURAL NETWORK ARCHITECTURE
# =====================================================================
class DifferentiableTokenizer(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(DifferentiableTokenizer, self).__init__()
        self.vocab_size = vocab_size
        self.token_scoring = nn.Linear(embedding_dim, 1) 
        
    def forward(self, embeddings, temperature=1.0):
        scores = self.token_scoring(embeddings).squeeze(-1)
        mask = F.gumbel_softmax(scores, tau=temperature, hard=True)
        return mask

class ActorCriticModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, action_dim):
        super(ActorCriticModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.tokenizer_vram_ctrl = DifferentiableTokenizer(vocab_size, embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.actor = nn.Linear(embedding_dim, action_dim)
        self.critic = nn.Linear(embedding_dim, 1)

    def forward(self, x, temperature=1.0):
        raw_embeddings = self.embedding(x)
        vram_mask = self.tokenizer_vram_ctrl(raw_embeddings, temperature)
        masked_embeddings = raw_embeddings * vram_mask.unsqueeze(-1)
        
        features = self.transformer(masked_embeddings)
        pooled_features = features.mean(dim=1)
        
        action_logits = self.actor(pooled_features)
        state_values = self.critic(pooled_features)
        
        return action_logits, state_values, vram_mask

# =====================================================================
# 3. RL REINFORCEMENT LEARNING TRAINING LOOP WITH PPO & CUDA
# =====================================================================
def train_rl_prover():
    vocab_size = 1000
    embedding_dim = 128
    hidden_dim = 256
    action_dim = 50 
    epochs = 100
    gamma = 0.99
    clip_eps = 0.2
    lambda_vram = 0.01 
    
    tactic_map = {0: "by rfl", 1: "by linarith", 2: "by ring", 3: "by intro"}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCriticModel(vocab_size, embedding_dim, hidden_dim, action_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    env = Lean4Environment()

    print(f"Starte Training auf Hardware-Target: {device.type.upper()}")

    for epoch in range(epochs):
        state_tokens = torch.randint(1, vocab_size, (1, 20)).to(device) 
        
        logits, value, vram_mask = model(state_tokens, temperature=1.0)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        tactic_idx = action.item() % len(tactic_map)
        chosen_tactic = tactic_map[tactic_idx]
        
        reward_val, done, feedback_info = env.execute_proof_step(
            theorem_name=f"epoch_proof_{epoch}",
            proof_script=f"(n : Nat) : n + 0 = n := {chosen_tactic}"
        )
        
        if device.type == "cuda":
            vram_allocated = torch.cuda.memory_allocated(device)
            vram_max = torch.cuda.get_device_properties(device).total_memory
            vram_ratio = vram_allocated / vram_max
            vram_mb = vram_allocated / (1024 * 1024)
        else:
            vram_ratio = 0.35 
            vram_mb = 142.0
            
        reward = torch.tensor([reward_val], device=device)
        advantage = reward - value.squeeze(-1).detach()
        
        ratio = torch.exp(log_prob - log_prob.detach())
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(value.squeeze(-1), reward)
        
        vram_loss = vram_ratio * torch.sum(vram_mask * torch.log(vram_mask + 1e-8))
        total_loss = actor_loss + 0.5 * critic_loss + lambda_vram * vram_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {total_loss.item():.4f} | VRAM: {vram_mb:.2f} MB | Ratio: {vram_ratio*100:.3f}% | Lean4: {feedback_info.strip()[:25]}")

if __name__ == "__main__":
    train_rl_prover()
