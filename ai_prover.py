import os
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F

# =====================================================================
# 1. REAL LEAN 4 COMPILER INTERACTION PIPELINE
# =====================================================================
class Lean4Environment:
    def __init__(self, project_dir="math_project"):
        self.project_dir = project_dir
        if not os.path.exists(project_dir):
            os.makedirs(project_dir)

    def execute_proof_step(self, theorem_name, proof_script):
        """
        Schreibt das mathematische Theorem live in eine Datei und
        jagt es direkt durch den frisch installierten Lean 4 Compiler.
        """
        file_path = os.path.join(self.project_dir, f"{theorem_name}.lean")
        
        # Erstelle ein echtes Lean 4 Dokument mit Anbindung an die heruntergeladene Mathlib
        lean_code = f"""import Mathlib

theorem {theorem_name} {proof_script}
"""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(lean_code)
            
        # Rufe den echten Lean 4 Compiler auf deinem G5 auf!
        try:
            result = subprocess.run(
                ["lean", file_path], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            
            # Auswertung des echten Compiler-Feedbacks
            output = result.stderr if result.stderr else result.stdout
            
            if result.returncode == 0 and "error" not in output.lower():
                reward = 1.0  # Der mathematische Beweis ist absolut fehlerfrei!
                done = True
                info = "Beweis akzeptiert!"
            else:
                done = False
                info = output if output else "Unbekannter Syntaxfehler"
                # Belohnungssystem basierend auf verbleibenden Zielen (Goals)
                if "remaining goals" in output.lower():
                    reward = 0.2
                else:
                    reward = -0.1  # Schlimmer Logikfehler im Beweis
                    
        except subprocess.TimeoutExpired:
            reward = -0.5
            done = False
            info = "Compiler Timeout"
            
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
# 3. RL REINFORCEMENT LEARNING TRAINING LOOP
# =====================================================================
def train_rl_prover():
    vocab_size = 1000
    embedding_dim = 128
    hidden_dim = 256
    action_dim = 4 
    epochs = 50  # Kompakter Testlauf für echte Compiler-Geschwindigkeit
    clip_eps = 0.2
    lambda_vram = 0.01 
    
    # Echte Lean 4 Beweis-Taktiken
    tactic_map = {0: ":= by rfl", 1: ":= by linarith", 2: ":= by ring", 3: "(n : Nat) : n + 0 = n := by induction n"}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCriticModel(vocab_size, embedding_dim, hidden_dim, action_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    env = Lean4Environment()

    print(f"Starte Echtzeit-Training auf Hardware-Target: {device.type.upper()}")

    for epoch in range(epochs):
        state_tokens = torch.randint(1, vocab_size, (1, 20)).to(device) 
        
        logits, value, vram_mask = model(state_tokens, temperature=1.0)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        chosen_tactic = tactic_map[action.item()]
        
        # Aufruf des echten System-Prüfers!
        reward_val, done, feedback_info = env.execute_proof_step(
            theorem_name=f"real_proof_{epoch}",
            proof_script=chosen_tactic
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
        
        if epoch % 5 == 0:
            # Bereinige Newlines aus den Compilerfehlern für saubere Anzeige
            clean_info = feedback_info.replace('\n', ' ').strip()[:30]
            print(f"Epoch {epoch:02d} | Loss: {total_loss.item():.4f} | VRAM: {vram_mb:.2f} MB | Lean4: {clean_info}")

if __name__ == "__main__":
    train_rl_prover()
