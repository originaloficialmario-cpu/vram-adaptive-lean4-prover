import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO)

# ==========================================
# 1. Lean 4 Compiler Feedback System
# ==========================================
class Lean4Environment:
    def step(self, action_str):
        """ Simuliertes Feedback der Lean 4 REPL """
        if "sorry" in action_str:
            return "Fehler: sorry nicht erlaubt", -1.0, True
        elif "rfl" in action_str:
            return "Proof Complete", 10.0, True
        else:
            return "Tactic state updated", 0.1, False

# ==========================================
# 2. Dynamischer VRAM-Tokenizer (Gumbel-Softmax)
# ==========================================
class DynamicVRAMTokenizer(nn.Module):
    def __init__(self, embed_dim, temperature=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.keep_predictor = nn.Linear(embed_dim, 2) 

    def forward(self, x):
        """
        x: [Batch (muss 1 sein für dynamischen Slice), Sequence_Length, Embed_Dim]
        """
        logits = self.keep_predictor(x) 
        
        if self.training:
            gumbel_out = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)
            keep_mask_discrete = gumbel_out[0, :, 1] 
        else:
            keep_mask_discrete = (logits[0, :, 1] > logits[0, :, 0]).float()
        
        soft_probs = F.softmax(logits / self.temperature, dim=-1)[0, :, 1]
        
        # --- STRAFFUNG DER SEQUENZ (Echte VRAM-Einsparung) ---
        # Wir extrahieren nur die Zeilen, bei denen die diskrete Maske == 1 ist.
        indices = torch.nonzero(keep_mask_discrete).squeeze(-1)
        
        if indices.numel() == 0:
            # Fallback: Falls die KI alles löschen will, behalten wir das erste Token
            indices = torch.tensor([0], device=x.device)
            
        # Dynamischer Slice der Sequenzlänge
        x_filtered = x[:, indices, :]
        
        return x_filtered, soft_probs, keep_mask_discrete

# ==========================================
# 3. RL-Policy Netzwerk (Prover Agent)
# ==========================================
class LeanProverNet(nn.Module):
    def __init__(self, vocab_size, embed_dim, action_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dynamic_tokenizer = DynamicVRAMTokenizer(embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.policy_head = nn.Linear(embed_dim, action_dim)
        self.value_head = nn.Linear(embed_dim, 1)

    def forward(self, state_tokens):
        x = self.embedding(state_tokens)
        
        # Sequenz wird hier physisch gekürzt, bevor sie in den Transformer geht!
        x_filtered, keep_probs, mask = self.dynamic_tokenizer(x)
        
        transformer_out = self.transformer(x_filtered)
        pooled = transformer_out.mean(dim=1)
        
        action_logits = self.policy_head(pooled)
        state_value = self.value_head(pooled)
        
        return action_logits, state_value, keep_probs

# ==========================================
# 4. Training Loop
# ==========================================
def train_prover():
    vocab_size, embed_dim, action_dim = 1000, 128, 4
    lr = 3e-4
    lambda_vram = 0.5  # Höhere Gewichtung für spürbaren Kompressionsdruck
    epochs = 50
    
    model = LeanProverNet(vocab_size, embed_dim, action_dim)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    env = Lean4Environment()
    
    action_mapping = {0: "rfl", 1: "intro x", 2: "simp", 3: "sorry"}
    
    for epoch in range(epochs):
        state_tokens = torch.randint(0, vocab_size, (1, 20)) 
        
        log_probs, values, rewards, token_keep_probs = [], [], [], []
        done = False
        
        while not done:
            action_logits, state_value, keep_probs = model(state_tokens)
            
            action_dist = torch.distributions.Categorical(logits=action_logits)
            action = action_dist.sample()
            
            log_prob = action_dist.log_prob(action)
            action_str = action_mapping.get(action.item(), "sorry")
            
            _, reward, done = env.step(action_str)
            
            log_probs.append(log_prob)
            values.append(state_value)
            rewards.append(reward)
            token_keep_probs.append(keep_probs.mean())
            
            state_tokens = torch.randint(0, vocab_size, (1, 20))
            if len(rewards) > 5: break
                
        # --- REINFORCE Verlustberechnung ---
        R = sum(rewards)
        policy_loss = []
        vram_loss = []
        
        for log_prob, val, keep_prob in zip(log_probs, values, token_keep_probs):
            advantage = R - val.item()
            policy_loss.append(-log_prob * advantage)
            
            # Entropie-Regularisierung hinzugefügt: Bestraft zu hohe "Keep"-Wahrscheinlichkeiten
            vram_loss.append(keep_prob)
            
        policy_loss = torch.stack(policy_loss).sum()
        vram_loss = torch.stack(vram_loss).mean()
        
        total_loss = policy_loss + lambda_vram * vram_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            logging.info(f"Epoch {epoch:02d} | Loss: {total_loss.item():.4f} | "
                         f"Tokens im VRAM behalten: {vram_loss.item():.2%} | Gesamt-Reward: {R}")

if __name__ == "__main__":
    train_prover()
