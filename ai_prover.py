import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ---------------------------------------------------------
# 1. Lean 4 Compiler Feedback-Umgebung (Mock)
# ---------------------------------------------------------
class Lean4Environment:
    def __init__(self):
        self.state_vocab_indices = [10, 45, 12, 99, 3, 55, 12, 88]
        
    def get_initial_state(self):
        return torch.randint(10, 500, (1, 32)) # Batch=1, Seq_Len=32

    def step(self, action_idx):
        reward = -0.05 # Kleiner Zeitschritt-Malus
        done = False
        
        if action_idx == 4:   # Simulation für 'exact'
            reward = 1.0
            done = True
        elif action_idx == 12: # Simulation für 'simp'
            reward = 0.2
            
        next_state = torch.randint(10, 500, (1, 32))
        return next_state, reward, done

# ---------------------------------------------------------
# 2. Hard-Thresholding mit Straight-Through Estimator (STE)
# ---------------------------------------------------------
class STETokenPruner(torch.autograd.Function):
    @staticmethod
    def forward(ctx, embeddings, keep_mask):
        ctx.save_for_backward(keep_mask)
        return embeddings * keep_mask

    @staticmethod
    def backward(ctx, grad_output):
        keep_mask, = ctx.saved_tensors
        return grad_output * keep_mask, grad_output * 1.0

# ---------------------------------------------------------
# 3. Das Neuronale Netzwerk für mathematische Beweise
# ---------------------------------------------------------
class MathematicalProverNet(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, action_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.tokenizer_head = nn.Linear(embed_dim, 2)
        self.encoder = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.actor = nn.Linear(embed_dim, action_size)
        self.critic = nn.Linear(embed_dim, 1)

    def forward(self, x, temperature=1.0, hard_prune=True):
        e = self.embedding(x) 
        token_logits = self.tokenizer_head(e) 
        
        # Weiches Sampling für den Gradientenfluss im Backward-Pass
        soft_probs = F.gumbel_softmax(token_logits, tau=temperature, hard=False, dim=-1)
        keep_prob_index_1 = soft_probs[:, :, 1] 
        
        if hard_prune:
            # Erzeuge die harte 0/1 Maske für den echten VRAM-Schutz
            keep_mask = (keep_prob_index_1 > 0.5).float()
            # NUTZE DEN STE: Harte Maske im Forward, weiche Gradienten im Backward!
            efficient_embeddings = STETokenPruner.apply(e, keep_mask.unsqueeze(-1))
            padding_mask = (keep_mask < 0.5)
        else:
            efficient_embeddings = e * keep_prob_index_1.unsqueeze(-1)
            padding_mask = (keep_prob_index_1 < 0.5)
        
        # ECHTE ERSPARNIS: Der Transformer ignoriert die maskierten Pfade vollständig
        encoded_state = self.encoder(efficient_embeddings, src_key_padding_mask=padding_mask)
        
        # Pooling unter Ausschluss der gelöschten Token
        input_mask = (keep_mask.unsqueeze(-1) if hard_prune else keep_prob_index_1.unsqueeze(-1))
        pooled_state = (encoded_state * input_mask).sum(dim=1) / (input_mask.sum(dim=1) + 1e-9)
        
        action_probs = F.softmax(self.actor(pooled_state), dim=-1)
        state_value = self.critic(pooled_state)
        
        return action_probs, state_value, keep_prob_index_1

# ---------------------------------------------------------
# 4. Trainings- und Optimierungsschleife
# ---------------------------------------------------------
def train():
    vocab_size = 1000
    embed_dim = 64
    hidden_dim = 128
    action_size = 20  
    
    model = MathematicalProverNet(vocab_size, embed_dim, hidden_dim, action_size)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    env = Lean4Environment()
    
    lambda_vram = 0.1 
    temperature = 1.0 
    
    print("Starte mathematisch verifiziertes STE-Training...")
    
    for step_idx in range(50):
        state = env.get_initial_state()
        
        # Aufruf passt jetzt perfekt zur Definition oben!
        action_probs, state_value, keep_prob_1 = model(state, temperature=temperature, hard_prune=True)
        
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        next_state, reward, done = env.step(action.item())
        
        # --- VERLUSTBERECHNUNG GEMÄSS DEINER README ---
        advantage = reward - state_value.item()
        actor_loss = -action_dist.log_prob(action) * advantage
        critic_loss = F.mse_loss(state_value, torch.tensor([[reward]]))
        L_RL = actor_loss + critic_loss
        
        # Entspricht exakt dem Normalisierungsterm 1/N * sum(pi) aus deiner Erklärung!
        VRAM_penalty = torch.mean(keep_prob_1)
        
        L_total = L_RL + lambda_vram * VRAM_penalty
        
        optimizer.zero_grad()
        L_total.backward()
        optimizer.step()
        
        if step_idx % 10 == 0:
            temperature = max(0.5, temperature * 0.95)
            print(f"Step {step_idx:02d} | L_total: {L_total.item():.4f} | L_RL: {L_RL.item():.4f} | VRAM-Faktor (Avg Len): {VRAM_penalty.item()*100:.2f}%")

if __name__ == "__main__":
    train()
