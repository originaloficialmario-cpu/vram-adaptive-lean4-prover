import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging

logging.basicConfig(level=logging.INFO)

# =====================================================================
# 1. HARDWARE-ACCELERATED DYNAMIC COMPRESSION TRANSFORMER
# =====================================================================
class RealVRAMOptimizedTransformer(nn.Module):
def forward(self, x, temperature=1.0):
        B, T = x.shape  # B=1, T=32
        h = self.embedding(x) + self.pos_embedding[:, :T, :]
        
        # 1. Gumbel-Softmax für diskrete Entscheidungen
        logits = self.pruning_head(h) 
        soft_probs = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)
        keep_mask = soft_probs[:, :, 1] # Shape: [B, T] (1.0 oder 0.0)
        
        # 2. Autograd-sicheres Sicherheitsnetz (Out-of-place Modifikation)
        if keep_mask.sum() == 0:
            fallback = torch.zeros_like(keep_mask)
            fallback[0, 0] = 1.0
            # Addition erhält den Gradientenfluss im Gegensatz zur In-Place Zuweisung
            keep_mask = keep_mask + fallback 
            
        # 3. Indizes der aktiven Token extrahieren
        active_indices = torch.nonzero(keep_mask[0])[:, 0]
        h_active = h[:, active_indices, :] # Shape: [1, T_aktiv, embed_dim]
        T_active = h_active.shape[1]
        
        # 4. CRITICAL FIX: Gradienten-Brücke für den Straight-Through Estimator bauen
        # keep_mask hat an diesen Indizes den Wert 1.0. Das ändert die Werte nicht, 
        # aber verknüpft h_active im Autograd-Graphen wieder mit dem pruning_head!
        mask_active = keep_mask[0, active_indices].unsqueeze(-1).unsqueeze(0)
        h_active = h_active * mask_active
        
        # 5. Transformer-Verarbeitung (Rest bleibt gleich)
        for i in range(self.depth):
            h_norm = self.ln1[i](h_active)
            # ... Rest deines Attention-Codes ...
            
            q = self.q_projections[i](h_norm).view(B, T_active, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_projections[i](h_norm).view(B, T_active, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_projections[i](h_norm).view(B, T_active, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Läuft jetzt mit maximaler FlashAttention-Geschwindigkeit auf der GPU
            attn_out = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None, # Keine verlangsamende Maske nötig!
                dropout_p=0.0,
                is_causal=False
            )
            
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T_active, -1)
            h_active = h_active + self.out_projections[i](attn_out)
            h_active = h_active + self.ffn2[i](F.gelu(self.ffn1[i](self.ln2[i](h_active))))
            
        # 4. Globales Pooling über die verbleibenden aktiven Repräsentationen
        pooled = h_active.mean(dim=1)
        
        action_logits = self.policy_head(pooled)
        state_value = self.value_head(pooled)
        
        return action_logits, state_value, keep_mask

# =====================================================================
# 2. OPTIMIERUNGSSCHLEIFE (RL + VRAM PENALTY)
# =====================================================================
def train_prover():
    vocab_size = 1000
    embed_dim = 128
    num_heads = 4
    depth = 2
    max_seq_len = 32
    num_actions = 4  
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Zünde ultra-optimierten FlashAttention-Kernel auf Device: {device.type.upper()}")
    
    model = RealVRAMOptimizedTransformer(
        vocab_size=vocab_size, 
        embed_dim=embed_dim, 
        num_heads=num_heads, 
        depth=depth, 
        max_seq_len=max_seq_len,
        num_actions=num_actions
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)
    
    lambda_vram = 0.1
    temperature = 1.0

    for step in range(50):
        state_tokens = torch.randint(0, vocab_size, (1, max_seq_len)).to(device)
        
        # Forward Pass mit deinem neuen Dynamic-Slicing-Kernel
        action_logits, state_value, keep_mask = model(state_tokens, temperature=temperature)
        
        action_dist = torch.distributions.Categorical(logits=action_logits)
        action = action_dist.sample()
        
        reward = 1.0 if action.item() == 0 else -0.05
        reward_tensor = torch.tensor([[reward]], device=device)
        
        # Verlustberechnung
        advantage = reward - state_value.item()
        actor_loss = -action_dist.log_prob(action) * advantage
        critic_loss = F.mse_loss(state_value, reward_tensor)
        L_RL = actor_loss + critic_loss
        
        VRAM_penalty = torch.mean(keep_mask)
        L_total = L_RL + lambda_vram * VRAM_penalty
        
        optimizer.zero_grad()
        L_total.backward()
        optimizer.step()
        
        if step % 10 == 0:
            temperature = max(0.5, temperature * 0.95)
            logging.info(f"Schritt {step:02d} | Total Loss: {L_total.item():.4f} | "
                         f"Physische GPU-Sparsity: {(1.0 - VRAM_penalty.item()) * 100:.2f}%")

if __name__ == "__main__":
    train_prover()
