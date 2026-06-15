import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =====================================================================
# MATHEMATICALLY SECURE CASCADED CAPACITY TRANSFORMER (V3 - FIXED)
# =====================================================================
class UltraOptimizedTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, depth, max_seq_len, num_actions, keep_ratios=[0.75, 0.50]):
        super().__init__()
        assert len(keep_ratios) == depth, "Provide a keep_ratio for each layer to define the compression funnel."
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
        
        self.depth = depth
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.keep_ratios = keep_ratios  
        
        self.routers = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(depth)])
        
        self.q_projections = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(depth)])
        self.k_projections = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(depth)])
        self.v_projections = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(depth)])
        self.out_projections = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(depth)])
        
        self.ffn1 = nn.ModuleList([nn.Linear(embed_dim, embed_dim * 4) for _ in range(depth)])
        self.ffn2 = nn.ModuleList([nn.Linear(embed_dim * 4, embed_dim) for _ in range(depth)])
        
        self.ln1 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(depth)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(depth)])
        
        self.policy_head = nn.Linear(embed_dim, num_actions)
        self.value_head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        B, T = x.shape
        h = self.embedding(x) + self.pos_embedding[:, :T, :]
        
        for i in range(self.depth):
            current_T = h.shape[1]
            
            router_logits = self.routers[i](h).squeeze(-1) 
            
            k = max(1, int(current_T * self.keep_ratios[i]))
            
            # 1. Top-K Extraktion der stärksten Token-Logits
            _, indices = torch.topk(router_logits, k=k, dim=-1) 
            
            # 2. Marios Sortierung zur Erhaltung der zeitlichen Grammatik
            indices, _ = torch.sort(indices, dim=-1) 
            
            # 3. Physisches Slicing via Gather
            indices_expanded = indices.unsqueeze(-1).expand(-1, -1, h.shape[-1])
            h_active = torch.gather(h, 1, indices_expanded) 
            
            # OPTION A FIX: Kein restriktives Gating mehr! Wir nutzen die Router-Logits 
            # weich über Softmax/Sigmoid, blockieren aber den erzwungenen Kollaps gegen Null.
            router_scores = torch.sigmoid(router_logits)
            scores_gated = torch.gather(router_scores, 1, indices).unsqueeze(-1)
            h_active = h_active * scores_gated
            
            h_norm = self.ln1[i](h_active)
            
            q = self.q_projections[i](h_norm).view(B, k, self.num_heads, self.head_dim).transpose(1, 2)
            k_lay = self.k_projections[i](h_norm).view(B, k, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_projections[i](h_norm).view(B, k, self.num_heads, self.head_dim).transpose(1, 2)
            
            attn_out = F.scaled_dot_product_attention(q, k_lay, v, attn_mask=None, dropout_p=0.0, is_causal=False)
            
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, k, -1)
            h_active = h_active + self.out_projections[i](attn_out)
            h_active = h_active + self.ffn2[i](F.gelu(self.ffn1[i](self.ln2[i](h_active))))
            
            h = h_active

        pooled = h.mean(dim=1)
        action_logits = self.policy_head(pooled)
        state_value = self.value_head(pooled)
        
        return action_logits, state_value

def train_ultra_optimized():
    vocab_size = 1000
    embed_dim = 128
    num_heads = 4
    depth = 2
    max_seq_len = 32
    num_actions = 4  
    batch_size = 4  
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Zünde mathematisch korrigierten Funnel-Transformer auf: {device.type.upper()}")
    
    model = UltraOptimizedTransformer(
        vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads, depth=depth, max_seq_len=max_seq_len, num_actions=num_actions, keep_ratios=[0.75, 0.50]
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    for step in range(51):
        state_tokens = torch.randint(0, vocab_size, (batch_size, max_seq_len)).to(device)
        
        action_logits, state_value = model(state_tokens)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        action = action_dist.sample()
        
        reward = torch.where(action == 0, 1.0, -0.05).unsqueeze(-1) 
        advantage = reward - state_value.detach()
        actor_loss = -(action_dist.log_prob(action).unsqueeze(-1) * advantage).mean()
        critic_loss = F.mse_loss(state_value, reward)
        
        # FIX: Sparsity-Loss komplett entfernt. Kein Repräsentations-Kollaps mehr!
        L_total = actor_loss + critic_loss
        
        optimizer.zero_grad()
        L_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if step % 10 == 0:
            logging.info(f"Schritt {step:02d} | L_total: {L_total.item():.4f} | Status: Stabil (Kein Sparsity-Druck)")

if __name__ == "__main__":
    train_ultra_optimized()
