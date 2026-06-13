# VRAM-Adaptive Lean 4 Prover

Reinforcement Learning based mathematical theorem prover with differentiable VRAM tokenizer optimization.

### 📐 Mathematical Foundation

$$L_{\text{total}}(\theta, \phi, \psi) = L_{\text{PPO}}(\theta, \phi) + \lambda \cdot L_{\text{VRAM}}(\psi)$$

Where the adaptive VRAM loss function is defined as:

$$L_{\text{VRAM}}(\psi) = \mathbb{E}_{s_t} \left[ \left( \frac{\text{VRAM}_{\text{allocated}}}{\text{VRAM}_{\text{max}}} \right) \cdot \sum_{i=1}^{V} \sigma_{\tau} \left( \pi_{\psi}(e_i \vert{} s_t) \right) \cdot \log\left( \frac{\sigma_{\tau} \left( \pi_{\psi}(e_i \vert{} s_t) \right)}{\mathcal{U}_i} \right) \right]$$
