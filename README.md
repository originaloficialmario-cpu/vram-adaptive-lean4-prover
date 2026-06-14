# VRAM-Adaptive Lean 4 Prover

Reinforcement Learning based mathematical theorem prover with differentiable VRAM tokenizer optimization.

### 📐 Mathematische Fundierung

$$L_{\text{total}} = L_{\text{RL}}(\theta) + \lambda \cdot L_{\text{VRAM}}(\phi)$$

Wobei der RL-Loss (hier als klassischer REINFORCE/Policy-Gradient mit Advantage $A_t$ dargestellt) definiert ist als:

$$L_{\text{RL}}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log \pi_{\theta}(a_t \mid s_t, m_t)A_t$$

Die Verlustfunktion für die dynamische Token-Anpassung ($L_{\text{VRAM}}$), welche die VRAM-Auslastung minimiert, berechnet sich aus dem Erwartungswert der Beibehaltungswahrscheinlichkeiten $p_i$ für jedes Token $i$ der Sequenzlänge $N$:

$$L_{\text{VRAM}}(\phi) = \frac{1}{N}\sum_{i=1}^{N} p_{\phi}(x_i)$$

Durch den Gumbel-Softmax-Trick wird die diskrete Maske $m_i \in \{0,1\}$ im Forward-Pass verwendet, um tatsächlichen VRAM durch das Entfernen von Token in den Attention-Schichten zu sparen, während im Backward-Pass die kontinuierlichen Wahrscheinlichkeiten $p_i$ für den Gradientenfluss genutzt werden.

### 💻 PyTorch-Implementierung

Das folgende Skript stellt eine vollständige Architektur dar. Es simuliert die Interaktion mit dem Lean 4 Compiler (da ein echter Aufruf eine spezifische Lean-REPL-Umgebung erfordert) und implementiert das neuartige dynamische Token-Masking.
