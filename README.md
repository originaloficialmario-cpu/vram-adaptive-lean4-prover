Die Verlustfunktion für dynamische Tokenizer-Effizienz

Um den Tokenizer während des Trainings anzupassen, fügen wir der RL-Verlustfunktion einen Regularisierungsterm hinzu. Wir definieren eine weiche Maske über die Sequenzlänge, die das Modell zwingt, nur die relevantesten Token im VRAM zu behalten.

Die kombinierte Verlustfunktion lautet:
Ltotal​=LRL​+λ(αt=1∑T​σ(W⋅ht​)−βt=1∑T​pt​log(pt​))

Erklärung der Komponenten:

    LRL​: Der Standard-Reinforcement-Learning-Verlust (z. B. PPO oder REINFORCE).

    λ: Skalierungsfaktor für den Tokenizer-Verlust.

    α∑σ(W⋅ht​): Bestraft die Anzahl der aktivierten Token (L1-ähnliche Sparsity), um VRAM freizugeben.

    β∑pt​log(pt​): Entropie-Term, der verhindert, dass das Modell alle Token komplett verwirft, und stattdessen eine informierte Entscheidung trifft.
