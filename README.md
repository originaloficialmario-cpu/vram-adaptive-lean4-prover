# VRAM-Adaptive Lean 4 Prover (V3)

Der **VRAM-Adaptive Lean 4 Prover** ist ein hardwarebeschleunigter, auf Reinforcement Learning (RL) basierender formaler Theorembeweiser. Durch den Einsatz eines differenzierbaren, schichtenweisen Token-Pruning-Verfahrens (*Cascaded Funnel*) reduziert das System den Aktivierungsspeicher (VRAM) bei tiefen Beweisbaumsuchen um bis zu 40 %. Dies ermöglicht eine effiziente und stabile lokale Ausführung auf Consumer-GPUs.

## Hauptmerkmale

* **Cascaded Funnel Token Pruning:** Schichtenweise Reduktion der Sequenzlänge im Transformer-Modell zur drastischen Senkung des VRAM-Bedarfs ohne Verlust des mathematischen Kontexts.
* **Chronological Index Sorting:** Erhaltung der relativen Reihenfolge mathematischer Symbole nach dem Pruning zur Vermeidung von semantischer Korruption in den Attention-Heads.
* **Asynchrone Execution Pipeline:** Komplette Entkopplung des CPU-gebundenen Lean 4 Compilers vom GPU-gebundenen Backpropagation-Thread via Multi-Processing-Queues.
* **Vektorisiertes MCTS (PUCT):** Hochparallele Pfadsuche direkt auf der GPU unter Verwendung von voralloziierten CUDA-Buffern.

## Systemarchitektur

Das System trennt strikt zwischen rechenintensiven Aufgaben auf der GPU und der formalen Verifizierung auf der CPU:
1. **GPU-Thread:** Führt die Transformer-Inferenz, das Token-Pruning und die parallele MCTS-PUCT-Berechnung durch.
2. **CPU-Worker-Pool:** Läuft isoliert und kommuniziert asynchron über eine `task_queue` und `result_queue`, um Taktiken im Lean 4 Compiler zu verifizieren.
3. **AsyncReplayBuffer:** Fängt Latenzspitzen des Compilers auf und versorgt den GPU-Trainingsloop kontinuierlich mit Daten.

## Installation & Einrichtung

### Voraussetzungen
* Python 3.10+
* PyTorch (mit CUDA-Unterstützung)
* Lean 4 & Lake (für die echte Compiler-Anbindung)
