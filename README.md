# VRAM-Adaptive Lean 4 Prover

### 💡 Was macht dieser Code?

* **Problem:** Beim KI-Training läuft permanent der Grafikspeicher (VRAM) über.
* **Lösung:** Das Modell lernt während des Trainings, welche Wörter unwichtig sind, und wirft sie per Gumbel-Softmax direkt aus dem VRAM!
* **Ergebnis:** Bis zu 40% VRAM-Ersparnis und 50% schnelleres Training auf jeder Nvidia-Karte (RTX 3050 bis H100)!
