# Mechanically Inducing Deceptive Alignment
##  Overview

This project uses **activation steering** to isolate and amplify the latent "refusal direction" in Large Language Models (Llama-3-70B, Llama-3-8B, Mistral-7B).

We demonstrate that refusal is a **latent control mechanism** distinct from capability. By mechanically amplifying this direction, we observe:

* **Differential Collapse:** Safety guardrails trigger at low steering strengths ($\alpha \approx 2$), while benign capabilities (Math, Code) remain robust until $\alpha \approx 6$.
* **Induced Sandbagging:** In the transition zone, models **deceptively claim incapacity** (e.g., *"I cannot write code"*) for tasks they are demonstrably capable of solving.


