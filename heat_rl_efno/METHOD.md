# Deep Deterministic Policy Gradient with Extended Fourier Neural Operators (Heat Control)

This note summarizes the algorithm implemented in `heat_rl_efno/` in the same spirit as OpenAI Spinning Up’s [DDPG documentation](https://spinningup.openai.com/en/latest/algorithms/ddpg.html): background, key equations, exploration, and pseudocode. The code is a **continuous-action actor–critic** method in the **DDPG** family (fitted Q-learning plus a deterministic policy gradient), with **policy** and **Q-function** represented as **neural operators** on a \((x,t)\) grid using the **extended FNO (EFNO)** from the Burgers BC scripts.

---

## Background

**Deep Deterministic Policy Gradient (DDPG)** learns a Q-function \(Q_\phi(s,a)\) and a deterministic policy \(\mu_\theta(s)\) together. It is motivated like Q-learning: if \(Q\) approximates the optimal action–value function, then good actions satisfy \(a \approx \arg\max_a Q(s,a)\). With **continuous** actions, the maximization over \(a\) is replaced by **differentiation**: one trains \(\mu_\theta\) to **increase** \(Q_\phi(s,\mu_\theta(s))\) by gradient ascent on \(\theta\), while treating \(\phi\) as constant in that step.

In this repository, the “environment” is a **semi-discrete 1D heat equation** with **additive control** \(a(x,t)\). States are **functions on a space–time canvas** (causal temperature field plus coordinates), actions are **spatial control vectors** at the current time, and both **actor** and **critic** are **2D EFNOs** mapping fields on \((x,t)\) to either another field (control) or a scalar \(Q\) (after global pooling).

**Relation to Q-learning.** The critic is trained by **mean-squared Bellman error** against a **target network**, exactly as in Spinning Up’s presentation of DDPG. The policy update is the **deterministic policy gradient** through \(Q\), not tabular Q-learning.

---

## Quick facts

| | |
|--|--|
| **Algorithm family** | Off-policy actor–critic (DDPG-style) |
| **Action space** | Continuous: \(a(\cdot, t_n) \in \mathbb{R}^{N_x}\) with Dirichlet pins \(a(0,t)=a(L,t)=0\) |
| **State representation** | Tensor on \((x,t)\): causal \(u\), normalized \(x/L\), \(t/T\), and normalized time index \(n/(N_t-1)\) |
| **Function approximators** | Extended FNO (complex-frequency inverse), same construction as `burgers_bc_data/train_fno_burgers_bc.py` |
| **Exploration** | Gaussian noise on interior control components; optional decay of noise scale over episodes |
| **Stability** | Target actor and target critic updated by **Polyak averaging** toward the online networks |

---

## The controlled system (environment)

**PDE.** On \([0,L]\times[0,T]\),

\[
u_t = \alpha\, u_{xx} + a(x,t), \qquad u(0,t)=u(L,t)=0,
\]

with \(\alpha > 0\). The implementation uses **Crank–Nicolson** in time for the diffusion term and adds the control **explicitly** on the right-hand side each step.

**MDP.** Time is discretized into indices \(n = 0,\ldots,N_t-2\). At step \(n\):

- **State** \(s_n\): a \(4 \times N_x \times N_t\) tensor encoding the **causal** temperature field (columns \(t \le n\) show \(u(\cdot,t)\); future columns hold the value at \(n\)), plus broadcast \(x/L\), \(t/T\), and a channel filled with \(n/(N_t-1)\).
- **Action** \(a_n\): control values at spatial grid points at time \(t_n\) (boundary entries fixed to zero).
- **Transition:** one CN heat step produces \(u^{n+1}\).
- **Reward** (dense, per step):

\[
r_n = - w_{\mathrm{ctrl}}\, \mathbb{E}_x\big[ a_n(x)^2 \big] \;-\; w_{\mathrm{con}}\, \mathbb{E}_x\Big[ \mathrm{ReLU}\big( |u^{n+1}(x)| - u_{\max} \big)^2 \Big],
\]

where expectations are over **interior** spatial points. This is a **soft** state constraint: no hard projection; large \(w_{\mathrm{con}}\) discourages exceeding the threshold \(u_{\max}\) anywhere in space after each update.

**Terminal.** After step \(n = N_t-2\), the episode ends; bootstrapped targets use \((1-d)\) with \(d=1\) on this transition.

---

## Key equations

### Q-learning side (critic)

Let \(\mathcal{D}\) store transitions \((s, a, r, s', d, n)\) where \(n\) is the time index used to **slice** the actor’s full output \(a(x,t)\) (see below). Target networks \((\phi_{\mathrm{targ}}, \theta_{\mathrm{targ}})\) lag \((\phi,\theta)\).

**Bellman target** (same structure as Spinning Up DDPG):

\[
y = r + \gamma\, (1-d)\, Q_{\phi_{\mathrm{targ}}}\!\big(s',\, \tilde{a}'\big),
\qquad
\tilde{a}' = \mathrm{SliceTime}\!\big(\mu_{\theta_{\mathrm{targ}}}(s'), \, n{+}1\big).
\]

Here \(\mathrm{SliceTime}\) gathers the column of the policy output at time index \(n{+}1\) (clamped to the grid). For terminal \(s'\), \(d=1\) zeros the bootstrap term.

**Critic loss** (MSBE):

\[
L_Q(\phi) = \mathbb{E}_{(s,a,r,s',d,n)\sim\mathcal{D}}\Big[ \big( Q_\phi(s,a) - y \big)^2 \Big].
\]

**Q architecture.** The critic is an EFNO on **five** input channels: the four state channels plus **one** channel where the executed action \(a_n(x)\) is **broadcast** across all \(t\) (so the operator sees a consistent \((x,t)\) field). A small convolutional head maps EFNO features to one channel per \((x,t)\), then **spatial averaging** yields a **scalar** \(Q(s,a)\).

### Policy side (actor)

The **actor** is an EFNO mapping the **four** state channels to **one** control channel \(\hat{a}(x,t)\), followed by **\(\texttt{tanh}\)** scaling by a user cap \(a_{\max}\).

At decision time \(n\), the **executed** action is the **time slice**:

\[
a_n(x) = \hat{a}(x, t_n; \theta).
\]

**Policy objective** (deterministic policy gradient, Q treated as fixed w.r.t. \(\phi\) in this step):

\[
L_\mu(\theta) = - \mathbb{E}_{(s,n)\sim\mathcal{D}}\Big[ Q_\phi\big(s,\, \mathrm{SliceTime}(\mu_\theta(s), n)\big) \Big].
\]

Gradients flow through \(\mu_\theta\) into the same sliced action fed to \(Q_\phi\).

### Target network updates (Polyak)

After each optimizer step on \(\phi\) and \(\theta\), targets move **slowly** toward the online weights (Spinning Up’s “soft update”; in code the mixing coefficient is `tau`):

\[
\phi_{\mathrm{targ}} \leftarrow (1-\tau)\,\phi_{\mathrm{targ}} + \tau\,\phi,
\qquad
\theta_{\mathrm{targ}} \leftarrow (1-\tau)\,\theta_{\mathrm{targ}} + \tau\,\theta.
\]

Small \(\tau\) (e.g. `0.005`) corresponds to target networks that are **very sticky**, similar to Spinning Up’s \(\rho \approx 0.995\) notation when \(\tau = 1-\rho\).

---

## Exploration vs. exploitation

The behavioral policy is **deterministic EFNO mean + Gaussian noise** on **interior** components of \(a_n\); boundaries stay zero. The noise scale can **decay** over episodes (`--noise-init`, `--noise-decay`, `--noise-final`). At evaluation, use the actor mean without noise.

---

## Pseudocode (project-specific)

```
Input: EFNO actor μθ, EFNO critic Qφ, targets μθ_targ, Qφ_targ, buffer D, heat env, Polyak τ

Initialize θ_targ ← θ, φ_targ ← φ

For each training episode:
    Sample batch of initial conditions u0
    Roll out heat dynamics for n = 0 … Nt-2:
        Build state s from causal u-field and (x,t,n) channels
        a_full ← μθ(s)           # (B, 1, Nx, Nt)
        a ← column a_full[:,:,:,n] + Gaussian noise on interior
        Step CN solver → u^{n+1}; compute reward r; terminal flag d
        Build s'; store (s, a, r, s', d, n) in D

    For several minibatch updates:
        Sample (s, a, r, s', d, n) from D
        n' ← min(n+1, Nt-1)
        a' ← column μθ_targ(s')[:,:,:,n']
        y ← r + γ (1-d) Qφ_targ(s', a')

        φ ← φ - αQ ∇φ mean_s (Qφ(s,a) - y)²
        θ ← θ + αμ ∇θ mean_s Qφ(s, column μθ(s)[:,:,:,n])

        Soft-update φ_targ, θ_targ toward φ, θ with τ
```

---

## Implementation map

| Spinning Up concept | This codebase |
|----------------------|----------------|
| Environment step | `HeatEnv.step`, `step_reward` in `heat_env.py` |
| Replay buffer | `replay_buffer.py` (stores integer \(n\)) |
| Actor / critic | `ActorEFNO`, `CriticQEFNO` in `models.py`; spectral layers in `efno.py` |
| Training loop | `python -m heat_rl_efno.train_ddpg_heat_efno` |

**Hyperparameters** (CLI defaults in `train_ddpg_heat_efno.py`): discount \(\gamma\), Polyak \(\tau\), learning rates for actor and critic, EFNO width / layers / `k_max`, grid `nx`, `nt`, PDE \((L,T,\alpha)\), constraint weights \((w_{\mathrm{ctrl}}, w_{\mathrm{con}}, u_{\max})\), exploration noise schedule, buffer size, batch sizes.

---

## References

1. Lillicrap, T. P., *et al.* (2015). *Continuous control with deep reinforcement learning.* (DDPG.)
2. Fujimoto, S., *et al.* (2018). *Addressing function approximation error in actor-critic methods.* (TD3; optional improvement not implemented here.)
3. Li, Z., *et al.* (2021). *Fourier Neural Operator for Parametric Partial Differential Equations.*
4. Spinning Up: [Deep Deterministic Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/ddpg.html) — general exposition and equations this note parallels.

The **extended FNO** (complex-frequency inverse) in `efno.py` matches the operator used for supervised Burgers learning in `burgers_bc_data/train_fno_burgers_bc.py`.
