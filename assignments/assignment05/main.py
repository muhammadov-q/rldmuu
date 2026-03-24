"""
Problem Description: "The Student's Dilemma"

This script models a Finite Horizon Markov Decision Process (MDP) representing
a student deciding whether to study or relax in the days leading up to a final exam.

- Horizon (H): 3 time steps (e.g., 3 days before the exam).
- State Space (S): ['Unprepared', 'Prepared'] representing knowledge level.
- Action Space (A): ['Study', 'Relax']

- Transition Distribution (P):
    * Unprepared + Study -> 80% chance to become Prepared, 20% remain Unprepared.
    * Unprepared + Relax -> 100% chance to remain Unprepared.
    * Prepared + Study   -> 100% chance to remain Prepared.
    * Prepared + Relax   -> 70% chance to remain Prepared, 30% become Unprepared (forgetting).

- Reward Function (R):
    * Immediate: Study = -2 (requires effort), Relax = +2 (enjoyable).
    * Terminal (at the end of day 3): Prepared = +20 (passing), Unprepared = -10 (failing).
"""


class FiniteHorizonMDP:
    def __init__(
        self, states, actions, transition_probs, rewards, horizon, terminal_rewards
    ):
        self.states = states
        self.actions = actions
        self.P = transition_probs
        self.R = rewards
        self.H = horizon
        self.terminal_rewards = terminal_rewards

    def backward_induction(self):
        V = {t: {s: 0 for s in self.states} for t in range(self.H + 1)}
        policy = {t: {s: None for s in self.states} for t in range(self.H)}

        for s in self.states:
            V[self.H][s] = self.terminal_rewards[s]

        for t in range(self.H - 1, -1, -1):
            for s in self.states:
                best_val = float("-inf")
                best_action = None

                for a in self.actions:
                    q_val = self.R[s][a]
                    for s_prime in self.states:
                        q_val += self.P[s][a][s_prime] * V[t + 1][s_prime]

                    if q_val > best_val:
                        best_val = q_val
                        best_action = a

                V[t][s] = best_val
                policy[t][s] = best_action

        return policy, V


states = ["Unprepared", "Prepared"]
actions = ["Study", "Relax"]

transitions = {
    "Unprepared": {
        "Study": {"Prepared": 0.8, "Unprepared": 0.2},
        "Relax": {"Prepared": 0.0, "Unprepared": 1.0},
    },
    "Prepared": {
        "Study": {"Prepared": 1.0, "Unprepared": 0.0},
        "Relax": {"Prepared": 0.7, "Unprepared": 0.3},
    },
}

rewards = {
    "Unprepared": {"Study": -2, "Relax": 2},
    "Prepared": {"Study": -2, "Relax": 2},
}

terminal_rewards = {"Prepared": 20, "Unprepared": -10}

horizon = 3

student_mdp = FiniteHorizonMDP(
    states, actions, transitions, rewards, horizon, terminal_rewards
)
optimal_policy, optimal_values = student_mdp.backward_induction()

print("--- Optimal Policy (What to do at each time step) ---")
for t in range(horizon):
    print(f"Day {t}: {optimal_policy[t]}")

print("\n--- Value Function (Expected total reward from that point forward) ---")
for t in range(horizon + 1):
    print(f"Day {t}: {optimal_values[t]}")
