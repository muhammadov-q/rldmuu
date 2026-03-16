import numpy as np


class SimpleMDP:
    def __init__(self):
        self.num_states = 2
        self.num_actions = 2

        self.P = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.P[0, 0, 0] = 0.5
        self.P[0, 0, 1] = 0.5
        self.P[0, 1, 1] = 1.0
        self.P[1, 0, 1] = 1.0
        self.P[1, 1, 1] = 1.0

        self.R = np.zeros((self.num_states, self.num_actions))
        self.R[0, 0] = 5
        self.R[0, 1] = 10
        self.R[1, 0] = -1
        self.R[1, 1] = -1

    def backwards_induction(self, T):
        V = np.zeros((T + 1, self.num_states))

        for t in range(1, T + 1):
            for s in range(self.num_states):
                Q_s = self.R[s] + np.dot(self.P[s], V[t - 1])
                V[t, s] = np.max(Q_s)

        return V

    def value_iteration(self, gamma=0.9, epsilon=1e-6):
        V = np.zeros(self.num_states)
        delta = float("inf")

        while delta > epsilon:
            v_old = np.copy(V)
            for s in range(self.num_states):
                Q_s = self.R[s] + gamma * np.dot(self.P[s], V)
                V[s] = np.max(Q_s)
            delta = np.max(np.abs(v_old - V))

        return V

    def policy_evaluation_dp(self, policy, gamma=0.9, epsilon=1e-6):
        V = np.zeros(self.num_states)
        delta = float("inf")

        while delta > epsilon:
            v_old = np.copy(V)
            for s in range(self.num_states):
                a = policy[s]
                V[s] = self.R[s, a] + gamma * np.dot(self.P[s, a], V)
            delta = np.max(np.abs(v_old - V))

        return V

    def policy_evaluation_matrix(self, policy, gamma=0.9):
        P_pi = np.zeros((self.num_states, self.num_states))
        R_pi = np.zeros(self.num_states)

        for s in range(self.num_states):
            a = policy[s]
            P_pi[s] = self.P[s, a]
            R_pi[s] = self.R[s, a]

        I = np.eye(self.num_states)
        V = np.linalg.inv(I - gamma * P_pi).dot(R_pi)

        return V


if __name__ == "__main__":
    mdp = SimpleMDP()

    print("--- 1. Backwards Induction (Finite T) ---")
    V_finite = mdp.backwards_induction(T=5)
    for t in range(1, 6):
        print(f"T={t}: V(s1)={V_finite[t, 0]:.2f}, V(s2)={V_finite[t, 1]:.2f}")

    print("\n--- 2. Value Iteration (gamma=0.95) ---")
    V_inf = mdp.value_iteration(gamma=0.95)
    print(f"Converged Values: V(s1)={V_inf[0]:.2f}, V(s2)={V_inf[1]:.2f}")

    print("\n--- 3 & 4. Policy Evaluation (gamma=0.95) ---")
    policy_a1 = [0, 0]
    policy_a2 = [1, 1]

    print("Policy always a1 (DP):", mdp.policy_evaluation_dp(policy_a1, gamma=0.95))
    print(
        "Policy always a1 (Matrix):",
        mdp.policy_evaluation_matrix(policy_a1, gamma=0.95),
    )

    print("Policy always a2 (DP):", mdp.policy_evaluation_dp(policy_a2, gamma=0.95))
    print(
        "Policy always a2 (Matrix):",
        mdp.policy_evaluation_matrix(policy_a2, gamma=0.95),
    )
