# ---------------- CANDIDATE ELIMINATION ALGORITHM ----------------

class CandidateElimination:
    def __init__(self, domains):
        self.domains = domains
        self.n = len(domains)

        # Most specific hypothesis
        self.S = ['Φ'] * self.n

        # Most general hypotheses
        self.G = [['?'] * self.n]

    def is_consistent(self, h, x):
        return all(h[i] == '?' or h[i] == x[i] for i in range(self.n))

    def more_general(self, h1, h2):
        return all(h1[i] == '?' or h1[i] == h2[i] for i in range(self.n))

    # ---------------- GENERALIZE S ----------------
    def generalize_S(self, x):
        for i in range(self.n):
            if self.S[i] == 'Φ':
                self.S[i] = x[i]
            elif self.S[i] != x[i]:
                self.S[i] = '?'

    # ---------------- SPECIALIZE G ----------------
    def specialize_G(self, x):
        new_G = []

        for g in self.G:
            if self.is_consistent(g, x):  # g covers negative example
                for i in range(self.n):
                    if g[i] == '?':
                        for value in self.domains[i]:
                            if value != x[i]:
                                h = g.copy()
                                h[i] = value
                                if self.more_general(h, self.S):
                                    new_G.append(h)
            else:
                new_G.append(g)

        self.G = new_G

    # ---------------- PRUNE G ----------------
    def prune_G(self):
        self.G = [
            g for g in self.G
            if not any(self.more_general(g2, g) and g2 != g for g2 in self.G)
        ]

    # ---------------- UPDATE ----------------
    def update(self, x, label):
        if label == 'Yes':  # Positive example
            self.G = [g for g in self.G if self.is_consistent(g, x)]
            self.generalize_S(x)
        else:  # Negative example
            self.specialize_G(x)

        self.prune_G()

    def display(self, step):
        print(f"\nAfter Example {step}")
        print("S =", self.S)
        print("G =", self.G)


# ---------------- MAIN PROGRAM ----------------

data = [
    (['Technical','Senior','Excellent','Good','Urban'], 'Yes'),
    (['Technical','Junior','Excellent','Good','Urban'], 'Yes'),
    (['Non-Technical','Junior','Average','Poor','Rural'], 'No'),
    (['Technical','Senior','Average','Good','Rural'], 'No'),
    (['Technical','Senior','Excellent','Good','Rural'], 'Yes')
]

domains = [
    ['Technical', 'Non-Technical'],
    ['Junior', 'Senior'],
    ['Average', 'Excellent'],
    ['Poor', 'Good'],
    ['Urban', 'Rural']
]

ce = CandidateElimination(domains)

step = 1
for x, label in data:
    print(f"\nProcessing Example {step}: {x} -> {label}")
    ce.update(x, label)
    ce.display(step)
    step += 1

print("\nFINAL RESULT")
print("Final S:", ce.S)
print("Final G:", ce.G)
