# Candidate Elimination Algorithm

def candidate_elimination(training_data):

    num_attributes = len(training_data[0]) - 1

    S = ['0'] * num_attributes
    G = [['?'] * num_attributes]

    for example in training_data:
        if example[-1] == 'Yes':
            S = example[:-1]
            break

    print("Initial S:", S)
    print("Initial G:", G)

    for example in training_data:
        attributes = example[:-1]
        label = example[-1]

        # POSITIVE EXAMPLE
        if label == 'Yes':
            for i in range(num_attributes):
                if attributes[i] != S[i]:
                    S[i] = '?'

            # Remove hypotheses from G that do not cover the example
            G = [g for g in G if is_consistent(g, attributes)]

        # NEGATIVE EXAMPLE
        else:
            new_G = []
            for g in G:
                if is_consistent(g, attributes):
                    specializations = specialize(g, S, attributes)
                    new_G.extend(specializations)
                else:
                    new_G.append(g)
            G = new_G

        print("\nAfter example:", example)
        print("S:", S)
        print("G:", G)

    return S, G


def is_consistent(hypothesis, instance):
    for h, i in zip(hypothesis, instance):
        if h != '?' and h != i:
            return False
    return True


def specialize(hypothesis, S, instance):
    specializations = []
    for i in range(len(hypothesis)):
        if hypothesis[i] == '?':
            if S[i] != instance[i]:
                new_h = hypothesis.copy()
                new_h[i] = S[i]
                specializations.append(new_h)
    return specializations

training_data = [
    ['Technical', 'Senior', 'Excellent', 'Good', 'Urban', 'Yes'],
    ['Technical', 'Junior', 'Excellent', 'Good', 'Urban', 'Yes'],
    ['Non Technical', 'Junior', 'Average', 'Poor', 'Rural', 'No'],
    ['Technical', 'Senior', 'Average', 'Good', 'Rural', 'No'],
    ['Technical', 'Senior', 'Excellent', 'Good', 'Rural', 'Yes']
  
]

S_final, G_final = candidate_elimination(training_data)

print("\nFinal Specific Hypothesis:", S_final)
print("Final General Hypotheses:", G_final)
