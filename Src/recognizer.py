from hmmlearn import hmm
import numpy as np

# HMM model for the word "EV"
model_ev = hmm.MultinomialHMM(n_components=2)

model_ev.startprob_ = np.array([1.0, 0.0])

model_ev.transmat_ = np.array([
    [0.6, 0.4],
    [0.2, 0.8]
])

model_ev.emissionprob_ = np.array([
    [0.7, 0.3],
    [0.1, 0.9]
])


# HMM model for the word "OKUL"
model_okul = hmm.MultinomialHMM(n_components=2)

model_okul.startprob_ = np.array([1.0, 0.0])

model_okul.transmat_ = np.array([
    [0.5, 0.5],
    [0.3, 0.7]
])

model_okul.emissionprob_ = np.array([
    [0.6, 0.4],
    [0.2, 0.8]
])


# Test observation sequence
# High = 0
# Low = 1

test_sequence = np.array([[0, 1]]).T

score_ev = model_ev.score(test_sequence)
score_okul = model_okul.score(test_sequence)

print("EV score:", score_ev)
print("OKUL score:", score_okul)

if score_ev > score_okul:
    print("Recognized word: EV")
else:
    print("Recognized word: OKUL")
