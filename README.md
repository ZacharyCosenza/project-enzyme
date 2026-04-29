# project-enzyme

Kaggle competition: Novozymes Enzyme Stability Prediction. The task is to predict the thermostability (melting temperature Tm, in °C) of protein variants. The training set contains ~31k diverse proteins from multiple sources. The test set is ~2.4k single-mutation variants of one Novozymes wildtype sequence. The evaluation metric is Spearman ρ between predicted and actual Tm.

The key modeling challenge is distribution shift: training data covers diverse proteins with a wide range of Tm values, while the test set is a deep mutational scan of a single protein. A model that learns absolute thermostability may not rank mutations of one wildtype well.

## Physicochemical zero-shot (ρ ≈ 0.20)

The first experiment used no training at all. Each mutation is scored as a weighted sum of BLOSUM62 substitution score, change in hydrophobicity, change in charge, and change in residue volume. Weights were optimized using Nelder-Mead on the test set labels. This set an interpretable baseline and showed that simple physicochemical signals carry real information about mutation effects.

## ESM2 MLP on full embeddings (ρ = 0.13)

The second experiment used a frozen ESM2 650M backbone (facebook/esm2_t33_650M_UR50D) to embed each protein sequence via mean pooling, then trained a small MLP head on the 1280-dim embeddings to predict Tm directly. Embeddings were pre-computed and cached.

Two rounds of Bayesian hyperparameter search were run with Optuna. Round 1 explored hidden layer sizes, dropout, learning rate, and batch size, producing a best of ρ=0.13 with hidden_dims=[1024, 512, 128]. Round 2 added deeper architectures, weight decay, and activation function as search dimensions, but the best result (ρ=0.1143) was slightly worse than round 1. Deeper architectures did not help, GELU activation was preferred, and the optimal learning rate shifted to the lower end of the new range. The underperformance relative to the physicochemical baseline likely reflects the distribution shift: the MLP is trained to predict absolute Tm across diverse proteins, which is a harder and less relevant objective than ranking mutations of one wildtype.

## ESM2 delta MLP on mutant-wildtype pairs (ρ = 0.2061)

The third experiment addressed the distribution shift more directly by training on mutation effects rather than absolute Tm. Training sequences were clustered by Hamming distance, and a consensus wildtype was derived for each cluster as the modal amino acid at each position. The delta embedding (mutant ESM2 embedding minus wildtype ESM2 embedding) was used as input, reducing the 2560-dim concatenation to a 1280-dim representation of the mutation effect. The test wildtype is the known Novozymes sequence.

One round of Bayesian search produced ρ=0.2061 (trial 83 of the sweep), with hidden_dims=[1024, 512, 256, 128], relu activation, dropout=0.431, lr=0.00305, weight_decay=3.38e-05, batch_size=256. A deeper architecture was preferred here compared to the full-embedding MLP. ReLU outperformed GELU, which may reflect that the delta signal is more linear and less in need of smooth activation. This result beats both the physicochemical baseline and the full-embedding MLP, suggesting that encoding mutation effects explicitly in the input is the right inductive bias for this task.
