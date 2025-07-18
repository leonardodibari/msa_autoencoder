import os
from Bio import SeqIO
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import collections


AA_to_indice = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
    '-': 20#, 'X': 21
}

hydrophobicity_dict = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4,
    'H': -0.2, 'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5,
    'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2,
    'W': -0.9, 'Y': -1.3
}

indice_to_AA = {
    indice: aa for aa, indice in AA_to_indice.items()
}


# Assuming AA_to_indice is a dictionary mapping amino acids to indices

def generation_dataset_single_file(fasta_path: str, limit: int = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Generates matrices X (one-hot encoded) and Y (class labels) from a single fasta file.
    
    Parameters:
    -----------
    fasta_path : str
        Path to the fasta file.
    limit : int or None
        Maximum number of sequences to read (None to read all).
    
    Returns:
    --------
    X : np.ndarray
        Matrix of one-hot encoded sequences.
    Y : np.ndarray
        Array of labels (all zeros since all sequences belong to one class).
    labels : dict
        Dictionary with label {0: 'single_class'}.
    """
    # Read all sequences from the file
    sequences = list(SeqIO.parse(fasta_path, "fasta"))
    if limit is not None:
        sequences = sequences[:limit]
    
    # Convert sequences to numeric indices (using AA_to_indice)
    numeric_seqs = [list(map(lambda x: AA_to_indice.get(x, AA_to_indice["-"]), seq.seq.replace("X", "-"))) for seq in sequences]
    
    X = []
    for seq in numeric_seqs:
        # One-hot encode: for each position, a list of length 22 with 1 at the AA index, 0 elsewhere
        X.append([1 if i == aa else 0 for aa in seq for i in range(len(AA_to_indice))])
    
    X = np.array(X)
    Y = np.zeros(len(X), dtype=int)  # All sequences belong to the same class 0
    labels = {0: "single_class"}
    
    return X, Y, labels

def generation_dataset(limite: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict[int: str]]:
    """
    Génére une matrice dont chaque ligne est une séquence.
    Les colonnes sont encodées par des blocs de taille 22 représentant un AA encodé en one hot.
    """
    np.random.seed(seed)
    random.seed(seed)

    X, Y, labels = [], [], {}

    for classe, fic in enumerate(os.listdir("data")):
        
        ss_domaine = fic.split("_has_")[1][:-4]
        labels[classe] = ss_domaine
        SEQS = list(SeqIO.parse(f"data/{fic}", "fasta"))
        SEQS = random.sample(SEQS, min(limite, len(SEQS)))
        numerique = [list(map(lambda x: AA_to_indice[x], ss.seq.replace("X", "-"))) for ss in SEQS]
        
        for seq in numerique:
            Y.append(classe)
            X.append([1 if i == aa else 0 for aa in seq for i in range(len(AA_to_indice))])

    return np.array(X), np.array(Y), labels

path_init = "/home/sadrin/Documents/code/leonardodibari/leonardodibari-Gen.jl-bbc6b32/scra/amino_mixed_pf13354_steps1000001_seqs500_T1.0p0.5/"

def generation_traj(limite: int, path_to_traj=path_init) -> tuple[np.ndarray, np.ndarray, dict[int: str]]:
    """
    Génére une matrice dont chaque ligne est une séquence.
    Les colonnes sont encodées par des blocs de taille 22 représentant un AA encodé en one hot.
    """
    #np.random.seed(seed)
    #random.seed(seed)

    X= []
    
    
    #SEQS = list(SeqIO.parse(path_to_traj + "equil_det_bal_pf13354_silico_chain_num_" + str(n) + "_T_1.0.mixedDNA", "fasta"))
    SEQS = list(SeqIO.parse(path_to_traj, "fasta"))[:limite]
    #SEQS = random.sample(SEQS, min(limite, len(SEQS)))
    numerique = [list(map(lambda x: AA_to_indice[x], ss.seq.replace("X", "-"))) for ss in SEQS]
        
    for seq in numerique:

        X.append([1 if i == aa else 0 for aa in seq for i in range(len(AA_to_indice))])

    return np.array(X)


def onehot(seq: str) -> list:
    """
    """
    seq =  [AA_to_indice[x] for x in seq]
    return [1 if i == aa else 0 for aa in seq for i in range(len(AA_to_indice))]


def plot_acp(X_pca: np.ndarray, Y: np.ndarray, fenetre: int, labels: dict) -> None:
    """
    Crée un plot qui affiche par deux les axes de l'ACP de la dimension 1 à fenetre+1 et la sauvegarde.
    """
    palette = sns.color_palette("tab10", len(labels))
    color_dict = {label: palette[i] for i, label in enumerate(labels)}
    colors = [color_dict[int(label)] for label in Y]

    fig, axes = plt.subplots(fenetre, fenetre, figsize=(fenetre*5, fenetre*5))

    for i in range(fenetre):
        for j in range(fenetre):
            if i != j:
                axes[i, j].scatter(X_pca[:, i], X_pca[:, j], c=colors, alpha=0.5, edgecolor="black")
                axes[i, j].set_xlabel(f'PC {i+1}')
                axes[i, j].set_ylabel(f'PC {j+1}')
            else:
                axes[i, j].axis('off')

    handles = [plt.Line2D([0], [0], marker='o', color='w', markersize=8, markerfacecolor=color_dict[label]) for label in labels]
    fig.legend(handles, [f"Classe {labels[label]}" for label in labels], title="Sous-domaines", loc="upper right")

    plt.tight_layout()
    plt.savefig(f"visualization/pca_{fenetre}_components.svg")
    plt.close()
    return


def composition_clusters(labels: dict[int: str], affectations: np.ndarray, Y: np.ndarray) -> dict[int: list[int]]:
    """
    Renvoie un dictionnaire affichant les labels par cluster.
    """
    composition = dict()
    for classe in np.unique(affectations):
        composition[classe] = [labels[Y[i]] for i in range(len(affectations)) if affectations[i] == classe]
    return composition


def plot_composition(algo: str, labels: dict[int: str], composition: dict[int: list[int]]) -> None:
    """
    Affiche la composition de chaque cluster.
    """
    sorted_labels = sorted(labels.items())
    label_values = [label for _, label in sorted_labels]

    _, axes = plt.subplots(1, len(composition), figsize=(len(composition)*10, 5))

    for i, key in enumerate(composition):
        axes[i].hist(composition[key], color="grey", alpha=0.5, edgecolor="black", width=0.5)
        axes[i].set_xlabel('Classe')
        axes[i].set_xticks(range(len(label_values)))
        axes[i].set_xticklabels(label_values, rotation=45)
        axes[i].set_ylabel('Fréquence')
        axes[i].set_title(f'Cluster({i})')

    plt.tight_layout()
    plt.savefig(f"visualization/{algo}_composition.svg")
    plt.close()
    return


def onehot2seq(X: np.ndarray) -> list[str]:
    """
    A partir d'une matrice one hot encodé recupère la liste des séquences associées.
    """
    seqs = []
    bloc = len(AA_to_indice)
    for vecteur in X:
        seq = ""
        for aa in range(0, vecteur.shape[0]//bloc):
            debut, fin = aa*bloc, aa*bloc + bloc
            indice = np.argmax(vecteur[debut: fin])
            seq += f"{indice_to_AA[indice]}"
        seqs.append(seq)
    return seqs


def hamming(seqA: str, seqB: str, distance: bool) -> float:
    """
    Renvoie le score d'un hamming entre deux séquences.
    Ce score est normalisé entre 0 et 1.
    Les deux séquences doivent être de la même taille.
    """
    score = 0
    for i in range(len(seqA)):
        if seqA[i] == seqB[i]:
            score += 1
    
    if distance:    return 1/(score + 1e-10)
    
    return score/len(seqA)


def batch_similarite(SEQtrain: list[str], Xtest: np.ndarray, Ytest: np.ndarray, SEQtest: list[str]):
    """
    Sépare un dataset de test en 5 sous ensemble de test en fonction du meilleur score d'alignement dans par rapport aux séquences de train.
    """
    Xtest90, Xtest80, Xtest70, Xtest60, Xtest50, Xtest40, Xtest30, Xtest20, Xtest10, Xtest0 = [], [], [], [], [], [], [], [], [], []
    Ytest90, Ytest80, Ytest70, Ytest60, Ytest50, Ytest40, Ytest30, Ytest20, Ytest10, Ytest0 = [], [], [], [], [], [], [], [], [], []

    for i, seqB in tqdm.tqdm(enumerate(SEQtest)):
        best_score = np.max([hamming(seqA, seqB, False)*100 for seqA in SEQtrain])

        if best_score >= 90:
            Xtest90.append(Xtest[i, :])
            Ytest90.append(Ytest[i])
        
        elif best_score >= 80:
            Xtest80.append(Xtest[i, :])
            Ytest80.append(Ytest[i])
        
        elif best_score >= 70:
            Xtest70.append(Xtest[i, :])
            Ytest70.append(Ytest[i])

        elif best_score >= 60:
            Xtest60.append(Xtest[i, :])
            Ytest60.append(Ytest[i])
        
        elif best_score >= 50:
            Xtest50.append(Xtest[i, :])
            Ytest50.append(Ytest[i])
        
        elif best_score >= 40:
            Xtest40.append(Xtest[i, :])
            Ytest40.append(Ytest[i])

        elif best_score >= 30:
            Xtest30.append(Xtest[i, :])
            Ytest30.append(Ytest[i])

        elif best_score >= 20:
            Xtest20.append(Xtest[i, :])
            Ytest20.append(Ytest[i])

        elif best_score >= 10:
            Xtest10.append(Xtest[i, :])
            Ytest10.append(Ytest[i])

        elif best_score >= 0:
            Xtest0.append(Xtest[i, :])
            Ytest0.append(Ytest[i])

    return [(np.array(Xtest90), np.array(Ytest90)), (np.array(Xtest80), np.array(Ytest80)), (np.array(Xtest70), np.array(Ytest70)), (np.array(Xtest60), np.array(Ytest60)), (np.array(Xtest50), np.array(Ytest50)), (np.array(Xtest40), np.array(Ytest40)), (np.array(Xtest30), np.array(Ytest30)), (np.array(Xtest20), np.array(Ytest20)), (np.array(Xtest10), np.array(Ytest10)), (np.array(Xtest0), np.array(Ytest0))]


def purete_score(Y: np.ndarray, Ypred: np.ndarray) -> float:
    """
    Calcule la pureté d'un clustering en fonction des étiquettes réelles (Y) 
    et des étiquettes des clusters (Ypred).
    """
    clusters = np.unique(Ypred)
    total_points = len(Y)
    purete = 0.0

    for cluster in clusters:
        cluster_indices = np.where(Ypred == cluster)[0]
        cluster_labels = Y[cluster_indices]  # Utilise Y, pas Ypred
        label_counts = collections.Counter(cluster_labels)
        purete += max(label_counts.values())

    return purete / total_points


def paire_distance(groupeA, groupeB, distance):
    """
    """
    D = np.zeros(shape=(len(groupeA), len(groupeB)), dtype=float)

    for i, seqA in enumerate(groupeA):
        for j, seqB in enumerate(groupeB):
            D[i, j] += distance(seqA, seqB, True)

    return D
