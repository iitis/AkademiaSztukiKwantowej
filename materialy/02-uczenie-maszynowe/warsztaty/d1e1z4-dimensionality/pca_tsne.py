import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


"""

Funkcja do wyznaczania projekcji PCA i TSNE

"""


def compute_embeddings(X, n_components=2, random_state=0):
    pca = PCA(n_components=n_components, random_state=random_state)
    emb_pca = pca.fit_transform(X)
    tsne = TSNE(
        n_components=n_components,
        init="pca",
        learning_rate="auto",
        perplexity=30,
        random_state=random_state,
    )
    emb_tsne = tsne.fit_transform(X)
    return emb_pca, emb_tsne


"""

Wczytanie zbiorów i wyznaczenie projekcji

"""


X_digits, y_digits = load_digits(return_X_y=True)
emb_pca_digits, emb_tsne_digits = compute_embeddings(X_digits)
    
X_wine, y_wine = load_wine(return_X_y=True)
X_wine_scaled = StandardScaler().fit_transform(X_wine)
emb_pca_wine, emb_tsne_wine = compute_embeddings(X_wine_scaled)


"""

Wyswieltanie kolejnych scenariuszy

"""


def plot_single(emb, title, y=None):
    plt.figure(figsize=(4, 4))
    if y is None:
        plt.scatter(emb[:, 0], emb[:, 1], c='k', s=60, alpha=0.5)  
    else:
        plt.scatter(emb[:, 0], emb[:, 1], c=y, cmap='tab10', 
                    s=60, alpha=0.5)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.title(title)
    plt.tight_layout()
    plt.show()


plot_single(emb_pca_wine, 'Wine, PCA, bez klas')
plot_single(emb_pca_wine, 'Wine, PCA, z klasami', y_wine)
plot_single(emb_pca_digits, 'Digits, PCA, bez klas')
plot_single(emb_pca_digits, 'Digits, PCA, z klasami', y_digits)

plot_single(emb_tsne_wine, 'Wine, TSNE, bez klas')
plot_single(emb_tsne_wine, 'Wine, TSNE, z klasami', y_wine)
plot_single(emb_tsne_digits, 'Digits, TSNE, bez klas')
plot_single(emb_tsne_digits, 'Digits, TSNE, z klasami', y_digits)

