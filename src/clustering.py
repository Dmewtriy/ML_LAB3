"""Кластеризация синтетических игр Steam методами K-means и Уорда."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score, rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "screenshots"
SEED = 42
FEATURES = [
    "team_size", "price_usd", "file_size_gb", "metacritic_score",
    "wishlists_k",
]
LOG_FEATURES = ["team_size", "file_size_gb", "wishlists_k"]


def save_figure(fig, name):
    fig.savefig(FIGURES / name, dpi=170, bbox_inches="tight")
    plt.close(fig)


def histograms(data, columns, ncols, name):
    rows = (len(columns) + ncols - 1) // ncols
    fig, axes = plt.subplots(rows, ncols, figsize=(3.1 * ncols, 2.25 * rows))
    for column, ax in zip(columns, axes.flat):
        values = data[column]
        if values.nunique() <= 16:
            counts = values.value_counts().sort_index()
            ax.bar(counts.index, counts, width=0.8)
            if len(counts) == 2:
                ax.set_xticks(counts.index)
        else:
            ax.hist(values, bins=30)
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-4, 4))
        ax.set_title(column)
        ax.set_ylabel("Число игр")
    for ax in list(axes.flat)[len(columns):]:
        ax.set_visible(False)
    fig.tight_layout()
    save_figure(fig, name)


def correlation_heatmap(correlation):
    labels = {
        "team_size": "Команда", "price_usd": "Цена",
        "file_size_gb": "Объём игры", "metacritic_score": "Оценка критиков",
        "wishlists_k": "Желаемое", "languages_supported": "Языки",
        "dlc_count": "DLC", "trailer_views_k": "Трейлер",
        "is_indie": "Инди-игра",
    }
    columns = list(labels)
    values = correlation.loc[columns, columns].to_numpy()
    fig, ax = plt.subplots(figsize=(9.5, 5.0), layout="constrained")
    heatmap = ax.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ticks = range(len(columns))
    ax.set_xticks(ticks, labels.values(), rotation=27, ha="right")
    ax.set_yticks(ticks, labels.values())
    for (row, col), value in np.ndenumerate(values):
        number = f"{value:.2f}".replace("-0.00", "0.00").replace(".", ",")
        color = "white" if abs(value) >= 0.65 else "black"
        ax.text(col, row, number, ha="center", va="center", color=color)
    fig.colorbar(heatmap, ax=ax, label="Коэффициент Спирмена ρ")
    save_figure(fig, "08-spearman.png")


def boxplots(raw, transformed):
    fig, axes = plt.subplots(2, len(FEATURES), figsize=(10, 5.8))
    fliers = {"marker": "o", "markersize": 3, "markerfacecolor": "#b34c32",
              "markeredgecolor": "none", "alpha": 0.65}
    for row, values in enumerate([raw, transformed]):
        for col, feature in enumerate(FEATURES):
            ax = axes[row, col]
            ax.boxplot(values[feature], whis=1.5, flierprops=fliers)
            ax.set_xticks([])
            ax.grid(axis="y", alpha=0.25)
            if row == 0:
                ax.set_title(feature, fontsize=10)
    axes[0, 0].set_ylabel("До преобразования")
    axes[1, 0].set_ylabel("После преобразования")
    fig.tight_layout()
    save_figure(fig, "07-boxplots.png")


def outliers(data, factor):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    return (data < q1 - factor * iqr) | (data > q3 + factor * iqr)


def fit_kmeans(x, k):
    return KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(x)


def numbered_labels(labels, games):
    # Номера групп идут от меньшей средней команды к большей.
    order = games["team_size"].groupby(labels).mean().sort_values().index
    numbers = {old: new for new, old in enumerate(order, 1)}
    return pd.Series(labels).map(numbers)


def describe_clusters(name, x, games, labels, sample):
    print(f"\n{name}")
    silhouette = silhouette_score(x[sample], labels.iloc[sample])
    print(f"Silhouette: {silhouette:.4f}")
    print(f"Calinski–Harabasz: {calinski_harabasz_score(x, labels):.2f}")
    print(f"Davies–Bouldin: {davies_bouldin_score(x, labels):.4f}")
    print(f"Rand с hit: {rand_score(games['hit'], labels):.4f}")
    print(f"ARI с hit: {adjusted_rand_score(games['hit'], labels):.4f}")

    groups = games.reset_index(drop=True).groupby(labels)
    profiles = groups[FEATURES].mean()
    profiles["n"] = groups.size()
    profiles["hit_pct"] = groups["hit"].mean() * 100
    profiles["indie_pct"] = groups["is_indie"].mean() * 100
    print("\nСредние по исходным признакам и дополнительные характеристики:")
    print(profiles.round(2).to_string())

    centers = []
    for group in profiles.index:
        points = x[labels == group]
        center = points.mean(axis=0)
        centers.append(center)
        radius = np.linalg.norm(points - center, axis=1).mean()
        pair_distance = pdist(points).mean()
        print(f"Группа {group}: средний радиус {radius:.3f}, "
              f"среднее попарное расстояние {pair_distance:.3f}")
    print("Центры в пространстве стандартизованных признаков:")
    center_table = pd.DataFrame(centers, index=profiles.index, columns=FEATURES)
    print(center_table.round(3))
    print("Попарные расстояния между центрами:", pdist(centers).round(3))
    return np.array(centers)


def main():
    FIGURES.mkdir(exist_ok=True)
    data = pd.read_csv(ROOT / "dataset" / "steam_game_hits.csv")
    numeric = data.select_dtypes(include="number")
    print(f"Размер: {data.shape}, пропусков: {data.isna().sum().sum()}, "
          f"дубликатов: {data.duplicated().sum()}")
    print(data.dtypes.to_string())
    print(numeric.describe().T.round(2).to_string())
    print("Число жанров:", data["genre"].nunique())
    print("Метки hit:", data["hit"].value_counts().to_dict())
    print("\nКорреляции Спирмена:")
    correlation = numeric.corr(method="spearman")
    print(correlation.round(2).to_string())
    correlation_heatmap(correlation)

    histograms(data, FEATURES, 3, "01-histograms.png")
    other = [column for column in numeric if column not in FEATURES]
    histograms(data, other, 3, "02-other-histograms.png")

    transformed = data[FEATURES].astype(float)
    transformed[LOG_FEATURES] = np.log1p(transformed[LOG_FEATURES])
    boxplots(data, transformed)
    print("\nАсимметрия до и после логарифмирования:")
    skew = pd.DataFrame({
        "до": data[FEATURES].skew(), "после": transformed.skew(),
    })
    print(skew.round(2))
    print("\nЧисло выходов за границы IQR по каждому признаку:")
    print(pd.DataFrame({
        "1.5 IQR до": outliers(data[FEATURES], 1.5).sum(),
        "1.5 IQR после": outliers(transformed, 1.5).sum(),
        "3 IQR после": outliers(transformed, 3).sum(),
    }))
    keep = ~outliers(transformed, 3).any(axis=1)
    games = data.loc[keep].reset_index(drop=True)
    x = StandardScaler().fit_transform(transformed.loc[keep])
    print(f"Оставлено: {len(games)}, исключено: {(~keep).sum()}")

    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(x), size=2000, replace=False)
    matrix = scatter_matrix(pd.DataFrame(x[sample], columns=FEATURES),
                            figsize=(9, 8), diagonal="hist", alpha=0.27, s=9)
    save_figure(matrix[0, 0].figure, "03-scatter-matrix.png")

    # Дерево Уорда строится один раз, затем разрезается на разное число групп.
    tree = linkage(x, method="ward")
    scan = []
    for k in range(1, 9):
        model = fit_kmeans(x, k)
        sil_kmeans = sil_ward = np.nan
        if k > 1:
            ward = fcluster(tree, k, criterion="maxclust")
            sil_kmeans = silhouette_score(x[sample], model.labels_[sample])
            sil_ward = silhouette_score(x[sample], ward[sample])
        scan.append([k, model.inertia_, sil_kmeans, sil_ward])
    scan = pd.DataFrame(scan, columns=["k", "Инерция", "K-means", "Уорд"])
    scan = scan.set_index("k")
    print("\nПодбор k:")
    print(scan.round(4).to_string())
    k_kmeans = int(scan["K-means"].idxmax())
    k_ward = int(scan["Уорд"].idxmax())

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), layout="constrained")
    scan["Инерция"].plot(ax=axes[0], marker="o", title="Метод локтя K-means")
    sil = scan[["K-means", "Уорд"]].dropna()
    sil.plot(ax=axes[1], marker="o", title="Силуэтный анализ")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].set(xticks=range(1, 9), ylabel="Инерция")
    axes[1].set(xticks=range(2, 9), ylabel="Средний силуэт")
    save_figure(fig, "04-k-selection.png")

    fig, ax = plt.subplots(figsize=(8.5, 3.6), layout="constrained")
    cut = (tree[-k_ward, 2] + tree[-k_ward + 1, 2]) / 2
    dendrogram(tree, truncate_mode="lastp", p=12, color_threshold=cut, ax=ax)
    ax.axhline(cut, color="gray", ls="--", label=f"Срез при k={k_ward}")
    ax.set_xlabel("В скобках число игр в свёрнутой ветви")
    ax.set_ylabel("Высота слияния")
    ax.legend()
    save_figure(fig, "05-dendrogram.png")

    km = numbered_labels(fit_kmeans(x, k_kmeans).labels_, games)
    ward = fcluster(tree, k_ward, criterion="maxclust")
    ward = numbered_labels(ward, games)
    labels = {"K-means": km, "Уорд": ward}
    centers = {}
    for name, predicted in labels.items():
        centers[name] = describe_clusters(name, x, games, predicted, sample)
    print(f"\nARI между методами: {adjusted_rand_score(km, ward):.4f}")
    print(pd.crosstab(km, ward, rownames=["K-means"], colnames=["Уорд"]))

    # Проверка устойчивости: на каждой подвыборке заново строим оба разбиения.
    for run in range(1, 4):
        take = rng.choice(len(x), size=int(0.8 * len(x)), replace=False)
        take.sort()
        km_small = fit_kmeans(x[take], k_kmeans).labels_
        wards_small = linkage(x[take], method="ward")
        ward_small = fcluster(wards_small, k_ward, criterion="maxclust")
        ari_km = adjusted_rand_score(km.iloc[take], km_small)
        ari_ward = adjusted_rand_score(ward.iloc[take], ward_small)
        print(f"Подвыборка {run}: ARI K-means = {ari_km:.4f}, "
              f"ARI Уорд = {ari_ward:.4f}")

    pca = PCA(n_components=2)
    points = pca.fit_transform(x)
    variance = pca.explained_variance_ratio_ * 100
    print("\nДоля дисперсии PCA:", variance.round(2))
    fig, axes = plt.subplots(
        1, 2, figsize=(9.3, 4), sharex=True, sharey=True, layout="constrained"
    )
    for ax, (name, predicted) in zip(axes, labels.items()):
        for number in sorted(predicted.unique()):
            group = points[predicted == number]
            ax.scatter(
                group[:, 0], group[:, 1], s=7, alpha=0.25,
                label=f"Группа {number}",
            )
        projected = pca.transform(centers[name])
        ax.scatter(
            projected[:, 0], projected[:, 1], marker="X", s=85,
            c="black", label="Центр",
        )
        ax.set(title=name, xlabel=f"ГК1 ({variance[0]:.1f} %)")
        ax.legend()
    axes[0].set_ylabel(f"ГК2 ({variance[1]:.1f} %)")
    save_figure(fig, "06-pca.png")


if __name__ == "__main__":
    main()
