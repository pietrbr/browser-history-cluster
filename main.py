import shutil
import sqlite3
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Path to Brave browser history
HISTORY_PATH = Path.home() / ".config/BraveSoftware/Brave-Browser/Default/History"

# Query configuration
DEFAULT_LIMIT = 20000  # Number of history entries to read

# Feature weighting (how much each component influences clustering)
DEFAULT_DOMAIN_WEIGHT = 2  # Weight for domain name
DEFAULT_PATH_WEIGHT = 1    # Weight for URL path
DEFAULT_TITLE_WEIGHT = 2   # Weight for page title

# Feature type weighting (controls importance of text vs numerical features)
TEXT_FEATURE_WEIGHT = 3.0      # Weight for all text features (TF-IDF from URL/title)
NUMERICAL_FEATURE_WEIGHT = 1.0  # Weight for numerical features (visit_count, typed_count, duration)
# Higher value = more importance in clustering
# Example: NUMERICAL_FEATURE_WEIGHT=2.0 means numerical features are 2x more important than text

# Clustering configuration
USE_NUMERICAL_FEATURES = True  # Include visit_count, typed_count, visit_duration
# True: cluster unique URLs (one per URL), False: cluster individual visits
CLUSTER_BY_URL = True
# URL mode: "youtube.com/video1" appears once with total stats
# Visit mode: "youtube.com/video1" appears multiple times if visited multiple times
MIN_CLUSTERS = 5               # Minimum number of clusters to test
MAX_CLUSTERS = 30              # Maximum number of clusters to test

# TF-IDF configuration
TFIDF_MAX_FEATURES = 150       # Maximum number of text features
TFIDF_NGRAM_RANGE = (1, 2)     # Use unigrams and bigrams
TFIDF_MIN_DF = 2               # Minimum document frequency
TFIDF_MAX_DF = 0.8             # Maximum document frequency (ignore too common terms)

# Display configuration
DISPLAY_TOP_N = 15             # Number of items to show per cluster

# ============================================================================


def copy_history_db(src_path):
    """Copy the history database to avoid locking issues"""
    temp_path = Path("/tmp/browser_history_copy.db")
    shutil.copy2(src_path, temp_path)
    return temp_path


def read_browser_history(db_path, limit=DEFAULT_LIMIT):
    """Read browser history from SQLite database with all available fields"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Extended query with all useful fields
    # Fields: id, url, title, visit_count, typed_count, last_visit_time, hidden,
    #         visit_id, visit_time, from_visit, transition, segment_id, visit_duration
    query = """
        SELECT 
            urls.id,
            urls.url,
            urls.title,
            urls.visit_count,
            urls.typed_count,
            urls.last_visit_time,
            urls.hidden,
            visits.id as visit_id,
            visits.visit_time,
            visits.from_visit,
            visits.transition,
            visits.segment_id,
            visits.visit_duration
        FROM urls
        LEFT JOIN visits ON urls.id = visits.url  -- LEFT JOIN: includes all URLs, even without visits
        ORDER BY visits.visit_time DESC
        LIMIT ?
    """

    cursor.execute(query, (limit,))
    results = cursor.fetchall()
    conn.close()

    return results


def aggregate_by_url(history_data):
    """Aggregate multiple visits to the same URL into a single record

    This converts visit-level data into URL-level data by:
    - Grouping all visits to the same URL
    - Summing visit counts and durations
    - Keeping the most recent title

    Args:
        history_data: List of visit records from read_browser_history()

    Returns:
        List of aggregated records (one per unique URL)

    Format of aggregated records:
    [0]=url_id, [1]=url, [2]=title, [3]=total_visit_count, [4]=total_typed_count,
    [5]=last_visit_time, [6]=hidden, [7]=None, [8]=most_recent_visit_time,
    [9]=None, [10]=None, [11]=None, [12]=total_visit_duration
    """
    from collections import defaultdict

    # Dictionary to aggregate data by URL
    url_data = defaultdict(lambda: {
        'url_id': None,
        'url': None,
        'title': None,
        'visit_count': 0,
        'typed_count': 0,
        'last_visit_time': 0,
        'hidden': 0,
        'most_recent_visit_time': 0,
        'total_duration': 0,
        'visit_times': []
    })

    # Aggregate visits by URL
    for record in history_data:
        url_id, url, title, visit_count, typed_count, last_visit_time, hidden, \
            visit_id, visit_time, from_visit, transition, segment_id, visit_duration = record

        if url is None:
            continue

        # Use URL as key
        key = url

        # Keep the first non-empty title we see
        if url_data[key]['title'] is None and title:
            url_data[key]['title'] = title

        # Aggregate data
        url_data[key]['url_id'] = url_id
        url_data[key]['url'] = url
        url_data[key]['visit_count'] = visit_count  # This is already total from urls table
        url_data[key]['typed_count'] = typed_count  # This is already total from urls table
        url_data[key]['last_visit_time'] = max(
            url_data[key]['last_visit_time'], last_visit_time or 0)
        url_data[key]['hidden'] = hidden

        # Track visit times and durations
        if visit_time:
            url_data[key]['visit_times'].append(visit_time)
            url_data[key]['most_recent_visit_time'] = max(
                url_data[key]['most_recent_visit_time'], visit_time)

        if visit_duration:
            url_data[key]['total_duration'] += visit_duration

    # Convert back to list format matching original structure
    aggregated = []
    for key, data in url_data.items():
        aggregated.append([
            data['url_id'],           # [0] id
            data['url'],              # [1] url
            data['title'],            # [2] title
            data['visit_count'],      # [3] visit_count (total from urls table)
            data['typed_count'],      # [4] typed_count (total from urls table)
            data['last_visit_time'],  # [5] last_visit_time
            data['hidden'],           # [6] hidden
            None,                     # [7] visit_id (not applicable for aggregated)
            data['most_recent_visit_time'],  # [8] most recent visit_time
            None,                     # [9] from_visit (not applicable)
            None,                     # [10] transition (not applicable)
            None,                     # [11] segment_id (not applicable)
            data['total_duration'],   # [12] total visit_duration across all visits
        ])

    # Sort by most recent visit time (descending)
    aggregated.sort(key=lambda x: x[8] if x[8] else 0, reverse=True)

    print(f"Aggregated {len(history_data)} visits into {len(aggregated)} unique URLs")

    return aggregated


def normalize_synonyms(text):
    """Normalize synonyms to group similar terms together"""
    # Define synonym mappings (target_word: [list of synonyms])
    synonym_map = {
        'aws': ['amazon', 'amazonaws'],
        'github': ['gh'],
        'google': ['goog'],
        'youtube': ['yt'],
        'linkedin': ['lnkd.'],
        'stackoverflow': ['stackexchange'],
        'documentation': ['docs', 'doc'],
        'tutorial': ['guide', 'howto'],
        'python': ['py'],
        'javascript': ['js'],
        'typescript': ['ts'],
    }

    # Convert to lowercase for case-insensitive matching
    text = text.lower()

    # Replace each synonym with the canonical term
    for canonical, synonyms in synonym_map.items():
        for synonym in synonyms:
            # Use word boundaries to avoid partial matches
            import re
            text = re.sub(r'\b' + synonym + r'\b', canonical, text)

    return text


def preprocess_url(url, domain_weight=2, path_weight=1):
    """Extract meaningful features from URL with weighted components

    Args:
        url: The URL to process
        domain_weight: How many times to repeat domain (higher = more weight)
        path_weight: How many times to repeat path (higher = more weight)
    """
    domain_names_to_remove = [
        'www.',
        '.com',
        '.org',
        '.gov',
        '.it',
        '.ai',
        '.io'
    ]
    parsed = urlparse(url)
    domain = parsed.netloc.replace('www.', '')
    domain = parsed.netloc.replace('.com', '')
    path = parsed.path.strip('/').replace('/', ' ')

    # Weight components by repetition
    # Repeating words makes them appear more frequently in TF-IDF
    weighted_domain = ' '.join([domain] * domain_weight)
    weighted_path = ' '.join([path] * path_weight)

    # Combine weighted components
    return f"{weighted_domain} {weighted_path}"


def prepare_features(history_data, domain_weight, path_weight, title_weight,
                     use_numerical_features=USE_NUMERICAL_FEATURES,
                     text_weight=TEXT_FEATURE_WEIGHT,
                     numerical_weight=NUMERICAL_FEATURE_WEIGHT):
    """Prepare TF-IDF features from history data with weighted components

    Args:
        history_data: Browser history records
        domain_weight: Weight for domain name
        path_weight: Weight for URL path
        title_weight: Weight for page title
        use_numerical_features: If True, include visit_count, typed_count, visit_duration
        text_weight: Weight multiplier for all text features
        numerical_weight: Weight multiplier for all numerical features

    Higher weights mean that component has more influence on clustering.
    For example, title_weight=3 means titles are 3x more important than paths.

    Text vs Numerical weighting:
    - text_weight=1.0, numerical_weight=2.0 means numerical features are 2x more important
    - text_weight=2.0, numerical_weight=1.0 means text features are 2x more important

    History data format (extended query):
    [0]=id, [1]=url, [2]=title, [3]=visit_count, [4]=typed_count, 
    [5]=last_visit_time, [6]=hidden, [7]=visit_id, [8]=visit_time, 
    [9]=from_visit, [10]=transition, [11]=segment_id, [12]=visit_duration
    """
    # Extract fields from extended query format
    urls = [item[1] for item in history_data]  # url is at index 1
    titles = [item[2] if item[2] else "" for item in history_data]  # title at index 2
    visit_counts = [item[3] if item[3] else 0 for item in history_data]  # visit_count at index 3
    typed_counts = [item[4] if item[4] else 0 for item in history_data]  # typed_count at index 4
    # visit_duration at index 12
    visit_durations = [item[12] if item[12] else 0 for item in history_data]

    # Create text features combining URL and title
    text_features = []
    for url, title in zip(urls, titles):
        processed_url = preprocess_url(url, domain_weight=domain_weight, path_weight=path_weight)

        # Weight title by repetition
        weighted_title = ' '.join([title] * title_weight)

        combined = f"{processed_url} {weighted_title}"

        # Apply synonym normalization
        combined = normalize_synonyms(combined)

        text_features.append(combined)

        if len(text_features) < 10:
            # visual check
            print(f"{url}======={processed_url}======={title}")

    # Vectorize using TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF
    )
    X_text = vectorizer.fit_transform(text_features)

    # Add numerical features if requested
    if use_numerical_features:
        # Normalize numerical features (convert microseconds to seconds, scale values)
        visit_counts_norm = np.array(visit_counts).reshape(-1, 1)
        typed_counts_norm = np.array(typed_counts).reshape(-1, 1)
        visit_durations_norm = np.array(visit_durations).reshape(-1, 1) / \
            1_000_000  # microseconds to seconds

        # Standardize numerical features (mean=0, std=1)
        scaler = StandardScaler()
        numerical_features = np.hstack([visit_counts_norm, typed_counts_norm, visit_durations_norm])
        X_numerical = scaler.fit_transform(numerical_features)

        # Apply weights to feature types
        # Convert sparse TF-IDF matrix to dense for weighting and concatenation
        X_text_dense = X_text.toarray()
        X_text_weighted = X_text_dense * text_weight
        X_numerical_weighted = X_numerical * numerical_weight

        # Combine weighted text and numerical features
        X_combined = np.hstack([X_text_weighted, X_numerical_weighted])

        print(f"\nFeature dimensions: {X_text_dense.shape[1]} text features (weight={text_weight}) + "
              f"{X_numerical.shape[1]} numerical features (weight={numerical_weight})")

        return X_combined, urls, titles
    else:
        # Even with only text features, apply text weight
        X_text_dense = X_text.toarray()
        X_text_weighted = X_text_dense * text_weight
        return X_text_weighted, urls, titles


def find_optimal_clusters(
    history_data, min_clusters=MIN_CLUSTERS,
    max_clusters=MAX_CLUSTERS,
    domain_weight=DEFAULT_DOMAIN_WEIGHT,
    path_weight=DEFAULT_PATH_WEIGHT,
    title_weight=DEFAULT_TITLE_WEIGHT,
    use_numerical_features=USE_NUMERICAL_FEATURES
):
    """Use elbow method to find optimal number of clusters"""
    print(f"\nFinding optimal number of clusters (testing {min_clusters}-{max_clusters})...")
    print(f"Weights: domain={domain_weight}, path={path_weight}, title={title_weight}")
    print(f"Using numerical features: {use_numerical_features}")

    # Prepare features
    X, urls, titles = prepare_features(history_data, domain_weight, path_weight, title_weight,
                                       use_numerical_features)

    # Adjust max clusters if we have less data
    max_clusters = min(max_clusters, len(history_data))
    min_clusters = min(min_clusters, max_clusters)

    # Calculate inertia (within-cluster sum of squares) for different k values
    inertias = []
    k_range = range(min_clusters, max_clusters + 1)

    for k in k_range:
        print(f"  Testing k={k}...", end=' ')
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        print(f"Inertia: {kmeans.inertia_:.2f}")

    # Calculate the rate of change (elbow detection)
    print("\nInertia decrease rate:")
    deltas = []
    for i in range(1, len(inertias)):
        delta = inertias[i-1] - inertias[i]
        delta_percent = (delta / inertias[i-1]) * 100
        deltas.append(delta)
        print(
            f"  k={list(k_range)[i-1]} -> k={list(k_range)[i]}: {delta:.2f} ({delta_percent:.1f}% decrease)")

    # Find the elbow point (where the decrease rate drops significantly)
    # Using second derivative approach
    if len(deltas) > 1:
        second_deltas = []
        for i in range(1, len(deltas)):
            second_delta = deltas[i-1] - deltas[i]
            second_deltas.append(second_delta)

        # The elbow is where the second derivative is maximum
        elbow_idx = second_deltas.index(max(second_deltas))
        optimal_k = list(k_range)[elbow_idx + 2]  # +2 because of the derivatives
    else:
        optimal_k = min_clusters

    print(f"\n{'='*80}")
    print(f"RECOMMENDED: k={optimal_k} clusters")
    print('='*80)

    # Plot the elbow curve
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
        plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
        plt.xlabel('Number of Clusters (k)', fontsize=12)
        plt.ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
        plt.title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        # Save the plot
        plot_path = Path('/tmp/elbow_plot.png')
        plt.savefig(plot_path, dpi=150)
        print(f"\nElbow plot saved to: {plot_path}")
        plt.show()
    except Exception as e:
        print(f"\nNote: Could not create plot: {e}")

    return optimal_k, inertias


def cluster_history_kmeans(history_data, n_clusters=10,
                           domain_weight=DEFAULT_DOMAIN_WEIGHT,
                           path_weight=DEFAULT_PATH_WEIGHT,
                           title_weight=DEFAULT_TITLE_WEIGHT,
                           use_numerical_features=USE_NUMERICAL_FEATURES):
    """Cluster browser history using K-means on TF-IDF features"""
    if not history_data:
        print("No history data to cluster")
        return

    # Prepare features
    X, urls, titles = prepare_features(history_data, domain_weight, path_weight, title_weight,
                                       use_numerical_features)

    # Adjust number of clusters if we have less data
    n_clusters = min(n_clusters, len(X.toarray()) if hasattr(X, 'toarray') else len(X))

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    return clusters, urls, titles


def cluster_history_gmm(history_data, n_clusters=10,
                        domain_weight=DEFAULT_DOMAIN_WEIGHT,
                        path_weight=DEFAULT_PATH_WEIGHT,
                        title_weight=DEFAULT_TITLE_WEIGHT,
                        use_numerical_features=USE_NUMERICAL_FEATURES):
    """Cluster browser history using Gaussian Mixture Model (soft clustering)

    GMM allows probabilistic cluster assignments - items can partially belong
    to multiple clusters. Good for pages that span multiple topics.
    """
    if not history_data:
        print("No history data to cluster")
        return

    # Prepare features
    X, urls, titles = prepare_features(history_data, domain_weight, path_weight, title_weight,
                                       use_numerical_features)

    # Adjust number of clusters if we have less data
    n_clusters = min(n_clusters, len(X))

    # Perform GMM clustering
    print(f"\nClustering with Gaussian Mixture Model (n_components={n_clusters})...")
    gmm = GaussianMixture(n_components=n_clusters, random_state=42, covariance_type='full')
    clusters = gmm.fit_predict(X)

    # Get probabilities for soft clustering
    probabilities = gmm.predict_proba(X)

    # Calculate cluster quality metrics
    bic = gmm.bic(X)
    aic = gmm.aic(X)
    print(f"GMM BIC: {bic:.2f}, AIC: {aic:.2f} (lower is better)")

    return clusters, urls, titles, probabilities


def cluster_history_hdbscan(history_data,
                            domain_weight=DEFAULT_DOMAIN_WEIGHT,
                            path_weight=DEFAULT_PATH_WEIGHT,
                            title_weight=DEFAULT_TITLE_WEIGHT,
                            use_numerical_features=USE_NUMERICAL_FEATURES,
                            min_cluster_size=5):
    """Cluster browser history using HDBSCAN (density-based, hierarchical)

    HDBSCAN automatically determines the number of clusters and marks outliers
    as noise (cluster -1). Good for finding natural groupings without
    specifying k in advance.

    Args:
        min_cluster_size: Minimum number of points in a cluster (default 5)
    """
    if not history_data:
        print("No history data to cluster")
        return

    # Prepare features
    X, urls, titles = prepare_features(history_data, domain_weight, path_weight, title_weight,
                                       use_numerical_features)

    # Perform HDBSCAN clustering
    print(f"\nClustering with HDBSCAN (min_cluster_size={min_cluster_size})...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1)
    clusters = clusterer.fit_predict(X)

    # Count clusters (excluding noise points marked as -1)
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)

    print(f"HDBSCAN found {n_clusters} clusters")
    print(f"Noise points (outliers): {n_noise} ({n_noise/len(clusters)*100:.1f}%)")

    # Cluster quality metrics
    if hasattr(clusterer, 'cluster_persistence_'):
        print("Cluster persistence scores available")

    return clusters, urls, titles


def find_optimal_hdbscan_params(history_data,
                                min_cluster_sizes=[50, 100, 200],
                                domain_weight=DEFAULT_DOMAIN_WEIGHT,
                                path_weight=DEFAULT_PATH_WEIGHT,
                                title_weight=DEFAULT_TITLE_WEIGHT,
                                use_numerical_features=USE_NUMERICAL_FEATURES):
    """Find optimal min_cluster_size parameter for HDBSCAN

    Tests different min_cluster_size values and returns clustering quality metrics

    Args:
        min_cluster_sizes: List of min_cluster_size values to test (default [50, 100, 200])
    """
    print(f"\nFinding optimal HDBSCAN parameters (testing min_cluster_size={min_cluster_sizes})...")

    # Prepare features
    X, urls, titles = prepare_features(history_data, domain_weight, path_weight, title_weight,
                                       use_numerical_features)

    results = []

    for min_size in min_cluster_sizes:
        print(f"\n  Testing min_cluster_size={min_size}...", end=' ')

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size, min_samples=1)
        clusters = clusterer.fit_predict(X)

        # Count clusters and noise
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = list(clusters).count(-1)
        noise_pct = n_noise / len(clusters) * 100

        # Calculate cluster sizes (excluding noise)
        from collections import Counter
        cluster_sizes = Counter([c for c in clusters if c != -1])

        avg_size = np.mean(list(cluster_sizes.values())) if cluster_sizes else 0
        min_cluster = min(cluster_sizes.values()) if cluster_sizes else 0
        max_cluster = max(cluster_sizes.values()) if cluster_sizes else 0

        results.append({
            'min_cluster_size': min_size,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_pct': noise_pct,
            'avg_cluster_size': avg_size,
            'min_cluster': min_cluster,
            'max_cluster': max_cluster,
            'clusters': clusters
        })

        print(f"Found {n_clusters} clusters, Noise: {n_noise} ({noise_pct:.1f}%)")

    # Print comparison table
    print(f"\n{'='*80}")
    print("HDBSCAN PARAMETER COMPARISON")
    print(f"{'='*80}")
    print(f"{'min_size':<12} {'Clusters':<10} {'Noise':<10} {'Noise %':<10} {'Avg Size':<10}")
    print("-" * 80)

    for r in results:
        print(f"{r['min_cluster_size']:<12} {r['n_clusters']:<10} {r['n_noise']:<10} "
              f"{r['noise_pct']:<10.1f} {r['avg_cluster_size']:<10.1f}")

    # Recommend parameter with lowest noise but reasonable number of clusters
    # Prefer fewer noise points while maintaining some granularity
    best = min(results, key=lambda x: x['noise_pct'])

    print(f"\n{'='*80}")
    print(f"RECOMMENDED: min_cluster_size={best['min_cluster_size']} "
          f"({best['n_clusters']} clusters, {best['noise_pct']:.1f}% noise)")
    print(f"{'='*80}")

    return best['min_cluster_size'], results


def compare_clustering_algorithms(history_data, n_clusters=10, hdbscan_min_size=None):
    """Compare K-means, GMM, and HDBSCAN on the same dataset

    Returns results from all three algorithms for comparison

    Args:
        n_clusters: Number of clusters for K-means and GMM (from elbow method)
        hdbscan_min_size: min_cluster_size for HDBSCAN (if None, uses parameter sweep)
    """
    print("\n" + "="*80)
    print("COMPARING CLUSTERING ALGORITHMS")
    print("="*80)

    results = {}

    # 1. K-means
    print("\n[1/4] Running K-means...")
    kmeans_result = cluster_history_kmeans(history_data, n_clusters=n_clusters)
    if kmeans_result:
        results['kmeans'] = {
            'clusters': kmeans_result[0],
            'urls': kmeans_result[1],
            'titles': kmeans_result[2],
            'n_clusters': n_clusters
        }

    # 2. GMM (uses same k as K-means)
    print("\n[2/4] Running Gaussian Mixture Model (using k={})...".format(n_clusters))
    gmm_result = cluster_history_gmm(history_data, n_clusters=n_clusters)
    if gmm_result:
        results['gmm'] = {
            'clusters': gmm_result[0],
            'urls': gmm_result[1],
            'titles': gmm_result[2],
            'probabilities': gmm_result[3],
            'n_clusters': n_clusters
        }

    # 3. HDBSCAN parameter sweep (if not specified)
    if hdbscan_min_size is None:
        print("\n[3/4] Finding optimal HDBSCAN parameters...")
        hdbscan_min_size, _ = find_optimal_hdbscan_params(history_data,
                                                          min_cluster_sizes=[50, 100, 200])

    # 4. HDBSCAN with optimal/specified parameter
    print(f"\n[4/4] Running HDBSCAN with min_cluster_size={hdbscan_min_size}...")
    hdbscan_result = cluster_history_hdbscan(history_data, min_cluster_size=hdbscan_min_size)
    if hdbscan_result:
        n_found = len(set(hdbscan_result[0])) - (1 if -1 in hdbscan_result[0] else 0)
        results['hdbscan'] = {
            'clusters': hdbscan_result[0],
            'urls': hdbscan_result[1],
            'titles': hdbscan_result[2],
            'n_clusters': n_found,
            'min_cluster_size': hdbscan_min_size
        }

    # Print comparison summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    print(f"\n{'Algorithm':<15} {'Clusters':<12} {'Noise':<12} {'Min Size':<10} {'Max Size':<10} {'Avg Size':<10}")
    print("-" * 80)

    for algo_name, result in results.items():
        clusters = result['clusters']
        n_clusters = result['n_clusters']

        # Calculate cluster sizes
        from collections import Counter
        cluster_sizes = Counter(clusters)

        # Remove noise cluster for HDBSCAN if present
        noise_info = ""
        if -1 in cluster_sizes:
            noise_count = cluster_sizes[-1]
            del cluster_sizes[-1]
            noise_info = f"{noise_count} ({noise_count/len(clusters)*100:.1f}%)"
        else:
            noise_info = "0 (0.0%)"

        avg_size = np.mean(list(cluster_sizes.values())) if cluster_sizes else 0
        min_size = min(cluster_sizes.values()) if cluster_sizes else 0
        max_size = max(cluster_sizes.values()) if cluster_sizes else 0

        algo_display = algo_name.upper()
        if algo_name == 'hdbscan' and 'min_cluster_size' in result:
            algo_display += f" (mcs={result['min_cluster_size']})"

        print(f"{algo_display:<15} {n_clusters:<12} {noise_info:<12} {min_size:<10} {max_size:<10} {avg_size:<10.1f}")

    print("\n" + "="*80)
    print("Notes:")
    print("  - K-means and GMM use optimal k from elbow method")
    print("  - HDBSCAN automatically determines number of clusters")
    print("  - HDBSCAN 'Noise' are outliers that don't fit any cluster well")
    print("="*80)

    return results


def display_clusters(clusters, urls, titles, history_data):
    """Display clustered results"""
    clustered_data = defaultdict(list)

    for idx, cluster_id in enumerate(clusters):
        clustered_data[cluster_id].append({
            'url': urls[idx],
            'title': titles[idx],
            'visit_count': history_data[idx][3],  # visit_count is at index 3 in query
            'visit_time': history_data[idx][8]     # visit_time is at index 8 in query
        })

    # Display clusters
    for cluster_id, items in sorted(clustered_data.items()):
        print(f"\n{'='*80}")
        print(f"CLUSTER {cluster_id + 1} ({len(items)} items)")
        print('='*80)

        # Show top N items in each cluster
        for item in items[:DISPLAY_TOP_N]:
            domain = urlparse(item['url']).netloc
            title = item['title'][:80] if item['title'] else "No title"
            print(f"  • {domain:30s} | {title:80s} | Visits: {item['visit_count']}")

        if len(items) > DISPLAY_TOP_N:
            print(f"  ... and {len(items) - DISPLAY_TOP_N} more items\n")


def main():
    print("Browser History Clustering")
    print("="*80)

    # Configuration option: set to True to compare all algorithms
    COMPARE_ALGORITHMS = True  # Set to False to use only K-means

    # Check if history file exists
    if not HISTORY_PATH.exists():
        print(f"Error: History file not found at {HISTORY_PATH}")
        return

    print(f"Reading history from: {HISTORY_PATH}")

    # Copy database to avoid locking issues
    temp_db = copy_history_db(HISTORY_PATH)

    try:
        # Read history
        history_data = read_browser_history(temp_db)
        print(f"Found {len(history_data)} history entries")

        if not history_data:
            print("No history data found")
            return

        # Aggregate by URL if configured
        if CLUSTER_BY_URL:
            print("\nClustering mode: Unique URLs (aggregating visits)")
            history_data = aggregate_by_url(history_data)
        else:
            print("\nClustering mode: Individual visits")

        if COMPARE_ALGORITHMS:
            # Compare all three algorithms
            # Find optimal k for K-means and GMM
            optimal_k, _ = find_optimal_clusters(history_data)

            # Run comparison
            results = compare_clustering_algorithms(history_data, n_clusters=optimal_k)

            # Display results for each algorithm
            for algo_name, result in results.items():
                print(f"\n{'='*80}")
                print(f"RESULTS FOR {algo_name.upper()}")
                print(f"{'='*80}")
                display_clusters(result['clusters'], result['urls'], result['titles'], history_data)
        else:
            # Standard K-means only
            # Find optimal number of clusters using elbow method
            optimal_k, _ = find_optimal_clusters(history_data)

            # Cluster the history with optimal k
            print(f"\nClustering history entries with k={optimal_k}...")
            clusters, urls, titles = cluster_history_kmeans(history_data, n_clusters=optimal_k)

            # Display results
            display_clusters(clusters, urls, titles, history_data)

    finally:
        # Clean up temporary file
        temp_db.unlink()
        print("\n" + "="*80)
        print("Clustering complete!")


if __name__ == "__main__":
    main()
