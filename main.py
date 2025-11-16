import shutil
import sqlite3
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# Path to Brave browser history
HISTORY_PATH = Path.home() / ".config/BraveSoftware/Brave-Browser/Default/History"


def copy_history_db(src_path):
    """Copy the history database to avoid locking issues"""
    temp_path = Path("/tmp/browser_history_copy.db")
    shutil.copy2(src_path, temp_path)
    return temp_path


def read_browser_history(db_path, limit=10000, extended_query=False):
    """Read browser history from SQLite database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query to get URLs and visit times
    # Available fields from urls table:
    # - id, url, title, visit_count, typed_count, last_visit_time, hidden
    # Available fields from visits table:
    # - id, url (foreign key to urls.id), visit_time, from_visit, transition, segment_id, etc.
    if extended_query:
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
    else:
        query = """
            SELECT
                urls.url,
                urls.title,
                urls.visit_count,
                visits.visit_time
            FROM urls
            LEFT JOIN visits ON urls.id = visits.url  -- LEFT JOIN: includes all URLs, even without visits
            ORDER BY visits.visit_time DESC
            LIMIT ?
        """

    cursor.execute(query, (limit,))
    results = cursor.fetchall()
    conn.close()

    return results


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


def prepare_features(history_data, domain_weight, path_weight, title_weight):
    """Prepare TF-IDF features from history data with weighted components

    Args:
        history_data: Browser history records
        domain_weight: Weight for domain name
        path_weight: Weight for URL path
        title_weight: Weight for page title

    Higher weights mean that component has more influence on clustering.
    For example, title_weight=3 means titles are 3x more important than paths.
    """
    # Prepare data
    urls = [item[0] for item in history_data]
    titles = [item[1] if item[1] else "" for item in history_data]

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
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    X = vectorizer.fit_transform(text_features)

    return X, urls, titles


def find_optimal_clusters(history_data, min_clusters, max_clusters,
                          domain_weight=1, path_weight=1, title_weight=1):
    """Use elbow method to find optimal number of clusters"""
    print(f"\nFinding optimal number of clusters (testing {min_clusters}-{max_clusters})...")
    print(f"Weights: domain={domain_weight}, path={path_weight}, title={title_weight}")

    # Prepare features
    X, urls, titles = prepare_features(history_data, domain_weight, path_weight, title_weight)

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


def cluster_history(history_data, n_clusters=10,
                    domain_weight=1, path_weight=1, title_weight=1):
    """Cluster browser history using K-means on TF-IDF features"""
    if not history_data:
        print("No history data to cluster")
        return

    # Prepare features
    X, urls, titles = prepare_features(history_data, domain_weight, path_weight, title_weight)

    # Adjust number of clusters if we have less data
    n_clusters = min(n_clusters, len(X.toarray()))

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    return clusters, urls, titles


def display_clusters(clusters, urls, titles, history_data):
    """Display clustered results"""
    clustered_data = defaultdict(list)

    for idx, cluster_id in enumerate(clusters):
        clustered_data[cluster_id].append({
            'url': urls[idx],
            'title': titles[idx],
            'visit_count': history_data[idx][2],
            'visit_time': history_data[idx][3]
        })

    # Display clusters
    for cluster_id, items in sorted(clustered_data.items()):
        print(f"\n{'='*80}")
        print(f"CLUSTER {cluster_id + 1} ({len(items)} items)")
        print('='*80)

        # Show top 15 items in each cluster
        for item in items[:15]:
            domain = urlparse(item['url']).netloc
            title = item['title'][:80] if item['title'] else "No title"
            print(f"  • {domain:30s} | {title:80s} | Visits: {item['visit_count']}")

        if len(items) > 15:
            print(f"  ... and {len(items) - 15} more items\n")


def main():
    print("Browser History Clustering")
    print("="*80)

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

        # Find optimal number of clusters using elbow method
        optimal_k, _ = find_optimal_clusters(history_data, min_clusters=5, max_clusters=30,
                                             domain_weight=3, path_weight=1, title_weight=2)

        # Cluster the history with optimal k
        print(f"\nClustering history entries with k={optimal_k}...")
        clusters, urls, titles = cluster_history(history_data, n_clusters=optimal_k,
                                                 domain_weight=3, path_weight=1, title_weight=2)

        # Display results
        display_clusters(clusters, urls, titles, history_data)

    finally:
        # Clean up temporary file
        temp_db.unlink()
        print("\n" + "="*80)
        print("Clustering complete!")


if __name__ == "__main__":
    main()
