# Browser History Clustering

Automatically groups Brave browser history into meaningful clusters using machine learning.

## Core Functionality

Analyzes browsing history and discovers patterns by grouping similar pages together based on content, topics, and behavior.

## Key Features

### 1. Automatic Cluster Detection

Tests different numbers of clusters (5-30) and automatically finds the optimal grouping using the elbow method - identifies the point where adding more clusters provides diminishing returns.

### 2. Component Weighting

Fine-tune which parts of your browsing data matter most:

- **Domain weight**: Emphasize the website (e.g., github.com vs stackoverflow.com)
- **Path weight**: Emphasize specific pages within a site
- **Title weight**: Emphasize page titles

Example: Higher domain weight groups all GitHub pages together; higher path weight separates different sections.

### 3. Feature Type Balancing

Choose between content-based vs behavior-based clustering:

- **Text features**: What you visit (domain, path, page title)
- **Numerical features**: How you visit (frequency, typed URLs, time spent)

High text weight → groups by topic (all AWS documentation together)
High numerical weight → groups by behavior (frequently visited sites together)

### 4. Synonym Normalization

Treats related terms as equivalent so they cluster together:

- amazon = aws = amazonaws
- github = gh
- documentation = docs = doc

Prevents artificial separation of conceptually identical content.

### 5. Clustering Granularity

Two modes for different insights:

- **URL-level**: One entry per unique website (aggregates all visits)
- **Visit-level**: Each visit is separate (reveals temporal patterns)

URL mode answers "what topics do I explore?"
Visit mode answers "how do my browsing sessions look?"

## How It Works

1. Reads browser history from SQLite database
2. Normalizes synonyms to group related terms
3. Extracts features: domain, path, title, visit count, duration
4. Applies configurable weights to emphasize important features
5. Tests multiple cluster counts to find optimal grouping
6. Performs K-means clustering on weighted features
7. Displays results grouped by similarity

## Output

Each cluster shows related pages with domain, title, and visit count - revealing your browsing themes.
