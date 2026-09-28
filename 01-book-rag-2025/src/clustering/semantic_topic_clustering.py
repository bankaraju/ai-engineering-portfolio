"""TF-IDF + KMeans clustering of sub-topic titles into named topic groups
(April 2025). Paths are CLI arguments."""
import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re
from collections import Counter

def clean_text(text):
    """Clean and normalize text for processing"""
    # Remove special characters, convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text

def extract_key_concepts(titles, top_n=3):
    """
    Extract most significant concepts across titles
    """
    # Combine all titles
    full_text = ' '.join(titles)
    cleaned_text = clean_text(full_text)
    
    # Custom investment-related keyword boosting
    investment_keywords = {
        'investment', 'stock', 'market', 'financial', 'risk', 
        'strategy', 'growth', 'portfolio', 'economic', 'value'
    }
    
    # Count words, giving extra weight to investment keywords
    words = cleaned_text.split()
    word_counts = Counter(words)
    
    # Boost investment-related keywords
    for keyword in investment_keywords:
        if keyword in word_counts:
            word_counts[keyword] *= 2
    
    # Get top concepts
    top_concepts = [word for word, _ in word_counts.most_common(top_n)]
    return top_concepts

def generate_cluster_name(cluster_titles):
    """
    Advanced cluster naming strategy
    """
    # Try multiple naming approaches
    naming_strategies = [
        lambda: f"Investment Insights: {' '.join(extract_key_concepts(cluster_titles))}",
        lambda: f"Strategic Approach: {' '.join(extract_key_concepts(cluster_titles))}",
        lambda: f"Financial Perspective: {' '.join(extract_key_concepts(cluster_titles))}"
    ]
    
    # Rotate through strategies to add variety
    from random import choice
    return choice(naming_strategies)()

def semantic_topic_clustering(
    subtopic_titles_path, 
    topic_map_path, 
    n_clusters=10
):
    # Load subtopic titles
    with open(os.path.expanduser(subtopic_titles_path), 'r') as f:
        subtopic_titles = json.load(f)
    
    # Load existing topic map for context
    with open(os.path.expanduser(topic_map_path), 'r') as f:
        existing_topics = json.load(f)
    
    # Incorporate existing topic titles as context
    context_titles = [topic['title'] for topic in existing_topics]
    return cluster_titles(subtopic_titles, context_titles, n_clusters)


def cluster_titles(subtopic_titles, context_titles=(), n_clusters=10):
    """Core TF-IDF + KMeans step. subtopic_titles: {id: title}."""
    # Extract titles and original IDs
    titles = list(subtopic_titles.values())
    original_ids = list(subtopic_titles.keys())
    all_titles = titles + list(context_titles)
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    title_vectors = vectorizer.fit_transform(all_titles)
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(title_vectors[:len(titles)])
    
    # Organize results
    semantic_topics = {}
    for cluster in range(n_clusters):
        # Find indices for this cluster
        cluster_indices = np.where(clusters == cluster)[0]
        
        # Collect titles for this cluster
        cluster_titles = [titles[i] for i in cluster_indices]
        
        # Generate cluster name
        cluster_name = generate_cluster_name(cluster_titles)
        
        # Create cluster details
        semantic_topics[int(cluster)] = {
            'cluster_name': cluster_name,
            'subtopics': [
                {
                    'id': original_ids[i],
                    'title': titles[i]
                } for i in cluster_indices
            ]
        }
    
    return semantic_topics

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cluster sub-topic titles with TF-IDF + KMeans")
    parser.add_argument("subtopic_titles", help="JSON object {id: title}")
    parser.add_argument("topic_map", help="JSON list of {title: ...} used as context")
    parser.add_argument("--output", default="semantic_topic_clusters.json")
    parser.add_argument("--clusters", type=int, default=10)
    args = parser.parse_args()
    output_path = args.output
    
    # Generate semantic topics
    semantic_topics = semantic_topic_clustering(
        args.subtopic_titles, 
        args.topic_map, 
        n_clusters=args.clusters
    )
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(semantic_topics, f, indent=2)
    
    # Print summary
    print("Semantic Topic Clusters Generated:")
    for cluster, details in semantic_topics.items():
        print(f"Cluster {cluster}: {details['cluster_name']}")
        print(f"  Subtopics: {len(details['subtopics'])}")
        print("  Sample Subtopics:")
        for subtopic in details['subtopics'][:3]:
            print(f"    - {subtopic['title']}")
        print()

if __name__ == '__main__':
    main()
