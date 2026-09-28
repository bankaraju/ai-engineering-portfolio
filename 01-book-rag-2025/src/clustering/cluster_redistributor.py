"""Rebalance oversized paragraph clusters by splitting them into chunks of at
most max_cluster_size (size-capped redistribution). Paths are CLI arguments."""
import json
import os
import numpy as np
from collections import Counter, defaultdict
import logging

class ClusterRedistributor:
    def __init__(self, 
                 input_embeddings_path, 
                 input_clusters_path, 
                 output_dir,
                 max_cluster_size=30):
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            filename=os.path.join(output_dir, 'cluster_redistribution.log')
        )
        self.logger = logging.getLogger(__name__)
        
        # Paths and configuration
        self.input_embeddings_path = input_embeddings_path
        self.input_clusters_path = input_clusters_path
        self.output_dir = output_dir
        self.max_cluster_size = max_cluster_size
        
        # Load data
        self.embeddings = self._load_json(input_embeddings_path)
        self.paragraph_clusters = self._load_json(input_clusters_path)
    
    def _load_json(self, path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load {path}: {e}")
            raise
    
    def analyze_cluster_distribution(self):
        # Count paragraphs in each cluster
        cluster_counts = Counter(self.paragraph_clusters.values())
        
        self.logger.info("Cluster Distribution Analysis")
        self.logger.info(f"Total Clusters: {len(cluster_counts)}")
        self.logger.info(f"Min Paragraphs per Cluster: {min(cluster_counts.values())}")
        self.logger.info(f"Max Paragraphs per Cluster: {max(cluster_counts.values())}")
        self.logger.info(f"Average Paragraphs per Cluster: {np.mean(list(cluster_counts.values())):.2f}")
        
        # Identify oversized clusters
        oversized_clusters = {
            cluster: count 
            for cluster, count in cluster_counts.items() 
            if count > self.max_cluster_size
        }
        
        self.logger.info("\nOversized Clusters:")
        for cluster, count in sorted(oversized_clusters.items(), key=lambda x: x[1], reverse=True):
            self.logger.info(f"Cluster {cluster}: {count} paragraphs")
        
        return cluster_counts, oversized_clusters
    
    def redistribute_clusters(self):
        # Group paragraphs by their current cluster
        cluster_paragraphs = defaultdict(list)
        for pid, cluster in self.paragraph_clusters.items():
            cluster_paragraphs[cluster].append(pid)
        
        # Redistribution strategy
        new_clusters = {}
        current_cluster = 0
        
        for original_cluster, paragraphs in sorted(
            cluster_paragraphs.items(), 
            key=lambda x: len(x[1]), 
            reverse=True
        ):
            # Redistribute large clusters
            for i in range(0, len(paragraphs), self.max_cluster_size):
                batch = paragraphs[i:i+self.max_cluster_size]
                for pid in batch:
                    new_clusters[pid] = current_cluster
                current_cluster += 1
        
        return new_clusters
    
    def save_redistributed_clusters(self, redistributed_clusters):
        output_path = os.path.join(
            self.output_dir, 
            'redistributed_paragraph_clusters.json'
        )
        
        with open(output_path, 'w') as f:
            json.dump(redistributed_clusters, f, indent=2)
        
        self.logger.info(f"Redistributed clusters saved to {output_path}")
    
    def process(self):
        # Analyze current distribution
        cluster_counts, oversized_clusters = self.analyze_cluster_distribution()
        
        # Redistribute if needed
        if oversized_clusters:
            self.logger.info("Redistributing Oversized Clusters")
            redistributed_clusters = self.redistribute_clusters()
            self.save_redistributed_clusters(redistributed_clusters)
        else:
            self.logger.info("No redistribution needed. Cluster sizes are optimal.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Split oversized paragraph clusters")
    parser.add_argument("embeddings", help="JSON embeddings file (loaded, not used by the split)")
    parser.add_argument("clusters", help="JSON {paragraph_id: cluster_id}")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--max-cluster-size", type=int, default=30)
    args = parser.parse_args()
    redistributor = ClusterRedistributor(
        input_embeddings_path=args.embeddings,
        input_clusters_path=args.clusters,
        output_dir=args.output_dir,
        max_cluster_size=args.max_cluster_size
    )
    redistributor.process()

if __name__ == "__main__":
    main()
