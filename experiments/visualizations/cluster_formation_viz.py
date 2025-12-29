import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Configuration for visual audit
FEATURES = ['MonthlyIncome', 'TotalWorkingYears']
DATA_SOURCE = Path("../../data/Attrition.csv")
OUTPUT_GIF = Path("../../outputs/kmeans_convergence")
K_CLUSTERS = 4

def prepare_analysis_data():
    path = DATA_SOURCE if DATA_SOURCE.exists() else Path("data/Attrition.csv")
    df = pd.read_csv(path)
    
    # Selecting primary drivers for segmentation
    X = df[FEATURES].values
    scaler = StandardScaler()
    return scaler.fit_transform(X)

class KMeansVisualizer:
    def __init__(self, data, k=4):
        self.data = data
        self.k = k
        self.centroids = self._init_centroids()
        self.history = []

    def _init_centroids(self):
        # Random initialization for demonstration
        indices = np.random.choice(len(self.data), self.k, replace=False)
        return self.data[indices]

    def _run_iteration(self):
        """Simulate one step of the Lloyd's algorithm."""
        # Assignment step
        distances = np.linalg.norm(self.data[:, np.newaxis] - self.centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Store state for animation
        self.history.append((self.centroids.copy(), labels))
        
        # Update step
        new_centroids = np.array([self.data[labels == i].mean(axis=0) for i in range(self.k)])
        
        if np.all(self.centroids == new_centroids):
            return False
        self.centroids = new_centroids
        return True

    def animate(self, frame):
        plt.cla()
        centroids, labels = self.history[min(frame, len(self.history)-1)]
        
        # Plot segmented data
        plt.scatter(self.data[:, 0], self.data[:, 1], c=labels, cmap='viridis', alpha=0.5, s=30)
        
        # Plot moving centroids
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, edgecolors='black', label='Centroids')
        
        plt.title(f"K-Means Convergence - Iteration {frame + 1}")
        plt.xlabel("Standardized Monthly Income")
        plt.ylabel("Standardized Working Years")
        plt.legend()

    def generate_gif(self, output_path):
        # Pre-calculate iterations
        for _ in range(20):
            if not self._run_iteration():
                break
                
        fig = plt.figure(figsize=(10, 7))
        anim = animation.FuncAnimation(fig, self.animate, frames=len(self.history), interval=500)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(f"{output_path}.gif", writer='pillow')
        plt.close()
        print(f"Cluster formation animation saved: {output_path}.gif")

if __name__ == "__main__":
    X_scaled = prepare_analysis_data()
    viz = KMeansVisualizer(X_scaled, k=K_CLUSTERS)
    viz.generate_gif(OUTPUT_GIF)