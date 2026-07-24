import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Plot Configuration
FEATURES = ['MonthlyIncome', 'TotalWorkingYears']
DATA_SOURCE = Path("../../data/Attrition.csv")
OUTPUT_GIF = Path("../../outputs/smote_process_viz")

def load_and_scale_data():
    path = DATA_SOURCE if DATA_SOURCE.exists() else Path("data/Attrition.csv")
    df = pd.read_csv(path)
    
    # Isolate minority class (Attrition = Yes)
    minority_df = df[df['Attrition'] == 'Yes']
    X = minority_df[FEATURES].values.astype(float)
    
    scaler = StandardScaler()
    return scaler.fit_transform(X), X

class SmoteVisualizer:
    def __init__(self, data, k_neighbors=5):
        self.data = data
        self.k = k_neighbors
        self.nn = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(data)
        self.synthetic_points = []

    def _generate_points(self):
        """Generator for SMOTE synthetic point creation."""
        for i in range(len(self.data)):
            # Find neighbors for the current point
            dist, indices = self.nn.kneighbors(self.data[i].reshape(1, -1))
            neighbor_idx = np.random.choice(indices[0][1:])
            
            # Linear interpolation logic
            diff = self.data[neighbor_idx] - self.data[i]
            new_point = self.data[i] + np.random.rand() * diff
            
            self.synthetic_points.append(new_point)
            yield np.array(self.synthetic_points), self.data[i], self.data[neighbor_idx], i

    def animate_step(self, frame_data):
        synthetic, current, neighbor, idx = frame_data
        plt.cla()
        
        # Plot original minority points
        plt.scatter(self.data[:, 0], self.data[:, 1], c='blue', label='Original Minority', alpha=0.5, s=30)
        
        # Plot synthetic points generated so far
        plt.scatter(synthetic[:, 0], synthetic[:, 1], c='red', marker='x', label='Synthetic (SMOTE)', s=40)
        
        # Draw line showing current interpolation
        plt.plot([current[0], neighbor[0]], [current[1], neighbor[1]], 'k--', alpha=0.3)
        
        plt.title(f"SMOTE Synthesis Iteration: {idx}")
        plt.xlabel("Standardized Feature 1")
        plt.ylabel("Standardized Feature 2")
        plt.legend(loc='upper right')

    def save_animation(self, filename):
        fig = plt.figure(figsize=(10, 7))
        anim = animation.FuncAnimation(fig, self.animate_step, self._generate_points, 
                                        save_count=len(self.data), interval=100)
        
        filename.parent.mkdir(parents=True, exist_ok=True)
        anim.save(f"{filename}.gif", writer='pillow', fps=5)
        plt.close()
        print(f"SMOTE validation GIF saved to: {filename}.gif")

if __name__ == "__main__":
    X_scaled, _ = load_and_scale_data()
    viz = SmoteVisualizer(X_scaled)
    viz.save_animation(OUTPUT_GIF)