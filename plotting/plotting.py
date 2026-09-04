import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, Patch  # Add this import at the top
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_train_accuracy(csv_file, output_file):
    df_train = pd.read_csv(csv_file, header=None, names=["epoch", "accuracy"])
    print(4)
    plt.figure(figsize=(10, 6))
    plt.plot(df_train['epoch'], df_train['accuracy'], label='Training Accuracy', color='blue', marker='o')
    print(5)
    plt.title('Training Accuracy per Epoch', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True)
    plt.legend()
    print(6)
    #os.makedirs(os.path.dirname(output_file), exist_ok=True) 
    plt.savefig(output_file, format='jpeg')
    plt.close()  
    print(f"Training accuracy plot saved to {output_file}")


    
def plot_test_accuracy(csv_file, output_file):
    df_test = pd.read_csv(csv_file)

    plt.figure(figsize=(10, 6))
    plt.plot(df_test['epoch'], df_test['accuracy'], label='Testing Accuracy', color='green', marker='s')

    plt.title('Testing Accuracy per Epoch', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True)
    plt.legend()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)  
    plt.savefig(output_file, format='jpeg')
    plt.close()  
    print(f"Testing accuracy plot saved to {output_file}")



from pathlib import Path

class IncrementalTrainingVisualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Colorblind-friendly
    
    def load_data(self, input_file):
        """Bulletproof data loader for your exact format"""
        df = pd.read_csv(input_file)
        df.columns = ['classes', 'iterations']  # Ensure consistent column names
        df['cumulative'] = df['iterations'].cumsum()
        df['change'] = -df['iterations'].diff().fillna(0)  # Negative of difference
        return df

    def plot_inc_training(self, csv_file, output_file):
        # Read the CSV file
        data = pd.read_csv(csv_file)  # Replace 'data.csv' with your file name

        # Extract the columns
        x = data.iloc[:, 0]  # First column for x-axis
        y = data.iloc[:, 1]  # Second column for y-axis

        plt.figure(figsize=(8, 5))  # Set the figure size
        plt.fill_between(x, y, step="pre", color='blue', alpha=0.4, label='Iterations')  # Filled step plot
        plt.plot(x, y, drawstyle='steps-pre', color='blue', linewidth=2)  # Add step lines for clarity
        plt.xlabel('Number of Classes')  # Label for x-axis
        plt.ylabel('Iterations')  # Label for y-axis
        plt.title('Iterations vs Number of classes')  # Title of the plot
        plt.grid(True, linestyle='--', alpha=0.7)  # Add a grid for better readability
        plt.legend() 
        os.makedirs(os.path.dirname(output_file), exist_ok=True)  
        plt.savefig(output_file, format='jpeg')
        plt.close()  
        print(f"Testing accuracy plot saved to {output_file}")

    def plot_inc_training_shapes2(self, csv_file, output_file):
        """Visualize training phases as clean rectangles without mid-text"""
        
        # Load data
        df = pd.read_csv(csv_file)
        df.columns = ['classes', 'iterations']
        df['cum_classes'] = df['classes'].cumsum()
        
        plt.figure(figsize=(14, 8))
        ax = plt.gca()
        
        # Blue-purple-red gradient (no yellow)
        colors = plt.cm.RdPu(np.linspace(0.2, 0.9, len(df)))
        
        # Plot rectangles
        x_start = 0
        for idx, row in df.iterrows():
            rect = plt.Rectangle(
                (x_start, 0), 
                width=row['classes'],
                height=row['iterations'],
                facecolor=colors[idx],
                edgecolor='white',
                linewidth=0.5,
                alpha=0.8
            )
            ax.add_patch(rect)
            
            # Add label ONLY at the top edge (no middle text)
            """if row['iterations'] > max(df['iterations']) * 0.1:  # Only label significant steps
                ax.text(
                    x_start + row['classes']/2,
                    row['iterations'] * 1.02,  # Just above the rectangle
                    f"{row['classes']}c/{row['iterations']}i",
                    ha='center',
                    va='bottom',
                    color='black',
                    fontsize=8,
                    fontweight='normal'
                )"""
            
            x_start += row['classes']
        
        # Axis formatting
        ax.set_xlim(0, df['cum_classes'].max())
        ax.set_ylim(0, df['iterations'].max() * 1.15)
        
        # Subtle reference lines
        for y in range(0, int(df['iterations'].max()) + 5, 5):
            ax.axhline(y, color='gray', linestyle=':', alpha=0.2, linewidth=0.7)
        
        plt.title("Incremental Training Phases", pad=20)
        plt.xlabel("Total Classes Trained", labelpad=12)
        plt.ylabel("Iterations Required", labelpad=12)
        
        # Add phase indicator without colorbar
        for idx in [0, len(df)//2, len(df)-1]:
            plt.plot([], [], color=colors[idx], 
                    label=f"Phase {idx+1}", 
                    linewidth=8)
        plt.legend(loc='upper right', framealpha=0.9)
        
        plt.grid(True, axis='y', alpha=0.15)
        self._save(output_file)

    def plot_inc_training_shapes3(self, csv_file, output_file):
        """Visualize training phases with enhanced bar visibility and better colors"""
        
        # Load data
        df = pd.read_csv(csv_file)
        df.columns = ['classes', 'iterations']
        df['cum_classes'] = df['classes'].cumsum()
        
        plt.figure(figsize=(16, 9))  # Slightly larger figure
        ax = plt.gca()
        
        # Improved color palette - sequential blue-purple with better contrast
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(df)))
        
        # Calculate bar width scaling factor
        max_width = df['classes'].max()
        min_width = df['classes'].min()
        width_scale = 0.5 + 1.5 * (df['classes'] - min_width) / (max_width - min_width)
        
        # Plot rectangles with variable width emphasis
        x_start = 0
        for idx, row in df.iterrows():
            rect = plt.Rectangle(
                (x_start, 0), 
                width=row['classes'],
                height=row['iterations'],
                facecolor=colors[idx],
                edgecolor='black',  # Darker border for clarity
                linewidth=1.5 * width_scale[idx],  # Scale border width with bar size
                alpha=0.85,
                hatch='//' if idx % 2 == 0 else None  # Alternate patterns for extra distinction
            )
            ax.add_patch(rect)
            
            # Label only significant phases (top 30%)
            """if row['iterations'] > df['iterations'].quantile(0.7):
                ax.text(
                    x_start + row['classes']/2,
                    row['iterations'] * 1.05,
                    f"{row['classes']} classes\n{row['iterations']} iters",
                    ha='center',
                    va='bottom',
                    color='black',
                    fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)
                )"""
            
            x_start += row['classes']
        
        # Enhanced axis formatting
        ax.set_xlim(0, df['cum_classes'].max() * 1.02)
        ax.set_ylim(0, df['iterations'].max() * 1.25)
        
        # Improved reference lines
        for y in range(0, int(df['iterations'].max()) + 10, 10):
            ax.axhline(y, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        
        # Title and labels with better styling
        plt.title("Incremental Training Phases", pad=25, fontsize=14, fontweight='bold')
        plt.xlabel("Total Classes Trained", labelpad=15, fontsize=12)
        plt.ylabel("Iterations Required", labelpad=15, fontsize=12)
        
        # Legend with better visibility
        legend_elements = []
        for idx in [0, len(df)//3, 2*len(df)//3, len(df)-1]:
            legend_elements.append(
                Patch(facecolor=colors[idx],
                    edgecolor='black',
                    label=f"Phase {idx+1} ({df.iloc[idx]['classes']} classes)",
                    linewidth=2)
            )
        plt.legend(handles=legend_elements, 
                loc='upper right', 
                framealpha=0.95,
                fontsize=10)
        
        # Grid and background improvements
        ax.set_facecolor('#f8f8f8')
        plt.grid(True, axis='y', alpha=0.2, linestyle='-')
        
        # Add subtle x-axis markers for class boundaries
        for cum_class in df['cum_classes']:
            ax.axvline(cum_class, color='gray', linestyle=':', alpha=0.15, linewidth=0.5)
        
        self._save(output_file)

    def plot_inc_training_shapes(self, csv_file, output_file):
        # Read the data from the CSV file
        data = pd.read_csv(csv_file)  # Replace 'data.csv' with your file name

        num_classes = data.iloc[:, 0]  # First column for x-axis (number of classes)
        iterations = data.iloc[:, 1]  # Second column for y-axis (iterations per increment)

        # Dynamically adjust figure size based on the number of entries
        num_entries = len(num_classes)
        fig_width = max(10, num_entries * 0.5)  # Adjust width based on the number of entries
        fig_height = 6  # Fixed height for better proportions
        plt.figure(figsize=(fig_width, fig_height))  # Set the figure size dynamically

        ax = plt.gca()  # Get the current axes

        x_start = 0  # Starting x-position for the first rectangle
        for i in range(num_entries):
            width = num_classes[i]  # Width of the rectangle (number of classes)
            height = iterations[i]  # Height of the rectangle (iterations)
            
            # Add the rectangle to the plot
            rect = Rectangle((x_start, 0), width, height, edgecolor='blue', facecolor='blue', alpha=0.4)
            ax.add_patch(rect)
            
            # Add text inside the rectangle
            #fontsize = max(8, 12 - num_entries * 0.1)  # Adjust font size dynamically
            #ax.text(x_start + width / 2, height / 2, f'{iterations[i]}', 
            #        ha='center', va='center', color='white', fontsize=fontsize)
            
            x_start += width  # Update the starting x-position for the next rectangle

        # Set axis limits and labels
        ax.set_xlim(0, x_start)  # Set x-axis limit
        ax.set_ylim(0, max(iterations) + 1)  # Set y-axis limit
        ax.set_xlabel('Number of Classes', fontsize=12)  # Label for x-axis
        ax.set_ylabel('Iterations (Epochs)', fontsize=12)  # Label for y-axis
        ax.set_title('Incremental Training: Iterations vs Number of Classes', fontsize=14)  # Title of the plot

        # Add a grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)

        # Ensure the output directory exists and save the plot
        os.makedirs(os.path.dirname(output_file), exist_ok=True)  
        plt.savefig(output_file, format='jpeg', dpi=300)  # Save with high resolution
        plt.close()  
        print(f"Plot saved to {output_file}")


    def plot_iteration_curve(self, input_file, output_file):
        """Primary decreasing iteration curve"""
        df = self.load_data(input_file)
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['classes'], df['iterations'], 
                marker='o', color=self.colors[0], linewidth=2)
        plt.title("Training Iterations vs Class Count", pad=20)
        plt.xlabel("Number of Classes", labelpad=10)
        plt.ylabel("Iterations Required", labelpad=10)
        plt.grid(True, alpha=0.3)
        self._save(output_file)

    def plot_log_comparison(self, input_file, output_file):
        """Log-linear comparison"""
        df = self.load_data(input_file)
        
        plt.figure(figsize=(12, 6))
        # Linear scale
        plt.plot(df['classes'], df['iterations'], 
                marker='o', color=self.colors[0], 
                label='Linear Scale', alpha=0.7)
        # Log scale
        plt.plot(df['classes'], df['iterations'], 
                '--', color=self.colors[1], 
                label='Log Scale')
        plt.yscale('log')
        plt.title("Logarithmic View of Training Efficiency", pad=20)
        plt.xlabel("Number of Classes", labelpad=10)
        plt.ylabel("Log(Iterations)", labelpad=10)
        plt.legend()
        self._save(output_file)

    def plot_cumulative_effort(self, input_file, output_file):
        """Total training effort accumulation"""
        df = self.load_data(input_file)
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(df['classes'], df['cumulative'], 
                        color=self.colors[2], alpha=0.2)
        plt.plot(df['classes'], df['cumulative'], 
                marker='o', color=self.colors[2], linewidth=2)
        plt.title("Cumulative Training Effort", pad=20)
        plt.xlabel("Number of Classes", labelpad=10)
        plt.ylabel("Total Iterations", labelpad=10)
        self._save(output_file)

    def plot_effort_reduction(self, input_file, output_file):
        """How much easier each increment becomes"""
        df = self.load_data(input_file)
        
        plt.figure(figsize=(12, 6))
        plt.bar(df['classes'], df['change'], 
               width=8, color=self.colors[3], alpha=0.7)
        plt.title("Effort Reduction per Additional Classes", pad=20)
        plt.xlabel("Number of Classes", labelpad=10)
        plt.ylabel("Iterations Saved", labelpad=10)
        plt.grid(axis='y', alpha=0.2)
        self._save(output_file)
        
    def plot_effort_reduction3(self, input_file, output_file):
        """Effort reduction with highly visible width scaling"""
        df = self.load_data(input_file)
        
        plt.figure(figsize=(16, 8))
        
        # Dramatic width scaling (min 2 units, max 40 units)
        min_width = 2
        max_width = 40
        widths = min_width + (max_width - min_width) * (df['classes'] / df['classes'].max())
        
        # Color gradient from red (high effort) to green (low effort)
        colors = plt.cm.RdYlGn(1 - df['change'] / df['change'].max())
        
        bars = plt.bar(df['classes'], df['change'], 
                    width=widths, color=colors,
                    edgecolor='black', linewidth=0.7,
                    alpha=0.8)
        
        # Annotate every 5th bar for clarity
        for i, bar in enumerate(bars):
            if i % 5 == 0 or bar.get_height() > 1:
                plt.text(bar.get_x() + bar.get_width()/2, 
                        bar.get_height() + 0.1,
                        f"{bar.get_height():.1f}\n({int(df['classes'].iloc[i])} cls)",
                        ha='center', va='bottom', 
                        fontsize=8, linespacing=1.2)
        
        plt.title("Training Effort Reduction (Bar Width ∝ Class Count)", pad=25, fontsize=14)
        plt.xlabel("Number of Classes", labelpad=15, fontsize=12)
        plt.ylabel("Iterations Saved", labelpad=15, fontsize=12)
        
        # Custom grid and aesthetics
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.gca().set_axisbelow(True)
        
        # Add reference line at y=0
        plt.axhline(0, color='black', linewidth=0.5, alpha=0.5)
        
        self._save(output_file)

    def plot_effort_per_class(self, input_file, output_file):
        """Shows iterations per individual class"""
        df = self.load_data(input_file)
        
        # Calculate effort per class (iterations / class count)
        df['effort_per_class'] = df['iterations'] / df['classes']
        
        plt.figure(figsize=(14, 7))
        
        # Use log scale because values decrease exponentially
        plt.semilogy(df['classes'], df['effort_per_class'], 
                    marker='o', color=self.colors[1], linewidth=2)
        
        plt.title("Training Effort per Individual Class (Log Scale)", pad=20)
        plt.xlabel("Total Number of Classes", labelpad=10)
        plt.ylabel("Iterations per Class (Log Scale)", labelpad=10)
        plt.grid(True, which='both', alpha=0.2)
        
        # Highlight key points
        for _, row in df[::5].iterrows():  # Every 5th point
            plt.text(row['classes'], row['effort_per_class']*1.2,
                    f"{row['effort_per_class']:.4f}",
                    ha='center', fontsize=8)
        
        self._save(output_file)

    def _save(self, output_path):
        """Guaranteed save operation"""
        Path(output_path).parent.mkdir(exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Plot saved to {output_path}")

class TrainingVisualizer:
    def __init__(self):

        self.cmap = LinearSegmentedColormap.from_list(
            'improved_plasma', 
            ['#2a03a8', '#a83279', '#f39b1e', '#fef200']
        )

    def plot_heatmap(self, df, output_file=None, figsize=(14, 8)):
        # Pivot data for heatmap
        heatmap_data = df.pivot(index='number_of_classes', columns='epoch', values='accuracy')
        
        plt.figure(figsize=figsize)
        sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu",
                    cbar_kws={'label': 'Accuracy (%)'},
                    vmin=10, vmax=90)
        
        plt.title('Accuracy Heatmap by Class Count and Epoch', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Number of Classes', fontsize=12)
        
        if output_file:
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            print(f"Plot saved to {output_file}")
        else:
            plt.tight_layout()
            plt.show()

    def plot_small_multiples(self, df, output_file=None, figsize=(16, 12)):
        class_counts = sorted(df['number_of_classes'].unique())
        n_cols = 4
        n_rows = int(np.ceil(len(class_counts)/n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True)
        fig.suptitle('Accuracy Progression by Class Count', y=1.02, fontsize=16)
        
        for i, count in enumerate(class_counts):
            ax = axes.flat[i]
            subset = df[df['number_of_classes'] == count]
            
            ax.plot(subset['epoch'], subset['accuracy'], 
                    marker='o', markersize=4, color='steelblue')
            
            ax.set_title(f'{count} classes', pad=5)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy (%)')
            ax.grid(True, alpha=0.2)
            ax.set_ylim(0, 90)
            
            # Hide empty subplots
            for j in range(i+1, n_rows*n_cols):
                axes.flat[j].axis('off')
            
            if output_file:
                plt.savefig(output_file, bbox_inches='tight', dpi=300)                
            else:
                plt.tight_layout()
                plt.show()
        print(f"Plot saved to {output_file}")
    
    def plot_cumulative_performance(self, df, output_file=None, figsize=(14, 8)):
        plt.figure(figsize=figsize)
        
        # Calculate max accuracy achieved for each class count
        max_acc = df.groupby('number_of_classes')['accuracy'].max().sort_index()
        
        # Calculate cumulative max accuracy
        cumulative_max = max_acc.cummax()
        
        # Plot both metrics
        plt.plot(max_acc.index, max_acc.values, 'o-', label='Per Configuration Max')
        plt.plot(cumulative_max.index, cumulative_max.values, 's--', 
                color='red', label='Cumulative Max')
        
        plt.title('Maximum Achieved Accuracy by Class Count', fontsize=14)
        plt.xlabel('Number of Classes', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.grid(True, alpha=0.2)
        #plt.ylim(0, 80)
        plt.legend()
        
        # Annotate key points
        for x, y in zip(max_acc.index, max_acc.values):
            #if y >= 70 or x % 100 == 0:
                plt.annotate(f'{y:.1f}%', (x, y), textcoords="offset points",
                            xytext=(0,2), ha='center')
        
        if output_file:
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            print(f"Plot saved to {output_file}")
        else:
            plt.tight_layout()
            plt.show()

    def compare_cumulative_performance(self, df1, df2, output_file=None, figsize=(14, 8)):

        plt.figure(figsize=figsize)
        
        # Process first dataset
        max_acc1 = df1.groupby('number_of_classes')['accuracy'].max().sort_index()
        cumulative_max1 = max_acc1.cummax()
        
        # Process second dataset
        max_acc2 = df2.groupby('number_of_classes')['accuracy'].max().sort_index()
        cumulative_max2 = max_acc2.cummax()
        
        # Plot configuration maxima
        plt.plot(max_acc1.index, max_acc1.values, 'o-', 
                color='#1f77b4', label='Config Max (Set 1)')
        plt.plot(max_acc2.index, max_acc2.values, 'o-', 
                color='#ff7f0e', label='Config Max (Set 2)')
        
        # Plot cumulative maxima
        plt.plot(cumulative_max1.index, cumulative_max1.values, 's--',
                color='#1f77b4', alpha=0.7, label='Cumulative Max (Set 1)')
        plt.plot(cumulative_max2.index, cumulative_max2.values, 's--',
                color='#ff7f0e', alpha=0.7, label='Cumulative Max (Set 2)')
        
        # Formatting
        plt.title('Comparative Accuracy by Class Count', fontsize=14, pad=20)
        plt.xlabel('Number of Classes', fontsize=12, labelpad=10)
        plt.ylabel('Accuracy (%)', fontsize=12, labelpad=10)
        plt.grid(True, alpha=0.2)
        
        # Smart annotation (only label notable points)
        for x, y in zip(max_acc1.index, max_acc1.values):
            if y >= max(max_acc1.values) * 0.8 or x % 200 == 0:
                plt.annotate(f'{y:.1f}%', (x, y), 
                            textcoords="offset points",
                            xytext=(0,5), ha='center',
                            color='#1f77b4')
        
        for x, y in zip(max_acc2.index, max_acc2.values):
            if y >= max(max_acc2.values) * 0.8 or x % 200 == 0:
                plt.annotate(f'{y:.1f}%', (x, y), 
                            textcoords="offset points",
                            xytext=(0,-15), ha='center',
                            color='#ff7f0e')
        
        plt.legend(loc='lower right', framealpha=0.9)
        
        if output_file:
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            print(f"Saved comparison plot to {output_file}")
        else:
            plt.tight_layout()
            plt.show()

    def compare_cumulative_performance2(s22elf, df1, df2, output_file=None, figsize=(14, 8)):
        plt.figure(figsize=figsize)
        
        # Process first dataset
        max_acc1 = df1.groupby('number_of_classes')['accuracy'].max().sort_index()
        cumulative_max1 = max_acc1.cummax()
        
        # Process second dataset
        max_acc2 = df2.groupby('number_of_classes')['accuracy'].max().sort_index()
        cumulative_max2 = max_acc2.cummax()
        
        # Plot configuration maxima
        plt.plot(max_acc1.index, max_acc1.values, 'o-', 
                color='#1f77b4', label='Config Max (Set 1)')
        plt.plot(max_acc2.index, max_acc2.values, 'o-', 
                color='#ff7f0e', label='Config Max (Set 2)')
        
        # Plot cumulative maxima
        plt.plot(cumulative_max1.index, cumulative_max1.values, 's--',
                color='#1f77b4', alpha=0.7, label='Cumulative Max (Set 1)')
        plt.plot(cumulative_max2.index, cumulative_max2.values, 's--',
                color='#ff7f0e', alpha=0.7, label='Cumulative Max (Set 2)')
        
        # Formatting (ONLY CHANGE: added ylim)
        plt.ylim(0, 80)  # This is the ONLY line I added/modified
        plt.title('Comparative Accuracy by Class Count', fontsize=14, pad=20)
        plt.xlabel('Number of Classes', fontsize=12, labelpad=10)
        plt.ylabel('Accuracy (%)', fontsize=12, labelpad=10)
        plt.grid(True, alpha=0.2)
        
        # Smart annotation (only label notable points)
        for x, y in zip(max_acc1.index, max_acc1.values):
            if y >= max(max_acc1.values) * 0.8 or x % 200 == 0:
                plt.annotate(f'{y:.1f}%', (x, y), 
                            textcoords="offset points",
                            xytext=(0,5), ha='center',
                            color='#1f77b4')
        
        for x, y in zip(max_acc2.index, max_acc2.values):
            if y >= max(max_acc2.values) * 0.8 or x % 200 == 0:
                plt.annotate(f'{y:.1f}%', (x, y), 
                            textcoords="offset points",
                            xytext=(0,-15), ha='center',
                            color='#ff7f0e')
        
        plt.legend(loc='lower right', framealpha=0.9)
        
        if output_file:
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            print(f"Saved comparison plot to {output_file}")
        else:
            plt.tight_layout()
            plt.show()
            
    def plot_incremental_trainingG_C(self, data_file, output_file=None, style='grouped', 
                                figsize=(18, 14), ylim=(0, 80), y_margin=5):
        # Load data
        try:
            df = pd.read_csv(data_file)
        except Exception as e:
            raise ValueError(f"Error loading data file: {e}")
        
        # Validate and calculate dynamic y-axis if needed
        if ylim == 'auto':
            max_acc = df['accuracy'].max()
            ylim = (0, max_acc + y_margin)
        
        # Create plot with taller proportions
        plt.figure(figsize=figsize)
        
        if style == 'grouped':
            self._plot_grouped(df, ylim)
        elif style == 'continuous':
            self._plot_continuous(df, ylim)
        else:
            raise ValueError("style must be either 'grouped' or 'continuous'")
        
        # Finalize plot with enhanced y-axis
        plt.ylim(ylim)
        plt.title('Incremental Training Performance', fontsize=16, pad=20)
        plt.ylabel('Accuracy (%)', fontsize=14, labelpad=15)
        plt.yticks(np.arange(ylim[0], ylim[1]+1, 5), fontsize=12)  # Show every 5%
        plt.grid(True, alpha=0.3)
        plt.legend(title='Class Count', title_fontsize=12, 
                  fontsize=11, bbox_to_anchor=(1.12, 1), loc='upper left')
        
        if output_file:
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            print(f"Plot saved to {output_file}")
        else:
            plt.tight_layout()
            plt.show()
    
    def _plot_grouped(self, df, ylim):
        """Plot with grouped class counts and taller y-axis"""
        class_counts = df['number_of_classes'].unique()
        colors = plt.cm.viridis(np.linspace(0, 1, len(class_counts)))
        
        x_offset = 0
        x_ticks = []
        x_tick_labels = []
        
        for i, count in enumerate(sorted(class_counts)):
            subset = df[df['number_of_classes'] == count]
            x_positions = np.arange(len(subset)) + x_offset
            
            plt.plot(x_positions, subset['accuracy'],
                     marker='o',
                     linestyle='-',
                     color=colors[i],
                     label=f'{count} classes',
                     markersize=8,  # Larger markers
                     linewidth=2)  # Thicker lines
            
            if i > 0:
                plt.axvline(x=x_offset-0.5, color='gray', linestyle=':', alpha=0.4)
            
            x_ticks.append(x_offset + len(subset)/2)
            x_tick_labels.append(str(count))
            x_offset += len(subset) + 2
        
        plt.xlabel('Number of Classes (Grouped)', fontsize=14, labelpad=15)
        plt.xticks(x_ticks, x_tick_labels, fontsize=12)
    
    def _plot_continuous(self, df, ylim):
        """Continuous plot with enhanced y-axis"""
        class_counts = df['number_of_classes'].unique()
        colors = plt.cm.plasma(np.linspace(0, 1, len(class_counts)))  # Different colormap
        
        for i, count in enumerate(sorted(class_counts)):
            subset = df[df['number_of_classes'] == count]
            plt.plot(subset['epoch'], subset['accuracy'],
                     marker='D',  # Diamond markers
                     linestyle='-',
                     color=colors[i],
                     label=f'{count} classes',
                     markersize=6,
                     linewidth=1.5,
                     alpha=0.8)
        
        plt.xlabel('Training Epoch', fontsize=14, labelpad=15)
        plt.xticks(fontsize=12)


def plot_all_training(input_file):
    visualizer = TrainingVisualizer()
    
    outputfolder = input_file.replace('.csv', '')

    os.makedirs(outputfolder, exist_ok=True)  

    outputfolder = outputfolder.replace('plotting/',  '')
    
    visualizer.plot_incremental_trainingG_C(data_file=input_file, output_file=  'plotting/' + outputfolder + '/training_plot_grouped.png', style='grouped')
    visualizer.plot_incremental_trainingG_C(data_file=input_file, output_file=  'plotting/' + outputfolder + '/training_plot_continuous.png', style='continuous')

    df = pd.read_csv(input_file)
    visualizer.plot_cumulative_performance(df, 'plotting/' + outputfolder + '/training_plot_cum.png')
    


# Example Usage
if __name__ == '__main__':
    
    
    input_file1 = 'plotting/compare1/staticalph100training_data_2025-03-25_19-16-57.csv'
    input_file2 = 'plotting/compare1/staticdiss100training_data_2025-03-26_03-11-53.csv'
    input_file3 = 'plotting/120E_20/dec_diss_seq_20training_data_.csv'
    input_file4 = 'plotting/120E_20/static_alph_20training_data_.csv'
    input_file5 = 'plotting/120E_20/static_diss_20training_data_.csv'

    input_file6 = 'plotting/120E_10/dec_alph_10training_data_2025-04-01_16-53-05.csv'
    input_file7 = 'plotting/120E_10/dec_diss_10training_data_2025-04-01_19-00-09.csv'
    input_file8 = 'plotting/120E_10/static_alph_10training_data_2025-04-01_12-27-36.csv'
    input_file9 = 'plotting/120E_10/static_diss-10training_data_2025-04-01_14-08-27.csv'

    input_file_diss = 'plotting/120E_20/dec_diss_20training_data_.csv'
    input_file_hybrid = 'plotting/120E_20/dec_h_diss_seq.csv'

    """plot_all_training(input_file1)  
    plot_all_training(input_file2)
    plot_all_training(input_file3)
    plot_all_training(input_file4)
    plot_all_training(input_file5)

    plot_all_training(input_file6)
    plot_all_training(input_file7)
    plot_all_training(input_file8)
    plot_all_training(input_file9)"""

    outputfolder = input_file_diss.replace('.csv', '')
    os.makedirs(outputfolder, exist_ok=True)  
    outputfolder = outputfolder.replace('plotting/',  '')

    df1 = pd.read_csv(input_file_diss) #blue
    df2 = pd.read_csv(input_file_hybrid) #Orange
    visualizer = TrainingVisualizer()

    output_file = 'plotting/' + outputfolder + '/compare.png'
    visualizer.compare_cumulative_performance2(df1, df2, output_file)


def plot_all_inc(inc_train_csv_file):
    outputfolder = inc_train_csv_file.replace('.csv', '')

    os.makedirs(outputfolder, exist_ok=True)  

    outputfolder = outputfolder.replace('plotting/',  '')


    visualizer = IncrementalTrainingVisualizer()
  #  visualizer.plot_inc_training_shapes(inc_train_csv_file, output_file=  'plotting/' + outputfolder + '/plot_inc_training_shapes.png')
   # visualizer.plot_inc_training_shapes2(inc_train_csv_file, output_file=  'plotting/' + outputfolder + '/plot_inc_training_shapes2.png')
    visualizer.plot_inc_training_shapes3(inc_train_csv_file, output_file=  'plotting/' + outputfolder + '/plot_inc_training_shapes3.png')

   # visualizer.plot_inc_training(inc_train_csv_file, output_file=  'plotting/' + outputfolder + '/training_plot.png')

  #  visualizer.plot_iteration_curve(inc_train_csv_file, "plotting/" + outputfolder + "/inc_" + "plot_iteration_curve.png")
  #  visualizer.plot_effort_reduction(inc_train_csv_file, "plotting/" + outputfolder + "/inc_" + "plot_effort_reduction.png")
  #  visualizer.plot_effort_reduction3(inc_train_csv_file, "plotting/" + outputfolder + "/inc_" + "plot_effort_reduction3.png") 
   # visualizer.plot_effort_per_class(inc_train_csv_file, "plotting/" + outputfolder + "/inc_" + "plot_effort_per_class.png") 

  #  visualizer.plot_cumulative_effort(inc_train_csv_file, "plotting/" + outputfolder + "/inc_" + "plot_cumulative_effort.png")

"""if __name__ == "__main__":
    print(1)

    inc_train_csv_file1 = 'plotting/innnc_inc_training_data_2025-03-27_17-54-16.csv'
    inc_train_csv_file2 = 'plotting/innnc_static_training_data_2025-03-25_19-16-57.csv'
    inc_train_csv_file3 = 'plotting/innnc-dec-diss_training_data_2025-03-25_17-59-34.csv'

    plot_all_inc(inc_train_csv_file1)
    plot_all_inc(inc_train_csv_file2)
    plot_all_inc(inc_train_csv_file3)"""

