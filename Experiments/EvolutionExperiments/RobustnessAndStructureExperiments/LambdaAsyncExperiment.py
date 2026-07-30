import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import zlib
from tqdm import tqdm
import pandas as pd
import plotly.express as px
import os

# --- 1. The Asynchronous Engine ---
@njit
def simulate_async_ca(rule_lut, seed, T, sync_prob=0.95):
    L = len(seed)
    output = np.zeros((T, L), dtype=np.uint8)
    output[0] = seed

    for t in range(1, T):
        for i in range(L):
            left = output[t - 1][(i - 1) % L]
            center = output[t - 1][i]
            right = output[t - 1][(i + 1) % L]
            neighborhood = (left << 2) | (center << 1) | right

            if np.random.random() < sync_prob:
                output[t][i] = rule_lut[neighborhood]
            else:
                output[t][i] = center
    return output

def get_lut(rule):
    return np.array([(rule >> i) & 1 for i in range(7, -1, -1)], dtype=np.uint8)

def calculate_langton_lambda(rule):
    """
    Calculates Langton's Lambda for a 1D binary CA rule.
    Since k=2 (binary) and K=8 (neighborhood size), 
    Lambda = (number of non-zero transitions) / K.
    In standard binary CA, the 'quiescent' state is usually 0.
    """
    lut = get_lut(rule)
    # Number of transitions to state 1
    non_quiescent_count = np.sum(lut)
    return non_quiescent_count / 8.0

# --- 2. The Experiment ---
def test_async_lambda_cliff(L=128, T=128, sync_prob=0.95, num_seeds=10):
    print(f"Running Langton's Lambda Async Experiment (Sync Prob = {sync_prob})...")

    lambdas = []
    robustness_scores = []
    rule_labels = []

    for rule in tqdm(range(256), desc="Rules"):
        lut = get_lut(rule)
        lam = calculate_langton_lambda(rule)

        rule_robustness = []

        for _ in range(num_seeds):
            seed = np.random.randint(0, 2, size=L, dtype=np.uint8)

            # 1. Run Perfect Clock (Control)
            sync_history = simulate_async_ca(lut, seed, T, sync_prob=1.0)

            # 2. Run Broken Clock (Experiment)
            async_history = simulate_async_ca(lut, seed, T, sync_prob=sync_prob)

            # 3. Measure Complexity (Zlib on the last 50 rows)
            sync_k = len(zlib.compress(sync_history[-50:].tobytes()))
            async_k = len(zlib.compress(async_history[-50:].tobytes()))

            # 4. Calculate Robustness
            deviation = abs(sync_k - async_k) / max(sync_k, 1)
            robustness = max(0.0, 1.0 - deviation)

            rule_robustness.append(robustness)

        lambdas.append(lam)
        robustness_scores.append(np.mean(rule_robustness))
        rule_labels.append(rule)

    # --- 3. Plotting ---
    
    # Calculate Averages for each Lambda value
    df_results = pd.DataFrame({
        'Lambda': lambdas,
        'Robustness': robustness_scores,
        'Rule': rule_labels
    })
    
    # Group by Lambda and calculate mean/std for the average line
    df_avg = df_results.groupby('Lambda')['Robustness'].agg(['mean', 'std']).reset_index()

    # 1. Matplotlib Scatter
    plt.figure(figsize=(10, 6))
    
    # Background: Individual rules in grey
    plt.scatter(df_results['Lambda'], df_results['Robustness'], 
                alpha=0.2, edgecolors='none', color='gray', label='Individual Rules')
    
    # Foreground: Average per Lambda value
    plt.plot(df_avg['Lambda'], df_avg['mean'], 'o-', color='crimson', 
             linewidth=2.5, markersize=8, label='Average Robustness')
    
    # Optional: Fill between mean +/- std
    plt.fill_between(df_avg['Lambda'], 
                     df_avg['mean'] - df_avg['std'], 
                     df_avg['mean'] + df_avg['std'], 
                     color='crimson', alpha=0.1)
    
    plt.title(f"Async Robustness vs Langton's Lambda (Sync Prob: {sync_prob*100}%)")
    plt.xlabel("Langton's Lambda (Non-zero transition ratio)")
    plt.ylabel("Structural Robustness (1.0 = Pattern Maintained)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 2. Interactive Plotly
    # We add a column to identify which points are individual rules
    df_results['Type'] = 'Rule'
    
    # Create the base scatter
    fig = px.scatter(
        df_results, 
        x='Lambda', 
        y='Robustness',
        hover_data=['Rule', 'Lambda', 'Robustness'],
        title=f"Robustness vs Lambda with Averages (Sync Prob: {sync_prob*100}%)",
        template="plotly_white",
        color_discrete_sequence=['lightgray']
    )
    
    # Add the average line trace
    import plotly.graph_objects as go
    fig.add_trace(go.Scatter(
        x=df_avg['Lambda'],
        y=df_avg['mean'],
        mode='lines+markers',
        name='Average',
        line=dict(color='crimson', width=3),
        marker=dict(size=10, symbol='circle')
    ))

    fig.update_traces(marker=dict(size=8, opacity=0.4), selector=dict(mode='markers', name='Rule'))
    
    save_dir = os.path.join("Saved Figures", "HTML")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    filename = os.path.join(save_dir, f"LambdaAsyncRobustness_with_Averages_Prob_{int(sync_prob*100)}.html")
    fig.write_html(filename)
    print(f"Interactive HTML saved to: {filename}")

if __name__ == "__main__":
    test_async_lambda_cliff()
