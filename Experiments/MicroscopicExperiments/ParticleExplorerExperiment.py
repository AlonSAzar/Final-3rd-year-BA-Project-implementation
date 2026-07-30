import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, find_objects
from engine import ElementaryCA


def discover_particles(rule, L=200, T=200):
    engine = ElementaryCA(L, T)

    # 1. Generate a sparse seed (to avoid overcrowding)
    # 95% zeros, 5% ones
    seed = (np.random.random(L) > 0.95).astype(np.uint8)

    raw_history = engine.run(rule, seed)

    # 2. Filter Background (Assuming period 1 or shift 1)
    # Simple XOR with time-shifted version removes static/checkerboard backgrounds
    filtered = raw_history[1:] ^ raw_history[:-1]

    # 3. Identify Blobs (Particles)
    # structure=[[1,1,1],[1,1,1],[1,1,1]] defines connectivity (8-neighbors)
    labeled_array, num_features = label(filtered, structure=np.ones((3, 3)))

    print(f"Found {num_features} potential particle interactions.")

    # 4. Extract and Visualize "Specimens"
    objs = find_objects(labeled_array)

    # Filter for interesting blobs (e.g., duration > 10 steps)
    interesting_blobs = []
    for i, slice_tuple in enumerate(objs):
        time_slice, space_slice = slice_tuple
        duration = time_slice.stop - time_slice.start
        width = space_slice.stop - space_slice.start

        if duration > 15 and width < 20:  # Long lived, localized
            # Extract the raw pattern from the ORIGINAL history
            specimen = raw_history[time_slice, space_slice]
            interesting_blobs.append(specimen)

    # Display the Zoo
    if not interesting_blobs:
        print("No distinct particles found (maybe pure chaos?). Try Rule 110.")
        return

    print(f"Displaying {min(5, len(interesting_blobs))} specimens:")
    fig, axes = plt.subplots(1, min(5, len(interesting_blobs)), figsize=(15, 3))
    if len(interesting_blobs) == 1: axes = [axes]

    for ax, specimen in zip(axes, interesting_blobs):
        ax.imshow(specimen, cmap='binary', interpolation='nearest')
        ax.axis('off')
        ax.set_title(f"{specimen.shape}")

    plt.show()

# Try with Rule 110 (famous for particles) or Rule 54
# discover_particles(110)