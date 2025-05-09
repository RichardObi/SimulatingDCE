
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


palette = [".85"] + sns.color_palette(sns.cubehelix_palette(), n_colors=4)
color="0.55"
# color=palette[3]
# color=palette[1]
# color=palette[i+1] for i in range(2)


# Load CSV
csv_path = '/workspace/Richard-JMI-extension/kinetic_curves/kinetic_curves/090_138/image_statistics.csv'
df = pd.read_csv(csv_path)

# Convert columns to numeric
df["Mean Intensity"] = pd.to_numeric(df["Mean Intensity"], errors='coerce')

# Extract phase ID and type
def extract_id(filename):
    if "CONCAT" in filename:
        return filename.split("CONCAT_")[1].split("_")[0]
    elif "syn" in filename:
        return filename.split("_0000_")[1].split("_")[0]
    elif "0000_slice" in filename:
        return "BASELINE"
    return "UNKNOWN"

df["group_id"] = df["Image Name"].apply(extract_id)
df["type"] = df["Image Name"].apply(lambda x: "CONCAT" if "CONCAT" in x else "SYN" if "syn" in x else "BASELINE")

# Get baseline value
baseline_value = df[df["type"] == "BASELINE"]["Mean Intensity"].values[0]

# Filter target phases (0001–0003)
target_ids = ['0001', '0002', '0003']
df_plot = df[df["group_id"].isin(target_ids) & (df["type"] != "BASELINE")]
df_plot = df_plot.sort_values(by=["group_id", "type"])

# Pivot for plotting
mean_pivot = df_plot.pivot(index="group_id", columns="type", values="Mean Intensity")

# X-axis positions and labels
x_labels = ['1', '2', '3']
x_positions = range(len(x_labels))

# Plot
plt.figure(figsize=(8, 6))

plt.plot(x_positions, mean_pivot.loc[target_ids, "CONCAT"], marker='o', label='real-contrast', color="0.55", linewidth=2)
plt.plot(x_positions, mean_pivot.loc[target_ids, "SYN"], marker='o', label='synthetic-contrast', color=palette[2], linewidth=2)

# Baseline line
plt.axhline(y=baseline_value, color='gray', linestyle='--', label='pre-contrast', linewidth=1.5)

# Formatting
plt.title("Contrast Enhancement Curve", fontsize=16)
plt.xlabel("Contrast Phase", fontsize=14)
plt.ylabel("Mean Intensity", fontsize=14)
plt.xticks(ticks=x_positions, labels=x_labels, fontsize=12)
plt.yticks(fontsize=12)
plt.yticks([0, 20])
plt.legend(fontsize=13)
plt.grid(True)
plt.tight_layout()

# Save and show
plt.savefig(csv_path.replace('image_statistics.csv', 'plot.png'))
plt.show()
