import json
import matplotlib.pyplot as plt

# For example, assume the JSON data is stored in a file called "data.json".
# You could also load it from a string.
with open("data.json", "r") as f:
    data = json.load(f)

models = data["models"]
results_matrix = data["results_matrix"]

plt.figure(figsize=(14, 7))

for model, answers in zip(models, results_matrix):
    # Convert None values to 4
    y_values = [4 if ans is None else ans for ans in answers]
    x_values = list(range(1, len(y_values) + 1))
    plt.plot(x_values, y_values, marker="o", label=model)

plt.xlabel("Question Number")
plt.ylabel("Answer (0-3, 4 represents None)")
plt.yticks([0, 1, 2, 3, 4])
plt.title("Model Answers by Question")
plt.legend(loc="upper right", fontsize="small", ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()
