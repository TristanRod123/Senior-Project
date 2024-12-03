import csv
from collections import Counter

def count_labels_in_csv(file_path):
    label_counts = Counter()  # Create a Counter to hold counts of each label

    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            if row:  # Check if the row is not empty
                label = row[0]  # Assuming the label is in the first column
                label_counts[label] += 1  # Increment the count for the label

    return label_counts

# Example usage
file_path = 'model/keypoint_classifier/keypoint_original_data_v2.csv'
counts = count_labels_in_csv(file_path)

# Print individual label counts
for label, count in counts.items():
    print(f"Label: {label}, Count: {count}")

# Print the total count of all labels
total_count = sum(counts.values())
print(f"\nTotal Count: {total_count}")
