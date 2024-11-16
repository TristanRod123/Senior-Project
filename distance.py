import csv
import math

def calculate_distance(point1, point2):
    """Calculate Euclidean distance in 2D or 3D."""
    if len(point1) == 2:  # 2D case
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    elif len(point1) == 3:  # 3D case
        return math.sqrt((point2[0] - point1[0])**2 +
                         (point2[1] - point1[1])**2 +
                         (point2[2] - point1[2])**2)

connections = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20)
]

# Read input CSV and process each line to append distances
with open('model\keypoint_classifier\keypoint.csv', 'r') as infile, open('keypoint_distance.csv', 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    for row in reader:
        # Extract landmarks (skip label in the first column)
        landmarks = [float(val) for val in row[1:]]
        points = [landmarks[i:i+2] for i in range(0, len(landmarks), 2)]

        # Calculate distances for the current frame
        distances = [calculate_distance(points[start], points[end]) for start, end in connections]

        # Append distances to the original row and write to the new file
        writer.writerow(row + distances)
