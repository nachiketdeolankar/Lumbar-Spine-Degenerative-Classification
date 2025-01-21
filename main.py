# Import custom modules for data analysis and clustering
from data_analytics import data_analytics  # Handles dataset exploration and visualizations
from visual_transformer import VisionTransformerTrainer, MultiColumnTrainer

# Define the path to the dataset directory
file_path = '/Users/nachiketdeolankar/Projects/2024/OpenSource/RSNA/dataset/'

# Perform general data analysis and visualization
# This includes tasks like loading the dataset, checking for missing values, and visualizing distributions
print("---------- Data Analysis ----------")
data_analytics(file_path)

print("---------- Visual Transformer Model for Prediction ----------")
target_columns = [
    "left_neural_foraminal_narrowing_l1_l2",
    "left_neural_foraminal_narrowing_l2_l3",
    "left_neural_foraminal_narrowing_l3_l4",
    "left_neural_foraminal_narrowing_l4_l5",
    "left_neural_foraminal_narrowing_l5_s1",
    "right_neural_foraminal_narrowing_l1_l2",
    "right_neural_foraminal_narrowing_l2_l3",
    "right_neural_foraminal_narrowing_l3_l4",
    "right_neural_foraminal_narrowing_l4_l5",
    "right_neural_foraminal_narrowing_l5_s1",
    "left_subarticular_stenosis_l1_l2",
    "left_subarticular_stenosis_l2_l3",
    "left_subarticular_stenosis_l3_l4",
    "left_subarticular_stenosis_l4_l5",
    "left_subarticular_stenosis_l5_s1",
    "right_subarticular_stenosis_l1_l2",
    "right_subarticular_stenosis_l2_l3",
    "right_subarticular_stenosis_l3_l4",
    "right_subarticular_stenosis_l4_l5",
    "right_subarticular_stenosis_l5_s1"
]
# Training parameters
batch_size = 16
num_classes = 3  # Number of classes (Normal/Mild, Moderate, Severe)
epochs = 25  # Number of training epochs

train_root = (r'/Users/nachiketdeolankar/Projects/2024/OpenSource/RSNA/dataset/train_images')
train_labels = (r'/Users/nachiketdeolankar/Projects/2024/OpenSource/RSNA/dataset/train.csv')

# Initialize and train the models
multi_trainer = MultiColumnTrainer(
    root_dir=train_root,
    train_labels=train_labels,
    target_columns=target_columns,
    model_dir="models/",
    batch_size=batch_size,
    num_classes=num_classes,
    epochs=epochs
)