# Import necessary libraries
import os  # For navigating the file system
import pandas as pd  # For data manipulation and analysis
import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns  # For advanced visualization techniques
from sklearn.preprocessing import LabelEncoder  # For encoding categorical variables
import random  # For random sampling
import pydicom  # For handling DICOM image files
import warnings  # For managing warnings in the code

# Define a class for data analytics
class data_analytics():
    # Method to calculate and visualize the correlation matrix
    def correlation_matrix(self, data):
        """
        Computes and visualizes the correlation matrix for numerical columns in the dataset.
        Args:
            data (pd.DataFrame): The dataset to analyze.
        """
        correlation_matrix = data.iloc[:, 1:].corr()  # Exclude non-numeric columns like 'study_id'

        # Plot the heatmap for the correlation matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title("Correlation Matrix Heatmap")
        plt.show()

    # Method to randomly select and display 10 DICOM images from the dataset
    def display_image_randomly(self, image_path):
        """
        Randomly selects and displays 10 DICOM images from the specified directory.
        Args:
            image_path (str): Path to the directory containing DICOM images.
        """
        images = []

        # Traverse the directory to collect all DICOM files
        for root, dirs, files in os.walk(image_path):
            for file in files:
                if file.lower().endswith('.dcm'):  # Filter for .dcm files
                    images.append(os.path.join(root, file))

        # Check if there are at least 10 images available
        if len(images) < 10:
            print("Not enough images in the dataset to sample 10 randomly.")
            return
        
        # Randomly select 10 images for display
        random_images = random.sample(images, 10)

        # Set up a grid for displaying the images
        cols = 5
        rows = 2
        fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
        axes = axes.flatten()  # Flatten axes array for easier iteration

        for i, img_path in enumerate(random_images):
            try:
                # Read the DICOM file
                dicom_img = pydicom.dcmread(img_path)
                
                # Display the image in grayscale
                axes[i].imshow(dicom_img.pixel_array, cmap='gray')
                axes[i].set_title(f"Image {i+1}")
                axes[i].axis('off')  # Hide axes for better visualization
            except Exception as e:
                # Handle errors for files that can't be processed
                axes[i].axis('off')
                axes[i].set_title("Error")
                print(f"Error loading DICOM file {img_path}: {e}")

        # Hide any unused subplots
        for j in range(len(random_images), len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()

    # Method to plot diagnosis distributions for specific categories
    def images_by_category(self, data):
        """
        Visualizes the distribution of diagnostic categories in the dataset.
        Args:
            data (pd.DataFrame): The dataset to analyze.
        """
        figure, axis = plt.subplots(1, 3, figsize=(20, 5)) 
        for idx, d in enumerate(['foraminal', 'subarticular', 'canal']):  # Categories of interest
            diagnosis = list(filter(lambda x: x.find(d) > -1, data.columns))  # Filter columns by category
            dff = data[diagnosis]
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=FutureWarning)
                value_counts = dff.apply(pd.value_counts).fillna(0).T
            value_counts.plot(kind='bar', stacked=True, ax=axis[idx])  # Stacked bar plot
            axis[idx].set_title(f'{d} distribution')

    def summarize_dataset(self):
        """
        Summarize the dataset by displaying distributions of diagnostic categories.
        """
        # Exclude 'study_id' column for analysis
        diagnostic_columns = self.data.columns[1:]

        # Display the distribution for each diagnostic column
        for col in diagnostic_columns:
            print(f"\nDistribution for {col}:")
            counts = self.data[col].value_counts()
            print(counts)
            
            # Plot the distribution
            counts.plot(kind='bar', figsize=(8, 4), title=f"Distribution of {col}")
            plt.xlabel("Diagnostic Categories")
            plt.ylabel("Frequency")
            plt.show()

    def missing_studies(self):

        print("---------- Checking Missing Values within the Dataset ----------")
        missing_values = self.data.isnull().sum()
        print("The dataset has the following missing values per column:")
        print("Missing Values:\n", missing_values)
        
        # Checking for Missing Study images
        print("---------- Checking Missing Image within the Dataset ----------")
        study_ids = self.data[data.columns[0]].astype(str)
        study_ids_set = set(study_ids)
        missing_studies = []

        # Get only directories in the specified path
        directories = [d for d in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, d))]


        for directory in directories:
            if directory not in study_ids_set:
                print("Study not found: ", directory)
                missing_studies.append(directory)


        print("Total number of studies missing: ", len(missing_studies))
        print("Total study ids of missing studies: ", missing_studies)

    # Initialization method for data analysis
    def __init__(self, file_path):
        """
        Initializes the data analysis process.
        Args:
            file_path (str): Path to the directory containing the dataset and images.
        """

        print("---------- Dataset Origin ----------")
        print("The dataset originates from RSNA competition on Kaggle.")
        print(r"The link for the dataset is: https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/overview")
        print("It was created to provide a comprehensive set of DICOM images for training and evaluating machine learning models in medical diagnosis.")
        print("The dataset is licensed under [insert license, e.g., CC BY-SA 4.0] and includes real-world data with annotations.")

        print("---------- Dataset Features ----------")
        print("The dataset contains the following key features:")
        print("- Study ID: Unique identifier for each medical case.")
        print("- Diagnostic Labels: Includes labels such as foraminal, subarticular, and canal.")
        print("- Images: 10,000+ DICOM (.dcm) files, representing cross-sectional scans.")
        print("- Metadata: A CSV file with patient information and diagnostic labels.")

        data_path = file_path + 'train.csv'

        # Load and display the first 5 rows of the dataset
        print("---------- First 5 Elements of the Dataset ----------")
        data = pd.read_csv(data_path)
        self.data = data
        print(self.data.head())

        # Display random images from the dataset
        print("---------- Showing Random Images from the Dataset ----------")
        image_path = file_path + 'train_images/'
        self.display_image_randomly(image_path)

        print("---------- Dataset Summary ----------")
        data_analytics.summarize_dataset(self)
        
        # Visualizing teh distribution of each category
        print("---------- Distribution per Category ----------")
        data_analytics.images_by_category(self, data)

        # Check for missing values in the dataset
        data_analytics.missing_studies(self)

        # Encode categorical columns into numerical values
        label_encoders = {}
        for column in data.columns[1:]:  # Skip the "study_id" column
            if data[column].dtype == 'object':  # Check if the column is categorical
                le = LabelEncoder()
                data[column] = le.fit_transform(data[column])  # Encode the column
                label_encoders[column] = le

        # Perform basic statistics
        print("---------- Basic Statistical Information on the Dataset ----------")
        self.basic_statistics(data)

        # Generate and visualize the correlation matrix
        print("---------- Correlation Matrix ----------")
        self.correlation_matrix(data)