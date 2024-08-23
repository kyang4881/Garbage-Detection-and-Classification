from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers import ViTImageProcessor, ViTForImageClassification, AutoImageProcessor
import torch
from torchvision.transforms import Normalize, ToTensor, Compose, transforms
import matplotlib.pyplot as plt
import numpy as np
import math

class imageClassification:

    def __init__(self, test_folder_path, fine_tuned_model, num_img_show, device):
        self.test_folder_path = test_folder_path
        self.fine_tuned_model = fine_tuned_model
        self.num_img_show = num_img_show
        self.device = device

    def load_data(self):
        """Load the required dataset using the load_dataset method
        """
        ds = load_dataset(self.test_folder_path)
        return ds['test']

    def image_transform(self, data):
        """Transform the input images to pixel values
        Args:
            Data(dataset): A dataset containing the images, labels, and pixel values
        Returns:
            An updated dataset with transformed pixel values
        """
        data['pixel_values'] = [self.image_transformation(image.convert("RGB")) for image in data['image']]
        return data

    def preprocess_data(self, test_data):
        """Preprocess the input images to the required format by applying various transformations
        Args:
            test_data(dataset): A test dataset containing the images, labels, and pixel values
        Returns:
            The test datasets with transformation applied; the id2label and label2id maps, and the model image processor
        """
        # Map labels to ids and ids to labels
        # Define the image processor based on a checkpoint ViT model to process the images
        # Normalize, resize, and convert the images to tensor format
        processor = ViTImageProcessor.from_pretrained(self.fine_tuned_model)
        image_mean, image_std = processor.image_mean, processor.image_std
        normalize = Normalize(mean=image_mean, std=image_std)
        # The pretrained model uses 224x224 images only; upscale the input images to this size
        self.image_transformation = Compose([ToTensor(), normalize, transforms.Resize((224, 224))])
        # Apply the transformation on the datasets
        test_data.set_transform(self.image_transform)
        return test_data

    def run_demo(self):
        plt.close()

        test_ds = self.load_data()
        id2label = {id:label for id, label in enumerate(test_ds.features['label'].names)}
        label2id = {label:id for id, label in id2label.items()}
        image_processor = AutoImageProcessor.from_pretrained(self.fine_tuned_model)
        model = ViTForImageClassification.from_pretrained(self.fine_tuned_model, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True).to(self.device)

        test_ds_processed = self.preprocess_data(test_ds)
        num_images = test_ds_processed.num_rows
        num_rows = int(math.ceil(min(self.num_img_show, num_images) / 3))  # Set the max number of images per row to 3
        num_cols = min(num_images, 3)  # Limit the number of columns to 3
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 3* num_rows))
        desired_size = (224, 224)

        preds = []
        labels = []
        for i in range(min(self.num_img_show, num_images)):
            inputs = image_processor(test_ds_processed[i]['image'], return_tensors="pt").to(self.device)
            with torch.no_grad(): logits = model(**inputs).logits
            predicted_label = logits.argmax(-1).item()

            preds.append(predicted_label)
            labels.append(test_ds_processed[i]['label'])

            img = test_ds_processed[i]['image']
            img = img.resize(desired_size)

            row = i // num_cols
            col = i % num_cols
            axs[row, col].imshow(img)
            title_color = "green" if predicted_label == test_ds_processed[i]['label'] else "red"
            axs[row, col].set_title(f"[Prediction = {id2label[predicted_label]}]\n[Truth = {id2label[labels[i]]}]", fontsize=10, color=title_color)  # Set the title
            axs[row, col].axis('off')

        for ax in axs.flat: ax.axis('off')
        plt.tight_layout()
        print(f"\nModel Accuracy: {accuracy_score(preds, labels)}\n")