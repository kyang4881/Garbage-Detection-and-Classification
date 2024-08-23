from datasets import load_dataset
from transformers import ViTImageProcessor
from torchvision.transforms import Normalize, ToTensor, Compose, transforms

class setupPipeline:
    """A pipeline for setting up the input images into the required format for training and inference
    Args:
          dataset_name(str): The name of the dataset to extract from the datasets library
          train_size(int): The size of the train dataset to extract from the datasets library
          test_size(int): The size of the test dataset to extract from the datasets library
          validation_split(float): A ratio for splitting the training data
          shuffle_data(bool): Wether to shuffle the data
          model_checkpoint(str): A pretrained model checkpoint from HuggingFace
          image_transformation(obj): An object specifying the type of image transformation required
    """
    def __init__(self, dataset_name, train_size, test_size, validation_split, shuffle_data, model_checkpoint):
        self.dataset_name = dataset_name
        self.train_size = train_size
        self.test_size = test_size
        self.validation_split = validation_split
        self.shuffle_data = shuffle_data
        self.model_checkpoint = model_checkpoint
        self.image_transformation = None

    def load_data(self):
        """Load the required dataset using the load_dataset method
        """
        ds = load_dataset(self.dataset_name) #, split=['train[:' + str(self.train_size) + ']', 'test[:'+ str(self.test_size) + ']']) #, split=['train[:' + str(self.train_size) + ']', 'test[:'+ str(self.test_size) + ']'])

        return ds['train'], ds['test']

    def image_transform(self, data):
        """Transform the input images to pixel values
        Args:
            Data(dataset): A dataset containing the images, labels, and pixel values
        Returns:
            An updated dataset with transformed pixel values
        """
        data['pixel_values'] = [self.image_transformation(image.convert("RGB")) for image in data['image']]
        return data

    def preprocess_data(self, train_ds, test_ds):
        """Preprocess the input images to the required format by applying various transformations
        Args:
            train_ds(dataset): A train dataset containing the images, labels, and pixel values
            test_ds(dataset): A test dataset containing the images, labels, and pixel values
        Returns:
            The train, validation, and test datasets with transformation applied; the id2label and label2id maps, and the model image processor
        """
        # Split the data into train and validation sets
        train_ds = train_ds.shuffle(seed=42).select(range(self.train_size))
        test_ds = test_ds.shuffle(seed=42).select(range(self.test_size))

        splits = train_ds.train_test_split(test_size=self.validation_split, shuffle=self.shuffle_data)
        train_ds = splits['train']
        val_ds = splits['test']
        # Map labels to ids and ids to labels
        id2label = {id:label for id, label in enumerate(train_ds.features['label'].names)}
        label2id = {label:id for id,label in id2label.items()}
        # Define the image processor based on a checkpoint ViT model to process the images
        processor = ViTImageProcessor.from_pretrained(self.model_checkpoint)
        # Normalize, resize, and convert the images to tensor format
        image_mean, image_std = processor.image_mean, processor.image_std
        normalize = Normalize(mean=image_mean, std=image_std)
        # The pretrained model uses 224x224 images only; upscale the input images to this size
        self.image_transformation = Compose([ToTensor(), normalize, transforms.Resize((224, 224))])
        # Apply the transformation on the datasets
        train_ds.set_transform(self.image_transform)
        val_ds.set_transform(self.image_transform)
        test_ds.set_transform(self.image_transform)
        return train_ds, val_ds, test_ds, id2label, label2id, processor
