# Garbage Detection and Classification

<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Computer%20Vision/Garbage%20Classification/docs/images/background.png" width="1200" />
</p>

---

## Introduction 

In today's world, the environmental challenges have become increasingly urgent, with 3.40 billion tonnes of waste expected to be generated annually by 2050 (Kaza et al, 2018), demanding innovative solutions to mitigate the detrimental impacts of human activities. One critical issue is the proper management of waste and the pressing need to minimise its adverse effects on the ecosystems. 

Recycling is one way to reduce waste. However, before waste can be recycled it needs to be sorted and processed. For example, segregating paper from plastics or removing hazardous waste ensures each type can be processed safely and appropriately. Apart from industrial waste production, consumers are also a first mile of the recycling process. 

---

## Motivation

Unfortunately, much more needs to be done to educate consumers on proper recycling. For example, Singapore’s domestic recycling rate declined from 22 per cent in 2018 to 17 per cent in 2019, below the European Union, which reported a domestic recycling rate of 46.4 per cent in 2017. (Channel News Asia, 2020). According to experts, a lack of recycling knowledge is one of the contributing factors to the low domestic recycling rate. A study by the Singapore Environment Council in 2018 also found that 70 per cent of respondents did not fully know what was recyclable and how waste should be sorted (Channel News Asia, 2020).

An application that can detect and inform consumers on whether an object may be recycled or not and which waste category they should be disposed to could be an intervention to potentially ameliorate the lack of education on domestic waste sorting. 

The project aims to use computer vision models to detect and classify waste materials. It is important that the models can differentiate between different waste materials so that it can inform consumers of proper waste sorting practice. For example, items that are detected as paper would be discarded into a different recycling bin compared to metal items. A trained model could be deployed to the use case of a waste sorting app, to increase domestic recycling rate by educating and informing consumers on proper waste sorting and waste identification. 

---

## Dataset

The dataset selected was sourced from Kumsetty et Al (2022). Titled ‘Trashbox’, the dataset was split into and labelled as seven distinct subcategories: cardboard, e-waste, glass, medical, metal, paper and plastic. Ultimately, five of the seven subcategories (cardboard, glass, metal, plastic and paper) were selected for the sake of brevity. The dataset consisted of 28,564 files and 4.29GB of memory. The images within the dataset largely comprised of stock images of varying sizes (figure 1, left), with few images representative of the waste found in an organic environment such as pavement, void decks, grassy environments etc. To better improve the performance of the models, the training data was altered from its initial state.

Thirty-one different background images containing no garbage were sourced from google images, with environments ranging from but not limited to roads, canals, tiled flooring, void decks, grassy environments etc. Rembg, PIL and OpenCv libraries were used to remove the existing background, crop the stock images found in the ‘Trashbox’ dataset and eventually overlaid at random locations on top of a random selection of the previously mentioned backgrounds. The locations and the image size of the overlaid garbage was noted as the ground truth of the object bounding boxes, used later in the assessment section of the project. 


<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Computer%20Vision/Garbage%20Classification/docs/images/data_overlay.png" width="1200" />
</p>

The resulting dataset consisted of 39,965 files and 67.6 GB of memory, too large of a dataset to be used unaltered without taking a significant amount of time for model training. The images were resized to 224 by 224 pixels from their original 1120 by 1120 pixels, accepting a trade-off in possible performance gains from the increased image resolution for training speed. 

---

## Methodology 

The project aimed to overcome two main tasks: Image classification and object detection. Image classification is defined as the ability of the ability for a trained model to correctly label a provided image among a set of trained classes, while object detection is defined as the ability for a train model to draw a bounding box around an object of interest within a provided image (thereby identifying the location) and then execute a proper classification of that object. A set of models were selected to tackle each of the two tasks, selected due to their recency, ease of implementation and performance whilst having a variety of architecture.

The models were trained and tested with the generated images, with the results documented in the attached appendix (figures 8 to 11). To validate the models, a curated set of non-generated images consisting of garbage found in organic environments (such as one shown in figure 2 below) were used.

<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Computer%20Vision/Garbage%20Classification/docs/images/train_val_test.png" width="1200" />
</p>


---

## Model Selection (Classification)

Transfer learning using models Resnet 50, EfficientNetV2L and Vision Transformer (ViT) were selected for the task of image classification. 

### Resnet 50

Developed by He et al (2015), Resnet50 is a computer vision model on a 50-layer convolutional neural network architecture (CNN). Utilization of residual learning (figure 2) allows the convolution network to overcome commonly encountered degradation associated with vanishing and exploding gradients. The pre-trained model was trained on images available on ImageNet.

<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Computer%20Vision/Garbage%20Classification/docs/images/resnet.png" width="1200" />
</p>

### EfficientNetV2L

EfficientNetV2L (Tan & Le, 2021) is built upon two main concepts, compound coefficient model scaling and neural architecture search (NAS). Often, the continual addition of neural network layers do not necessarily result in a performance improvement of a CNN. By having a set of scaling coefficients, EfficientNet architecture allows for neural networks to be developed with a uniform set of neural network widths, depth and resolution (figure 3). NAS allows for a systematic approach to model tuning via defining search space, search strategy and set performance metrics to further develop a model with good performance. 

<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Computer%20Vision/Garbage%20Classification/docs/images/effnet.png" width="1200" />
</p>


### Vision Transformer (ViT)

Vision Transformer (henceforth referred to as ViT) was developed by Dosovitskiy et al (2020). Trained on Google’s JFT-300M image dataset, ViT architecture (figure 4) differs vastly from CNN architecture. Transformers, often used in natural language processing (NLP), focus on creating encodings for each set of data (such as a sentence, document or image) by forming associations between a token (or image pixel) and all other tokens. To apply a similar NLP approach to an image without alteration will be impractical, as the time complexity of such an operation would be O(n2), impractically large for images often thousands of pixels in width and height. Instead, ViT segments each image into multiple patches (sub-images 16 by 16 pixels in size), creates embeddings for each patch before creating a global association through a transformer encoder. Multi-layer perceptrons (MLP) consolidate the learned weights to form the classification layer of the neural network.

<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Computer%20Vision/Garbage%20Classification/docs/images/vit.png" width="1200" />
</p>

---

## Model Selection (Object detection)

Transfer learning using model YOLOV3 was selected for the task of object identification. 

### YOLOV3

YOLOV3 (Redmon et al, 2016) tackles object identification in a method not unlike ViTs, by splitting a image into a series of sub-images in a grid like fashion. Conventionally, the sliding window object detection method is used for the task of object detection, which uses an approach similar to kernels in CNNs, the model learning from a window moved across the image with the image data and bounding box data. What makes YOLO unique in its approach to object detection is to first split an image into a grid and embedding visual and bounding box data within each cell of the grid. Feeding each sub-image through a CNN, the assessment of the location and appropriate label of the object of interest is assessed as a whole image. A trained model would then be able to predict several viable bounding boxes, with nonmax suppression, a method used to assess probabilities of the bounding boxes, used to determine the most appropriate bounding box for that image.

<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Computer%20Vision/Garbage%20Classification/docs/images/yolo.png" width="1200" />
</p>

---

#### YOLO Methodology

After the data augmentation step, a separate input file was created for each image, containing the bounding box information in the format: class x coordinate, y coordinate, width, height. Subsequently, each image was labelled using the same name as the image file for the model's data processing. Then, the darknet weights and YOLO configuration, trained on the imagenet dataset, were utilized to train the model. Model training however, took a very long time due to the large size of the custom dataset and the specific requirements of the YOLO architecture. 
    
Consequently, a pretrained YOLO model was used to test the custom images. To achieve this, the hyperparameters were tuned according to the dataset, which consisted of 5 classes, and the output layer of the YOLO architecture was modified accordingly.
    
Regarding the architecture, the image was initially resized to 448x448 pixels, and the pixel values were normalized within the range of -1 to 1. These values were then processed through the network, which produced the network output. The architecture primarily consisted of a CNN network that extracted high-level features through a series of convolution and pooling layers. These layers captured contextual information, enabling the network to gain a deeper understanding of the image. The detection layers were responsible for predicting the bounding boxes and class probabilities. The output of the network was a tensor representing grid cells, which contained information about the bounding boxes and class probabilities.
    
When comparing the performance between overlayed images and natural images, the model exhibits superior performance on natural images compared to overlayed images. This discrepancy can be attributed to the dataset on which the model was trained. The pretrained model was trained on the COCO dataset, which consisted of common object images. As a result, the model is more adept at recognizing objects in their original context, where the background is coherent with the object itself, rather than objects superimposed on a different background.

---
    
##  Evaluation 

Evaluation of the previously mentioned models was be done primarily through the analysis of performance metrics. The main performance metrics are the accuracy, precision, recall and eventually the F1 score. For object detection, intersection over union (IOU) will be used as another evaluation metric. Intersection over union (IOU), the measure of the model’s ability to distinguish the objects from the background, will also be used as a performance metric. In addition, models were tested against a selection of images contributed by team members to roughly determine the model accuracies.

---

##  Results

For image classification, it appears that performance of the model improves with the more recent, sophisticated models. For YOLOV3, while the accuracy, precision and recall performance metrics are not as high as the test classification models, the IOU was found to be 57%, a relatively acceptable level for a model not yet trained specially on the selected dataset.
    
 <p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Computer%20Vision/Garbage%20Classification/docs/images/results.png" width="1200" />
</p>   

 <p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Computer%20Vision/Garbage%20Classification/docs/images/pred.png" width="1200" />
</p>   

---

## Challenges

It is to be noted that the number of tests datapoints were not consistent between model evaluations. Through the refinement from Trashbox’s stock images, the resulting dataset consisted of images 1120 by 1120 pixels in size, approximately 40, 000 images over the 4 sub-categories, spanning 67.6 GB. With the limitation of time and computing resources, images were resized down to 224 by 224 pixels in size to allow for ease of model training. Despite this however, model training took a significant amount of time, with training of an object detection model with additional google colaboratory GPU resources on 20% of the training dataset took upwards of 7 hours. Each model was trained and tested on a variety of subset sizes in an attempt to feed in as much training and test data as possible in a reasonable amount of time. While this will introduce a level of inconsistency into model comparisons, representative sampling of both the training and testing datasets make it less likely to have large deviations in model performance than what has been tested.

---

## Memory Optimization

To grapple with the large memory size of the image data, two memory optimization methods had to be implemented, the usage of the DeepSpeed library and the application of low rank approximation (LoRA).

DeepSpeed is an open-source deep learning optimization library for PyTorch. It aims to enhance the efficiency of deep learning training by reducing computational power and memory usage while enabling better parallelism on existing hardware. The library is specifically designed for training large, distributed models with a focus on low latency and high throughput. DeepSpeed incorporates the Zero Redundancy Optimizer (ZeRO), which enables training models with 1 trillion or more parameters. Key features of DeepSpeed include mixed precision training, support for single-GPU, multi-GPU, and multi-node training, as well as customizable model parallelism.

Given the limitations imposed by computational resources, recurrent instances of GPU memory overflow were encountered during the training of the models. Despite attempts to mitigate these issues by reducing sample size and image resolution, the fine-tuning of expansive models like google/vit-large-patch16-224-in21k, available on HuggingFace, demanded substantial computing power and consistently led to runtime crashes. However, by harnessing the power of DeepSpeed, not only was a successful execution of the prodigious ViT model achieved, but significant advancements in the model optimization endeavours. Specifically, larger batch sizes, finer-grained learning rates and more expansive training sample sizes were able to be incorporated, thus capitalizing on the enhanced capabilities provided by DeepSpeed's state-of-the-art deep learning optimization library for PyTorch.
Low Rank Approximation (LoRA) is an optimization technique that reduces the number of trainable parameters by learning pairs of rank-decomposition matrices while freezing the original weights. By leveraging LoRA, the storage footprint and memory usage of the model were able to be reduced, and larger models with better performance on the downstream classification task were able to be trained.

---

## Future Work

These are some approaches that may be taken to further refine model performance.

1. Further image augmentation: The current dataset is limited to the overlay of cropped stock images at random points on the selection of background. The inclusion of additional backgrounds, rotation and resizing of the cropped images during the overlay process would further increase the amount of training data available. Image level transformations such as flips and rotation, as well as pixel level transformations like brightness, contrast and hue adjustments is likely to provide a model performance improvement.

2. Increased training time and resources: It is to be noted however, that the proposed image augmentation will further inflate the datasets, resulting in significant increases in the training and testing time required for each model. Given sufficient time and computing resources, the models may be trained and tested using the augmented dataset consisting of the original sized images (1120x1120). This will allow for the standardization of the training, as well as the testing of the models thereby resulting in a more objective comparison in model performance.

3. Hyperparameter tuning: More extensive hyperparameter tuning is likely to further improve model performance. Different approaches, such as grid searches, random searches or execution of hyperparameter sweeps.

--- 

## Conclusion

The ability to classify images and identify objects was tested through transfer learning of models Resnet50, EfficientNetV2L, ViT and YOLOV3. Preliminary testing shows a general improvement of performance with more recent and sophisticated models. Given additional time and resources, a properly tuned model will be able to assist members of the public in the sorting of recyclables.

---


## Notebook

Import necessary libraries.

```python
import deepspeed
from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from transformers import TrainingArguments, Trainer, ViTImageProcessor, ViTForImageClassification, AutoImageProcessor
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import Normalize, ToTensor, Compose, transforms
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import os
import matplotlib.pyplot as plt
import numpy as np
import math
```

Set up DeepSpeed for optimization

```python
# DeepSpeed requires a distributed environment even when only one process is used.
# This emulates a launcher in the notebook
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "9994"  # modify if RuntimeError: Address already in use
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
```

A class for preprocessing the data.

```python
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

```

A class for executing the training and inference steps.

```python
class runPipeline(setupPipeline):
    """A pipeline for executing the training and inference steps
    Args:
          learning_rate (float): The initial learning rate for AdamW optimizer
          per_device_train_batch_size (int): The batch size per GPU/TPU core/CPU for training
          per_device_eval_batch_size (int): The batch size per GPU/TPU core/CPU for evaluation
          num_train_epochs (int): Number of epoch to train
          weight_decay (float): The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights
          eval_metric(str): A evaluation metric to be displayed when training
          pipeline_type(str): Specifying whether to use the pipeline for training or making prediction
          dataset_name(str): The name of the image dataset
          train_ds(dataset): A train dataset containing the images, labels, and pixel values
          val_ds(dataset): A validation dataset containing the images, labels, and pixel values
          test_ds(dataset): A test dataset containing the images, labels, and pixel values
          label2id(dict): A dictionary to map labels to ids
          id2label(dict): A dictionary to map ids to labels
          model_checkpoint(str): Specifying the model checkpoint based on the HuggingFace API
          processor(obj): A torchvision object for tokenizing the images
          torch_weights_filename(str): A pytorch file containing the fine-tuned weights of the model
          device (obj): Specifies whether to use cpu or gpu
          apply_lora (bool): Whether to apply Lora
          load_weights (bool): Whether to saved torch weights
    """
    def __init__(self, learning_rate, per_device_train_batch_size, per_device_eval_batch_size, num_train_epochs, weight_decay, eval_metric, pipeline_type, dataset_name, train_ds, val_ds, test_ds, label2id, id2label, model_checkpoint, processor, torch_weights_filename, device, apply_lora, load_weights):
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.num_train_epochs = num_train_epochs
        self.weight_decay = weight_decay
        self.eval_metric = eval_metric
        self.pipeline_type = pipeline_type
        self.dataset_name = dataset_name
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.label2id = label2id
        self.id2label = id2label
        self.model_checkpoint = model_checkpoint
        self.processor = processor
        self.torch_weights_filename = torch_weights_filename
        self.device = device
        self.apply_lora = apply_lora
        self.load_weights = load_weights

    def collate_fn(self, data):
        """A custom collate function for the dataLoader
        Args:
            data(list): List of individual samples
        Returns:
            A dictionary containing the batched pixel values and labels
        """
        pixel_values = torch.stack([d["pixel_values"] for d in data])
        labels = torch.tensor([d["label"] for d in data])
        return {"pixel_values": pixel_values, "labels": labels}

    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics based on the predicted and true labels
        Args:
            eval_pred (tuple): Tuple containing predicted labels and true labels.
        Returns:
            A dictionary containing the computed evaluation metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return dict(accuracy=accuracy_score(predictions, labels))

    def execute_pipeline(self):
        """Execute the pipeline based on the specified pipeline type
        Returns:
            The Trainer object for training or prediction
        """
        # Load the ViT model for image classification
        model = ViTForImageClassification.from_pretrained(self.model_checkpoint, id2label=self.id2label, label2id=self.label2id, ignore_mismatched_sizes=True)
        if self.load_weights: model.load_state_dict(torch.load("./" + self.torch_weights_filename, map_location=torch.device(self.device.type)))
        if self.apply_lora: model = get_peft_model(model, peft_config)
        # Set the training arguments for the Trainer
        args = TrainingArguments(
            output_dir = self.dataset_name,
            save_strategy = "epoch",
            evaluation_strategy = "epoch",
            learning_rate = self.learning_rate,
            per_device_train_batch_size = self.per_device_train_batch_size,
            per_device_eval_batch_size = self.per_device_eval_batch_size,
            num_train_epochs = self.num_train_epochs,
            weight_decay = self.weight_decay,
            load_best_model_at_end = True,
            metric_for_best_model = self.eval_metric,
            logging_dir = 'logs',
            remove_unused_columns = False,
            deepspeed="./ds_config_zero3.json"
        )
        # Check the pipeline type and create the Trainer accordingly
        if self.pipeline_type.lower() == "train":
            executor = Trainer(
                model=model,
                args=args,
                train_dataset=self.train_ds,
                eval_dataset=self.val_ds,
                data_collator=self.collate_fn,
                compute_metrics=self.compute_metrics,
                tokenizer=self.processor
            )
        if self.pipeline_type.lower() == "predict":
            # Load the pre-trained weights for prediction
            executor = Trainer(
                model=model,
                args=args,
                train_dataset=self.train_ds,
                eval_dataset=self.val_ds,
                data_collator=self.collate_fn,
                compute_metrics=self.compute_metrics,
                tokenizer=self.processor
            )
        return executor

    def visualize_results(self, preds):
        """Visualize the evaluation results
        Args:
            preds(obj): A transformer object containing prediction outputs
        Returns:
            None
        """
        # Print the evaluation metrics
        print(f"\n\n{preds.metrics} \n")
        # Get the true labels and predicted labels
        y_true = preds.label_ids
        y_pred = preds.predictions.argmax(1)
        # Get the label names
        labels = self.test_ds.features['label'].names
        # Compute the confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Create a ConfusionMatrixDisplay and plot the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(xticks_rotation=45)
```

A class for performing the image classification process.

```python
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
```

Step 1: Compile and Preprocess Data

```python
pipe1 = setupPipeline(
    dataset_name='./ky_test_natural',
    train_size= 10000, #30416,
    test_size= 1242,
    validation_split=0.2,
    shuffle_data=True,
    model_checkpoint="google/vit-large-patch16-224-in21k"
)

train_dsx, test_dsx = pipe1.load_data()
train_dsx, val_dsx, test_dsx, id2labelx, label2idx, processorx = pipe1.preprocess_data(train_ds=train_dsx, test_ds=test_dsx)
dataset_loaded_train, dataset_loaded_test = pipe1.load_data()
train_ds, val_ds, test_ds, id2label, label2id, processor = pipe1.preprocess_data(train_ds=dataset_loaded_train, test_ds=dataset_loaded_test)

```

Set up LoRa for efficient fine-tuning

```python
%%bash
cat <<'EOT' > ds_config_zero3.json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
EOT

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)


```

Step 2: Train the model

```python
# Note:
# The first model checkpoint input was "google/vit-large-patch16-224-in21k", executed on 5 epochs but due to
# insufficient storage space, only the first 3 model checkpoints were saved.
# "./garbage/checkpoint-3000" was further fine-tuned to produce the final checkpoint "garbage/checkpoint-250-final"

%%time
run1 = runPipeline(
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    weight_decay=0.001,
    eval_metric="accuracy",
    pipeline_type="train",
    dataset_name='garbage',
    train_ds=train_ds,
    val_ds=val_ds,
    test_ds=test_ds,
    label2id=label2id,
    id2label=id2label,
    model_checkpoint="./garbage/checkpoint-3000", # "google/vit-large-patch16-224-in21k",
    processor=processor,
    torch_weights_filename=None,
    device=device,
    apply_lora=False,
    load_weights=False
)
executor1 = run1.execute_pipeline()
executor1.train()
torch.save(executor1.model.state_dict(), 'AML_p1_v5.pt')
```


Step 3: Inference

* Fine-tuned weights and model checkpoint can be downloaded via the Google drive link below:

* Weights: https://drive.google.com/file/d/10N5Lnb2Kd_3rZ6qniMAE2jnDPk3P7JPV/view?usp=sharing

* Checkpoint: https://drive.google.com/drive/folders/1-Kc6ymVShEIpCry59IKKmXw7K9XY10sc?usp=sharing


```python
%%bash
cat <<'EOT' > ds_config_zero3.json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
EOT


%%time
run2 = runPipeline(
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.001,
    eval_metric="accuracy",
    pipeline_type="predict",
    dataset_name='garbage',
    train_ds=train_ds,
    val_ds=val_ds,
    test_ds=test_ds,
    label2id=label2id,
    id2label=id2label,
    model_checkpoint="garbage/checkpoint-250-final",
    processor=processor,
    torch_weights_filename="AML_p1_v5.pt",
    device=device,
    apply_lora=False,
    load_weights=True
)
executor2 = run2.execute_pipeline()
preds = executor2.predict(test_ds)
run2.visualize_results(preds)
```

Visualize performance

<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Computer%20Vision/Garbage%20Classification/docs/images/pred_results.png" width="1200" />
</p>

---


## Sources

1. Channel News Asia. 2020. IN FOCUS: 'It is not easy, but it can be done' - The challenges of raising Singapore's recycling rate. https://www.channelnewsasia.com/singapore/in-focus-singapore-recycling-sustainability-blue-bins-waste-1339091 
2. Kaza, S., Yao, L. C., Bhada-Tata, P., & Van Woerden, F. (2018). What a Waste 2.0: A Global Snapshot of Solid Waste Management to 2050. Washington, DC: World Bank. https://doi.org/10.1596/978-1-4648-1329-0
3. N. V. Kumsetty, A. Bhat Nekkare, S. K. S. and A. Kumar M. 2018. TrashBox: Trash Detection and Classification using Quantum Transfer Learning. 31st Conference of Open Innovations. Association (FRUCT), 2022, pp. 125-130, doi: 10.23919/FRUCT54823.2022.9770922. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9770922&isnumber=9770880
4. He, K., Zhang, X., Ren, S., Sun, J. 2015. Deep Residual Learning for Image Recognition. arXiv.org. https://arxiv.org/abs/1512.03385 
5. Boesch, G. (n.d). Vision Transformer (ViT) in Image Recognition – 2023 Guide. https://viso.ai/deep-learning/vision-transformer-vit/
6. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021, June 3). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv.org. https://arxiv.org/abs/2010.11929
7. Tan, M., Le, Q., 2021. EfficientNetV2: Smaller Models and Faster Training. arXiv.org: https://arxiv.org/abs/2104.00298
8. Redmon, K. Divvala, S., Girshick, R., Farhadi, A. 2016, You Only Look Once: Unified, Real-Time Object Detection arXiv.org. https://arxiv.org/abs/1506.02640v5

