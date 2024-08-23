from preprocessing.preprocess import setupPipeline
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from transformers import TrainingArguments, Trainer ViTForImageClassification, AutoImageProcessor
import torch
from peft import get_peft_model
import numpy as np

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