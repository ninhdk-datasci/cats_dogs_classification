#type: ignore 
import argparse 
import os 
import shutil 
import numpy as np 
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from cats_dogs_classification.dataset.cats_dogs_dataset import CatsDogsDataset 
from cats_dogs_classification.models.cnn import CNN 
from pathlib import Path
from sklearn.model_selection import train_test_split 
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter  
import mlflow   
import mlflow.pytorch
from tqdm import tqdm 
from sklearn.metrics import accuracy_score

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent 
SEED = 42 


def get_args(): 
    parser = argparse.ArgumentParser(description="Train a CNN model for cats and dogs classification") 
    parser.add_argument("--data_dir", type=str, default="data_collection/train", help="Path to the training data directory") 
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training") 
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train") 
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer") 
    parser.add_argument("--model_save_path", type=str, default="cnn_model.pth", help="Path to save the trained model") 
    parser.add_argument("--image_width", type=int, default=128, help="Width of the input images")
    parser.add_argument("--image_height", type=int, default=128, help="Height of the input images")
    parser.add_argument("--log_path", type=str, default="tensorboard", help="Path to save the TensorBoard logs") 
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to save the model checkpoints")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of the dataset to include in the test split")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes for classification") 
    parser.add_argument("--trained_model_path", type=str, default="cnn_model", help="Path to the trained model for evaluation")
    return parser.parse_args()

def train(opt): 
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")) 
    mlflow.set_experiment("cats_dogs_classification") 
    # args = get_args() 
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(123) 
    else: 
        torch.manual_seed(123)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((opt.image_height, opt.image_width))
        ]
    )

    dataset = CatsDogsDataset(PROJECT_DIR / opt.data_dir, transform=transform)
    training_set, testing_set = train_test_split(dataset, test_size=opt.test_size, random_state=SEED)

    training_generator = DataLoader(
        training_set, 
        batch_size=opt.batch_size, 
        shuffle=True, 
        drop_last=False, 
        num_workers=2
    )
    testing_generator = DataLoader(
        testing_set, 
        batch_size=opt.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=2
    )
    # if os.path.isdir(PROJECT_DIR / opt.log_path): 
    #     shutil.rmtree(PROJECT_DIR / opt.log_path) 
    # os.mkdir(PROJECT_DIR / opt.log_path)



    model = CNN(num_classes=opt.num_classes) 

    if torch.cuda.is_available(): model.cuda() 

    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)

    if opt.checkpoint_path: 
        checkpoint = torch.load(opt.checkpoint_path) 
        model.load_state_dict(checkpoint["model_state_dict"]) 
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1 
        best_accuracy = checkpoint["best_accuracy"] 
        mlflow_run_id = checkpoint["mlflow_run_id"]   
    else: 
        start_epoch = 0 
        best_accuracy = 0 
        mlflow_run_id = None

    # lan sau: with mlflow.start_run(run_id=mlflow_run_id) as run:
    with mlflow.start_run(run_id = mlflow_run_id) as run:
        run_id = run.info.run_id 
        mlflow.log_params(
            {
                "num_epochs": opt.num_epochs, 
                "batch_size": opt.batch_size, 
                "learning_rate": opt.learning_rate, 
                "image_width": opt.image_width, 
                "image_height": opt.image_height, 
                "optimizer": "Adam", 
                "loss_function": "CrossEntropyLoss", 
                "seed": SEED, 
                "test_size": opt.test_size, 
                "mlflow_run_id": run_id
            }
        )

        num_iters = len(training_generator) 
        best_accuracy = best_accuracy
        writer = SummaryWriter(log_dir=PROJECT_DIR / opt.log_path) # Write tensorboard logs 
        num_iters_test = len(testing_generator)
        for epoch in range(start_epoch, opt.num_epochs): 
            model.train() 
            progress_bar = tqdm(training_generator) 
            for iter, (image, label) in enumerate(progress_bar): 
                if torch.cuda.is_available(): 
                    image = image.cuda()
                    label = label.cuda() 

                output = model(image) 
                loss = criterion(output, label)  # ////////////// 
                accuracy = accuracy_score(label.cpu().numpy(), torch.argmax(output, dim=1).cpu().numpy())
                progress_bar.set_description(f"Epoch: {epoch+1}/{opt.num_epochs}, Iteration: {iter+1}/{num_iters}, Loss: {loss.item():.4f}")
                writer.add_scalar("Train/Accuracy", accuracy, epoch * num_iters + iter)  
                writer.add_scalar("Train/Loss", loss.item(), epoch * num_iters + iter)

                mlflow.log_metric(
                    {
                        "Train/Accuracy": accuracy, 
                        "Train/Loss": loss.item() 
                    }, step = epoch * num_iters + iter 
                )

                # backpropagation
                optimizer.zero_grad() 
                loss.backward() 
                optimizer.step()

            model.eval() 

            all_labels = []
            all_preds = [] 

            for iter, (image, label) in enumerate(testing_generator): 
                if torch.cuda.is_available(): 
                    image = image.cuda()  
                    label = label.cuda()  
                all_labels.extend(label)  
                with torch.no_grad(): 
                    output = model(image) 
                    preds = torch.argmax(output, dim=1) 
                    all_preds.extend(preds) 
                    accuracy_val = accuracy_score(label.cpu().numpy(), preds.cpu().numpy()) 
                    loss_val = criterion(output, label) 
                    
                    writer.add_scalar("Test/Accuracy", accuracy_val, epoch * num_iters_test + iter)
                    writer.add_scalar("Test/Loss", loss_val.item(), epoch * num_iters_test + iter)

                    mlflow.log_metric(
                        {
                            "Test/Accuracy": accuracy_val, 
                            "Test/Loss": loss_val.item()
                        }, step = epoch * num_iters_test + iter 
                    )
            all_labels = [label.item() for label in all_labels] 
            all_preds = [pred.item() for pred in all_preds] 
            total_accuracy = accuracy_score(all_labels, all_preds) 

            is_best = total_accuracy > best_accuracy 

            if is_best: 
                best_accuracy = total_accuracy 

            checkpoint = {
                "run_id": run_id,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), 
                "best_accuracy": best_accuracy, 
                "val_accuracy": total_accuracy 
            }

             # create a new log directory for TensorBoard logs 
            log_dir = PROJECT_DIR / opt/log_path 
            log_dir.mkdir(parents=True, exist_ok=True)

            last_checkpoint_path = PROJECT_DIR / opt.trained_model_path / f"epoch_{epoch}.pth"
            torch.save(checkpoint, last_checkpoint_path)

            mlflow.log_artifact(
                last_checkpoint_path, 
                artifact_path=f"checkpoints"
            )

            if is_best: 
                best_checkpoint_path = PROJECT_DIR / opt.trained_model_path / "best_checkpoint.pth"
                # checkpoint["best_accuracy"] = total_accuracy
                torch.save(checkpoint, best_checkpoint_path) 
                mlflow.log_artifact(
                    best_checkpoint_path, 
                    artifact_path=f"checkpoints"
                )

if __name__ == "__main__": 
    args = get_args() 
    train(args)

            




            
                    



















    


