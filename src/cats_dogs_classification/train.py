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
    return parser.parse_args()
def train(opt): 
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
    training_set, testing_set = train_test_split(dataset, test_size=0.2, random_state=SEED)

    training_generator = DataLoader(
        training_set, 
        batch_size=opt.batch_size, 
        shuffle=True, 
        drop_last=False, 
        num_workers=-1 
    )

    testing_generator = DataLoader(
        testing_set, 
        batch_size=opt.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=-1
    )

    if os.path.isdir(opt.log_path): 
        shutil.rmtree(opt.log_path) 
    os.mkdir(opt.log_path)
    writer 

    model = CNN(num_classes=2) 

    if torch.cuda.is_available(): model.cuda() 

    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)




    # print(len(training_set), len(test_set))

# train() 
    


