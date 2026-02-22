import torch
import torchvision
from torchvision.transforms import v2

import time
import numpy as np

from argparse import ArgumentParser

from io import BytesIO
from PIL import Image

class AbsModel(torch.nn.Module):
    def __init__(self):
        super(AbsModel, self).__init__()


    def train_on_data(self, train_dataset, val_dataset=None, epochs=10, lr=1e-3, optimiser=None, verbose=False, logger=None, cuda=True):
        device = 'cuda' if cuda and torch.cuda.is_available() else 'cpu'
        self.to(device)

        if not optimiser:
            opt = torch.optim.Adam(params=self.parameters(), lr=lr)
        else:
            opt = optimiser

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=50)
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(1, epochs+1):
            epoch_start_time = time.time()
            self.train()
            train_losses = []
            train_correct = 0
            train_total = 0

            for idx, data in enumerate(train_dataset):

                batch_start_time = time.time()

                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                opt.zero_grad()
                out = self.forward(inputs)
                #out = torch.nn.functional.softmax(out, dim=1) # Softmax done implicitly in the Cross Entropy loss

                loss = loss_fn(out, labels)
                loss.backward()
                opt.step()
                train_losses.append(loss.to('cpu').detach().numpy())

                predictions = out.argmax(dim=1)
                train_total += labels.shape[0]
                train_correct += int((predictions == labels).sum())


                if verbose:
                    if logger:
                        logger.info("Batch {}/{}: {:.3f}s".format(idx, len(train_dataset), time.time()-batch_start_time))
                    else:
                        print("Batch {}/{}: {:.3f}s".format(idx, len(train_dataset), time.time()-batch_start_time), end='\r')

            if logger:
                logger.info("Epoch {}: Took {:.3f}s".format(epoch, time.time()-epoch_start_time))
            else:
                print("\nEpoch {}: Took {:.3f}s".format(epoch, time.time()-epoch_start_time))

            if val_dataset != None:
                self.eval()
                val_start_time = time.time()
                val_losses = []
                val_correct = 0
                val_total = 0

                for data in val_dataset:
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    out = self.forward(inputs)
                    #out = torch.nn.functional.softmax(out, dim=1)

                    loss = loss_fn(out, labels)
                    val_losses.append(loss.cpu().detach().numpy())

                    predictions = out.argmax(dim=1)
                    val_total += labels.shape[0]
                    val_correct += int((predictions == labels).sum())



                if logger:
                    logger.info("Train acc: {:.4f}, Train loss: {:.4f} - Val acc: {:.4f}, Val loss: {:.4f}, Took: {:.3f}s"
                        .format(train_correct/train_total, np.mean(train_losses), val_correct/val_total, np.mean(val_losses), time.time()-val_start_time))
                else:
                    print("Train acc: {:.4f}, Train loss: {:.4f} - Val acc: {:.4f}, Val loss: {:.4f}, Took: {:.3f}s"
                        .format(train_correct/train_total, np.mean(train_losses), val_correct/val_total, np.mean(val_losses), time.time()-val_start_time))

                # Update scheduler based on validation results
                scheduler.step(np.mean(val_losses))

    def evaluate(self, test_dataset, logger=None, cuda=True):
        device = 'cuda' if cuda and torch.cuda.is_available() else 'cpu'
        self.to(device)
        self.eval()
        eval_start_time = time.time()
        correct = 0
        total = 0

        for data in test_dataset:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            out = self.forward(inputs)
            predictions = out.argmax(dim=1)
            total += labels.shape[0]
            correct += int((predictions == labels).sum())

        if logger:
            logger.info("Test set acc: {:.4f} - {:.3f}s"
                    .format(correct/total, time.time()-eval_start_time))
        else:
            print("Test set acc: {:.4f} - {:.3f}s"
                    .format(correct/total, time.time()-eval_start_time))

    def save_model(self):
        torch.save(self.state_dict(), "mnist_model.pth")
        print("Model saved!")

    def load_model(self):
        self.load_state_dict(torch.load("mnist_model.pth", weights_only=True, map_location=torch.device('cpu')))


class MNISTModel(AbsModel):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.fc1 = torch.nn.Linear(7*7*64, 1024)
        self.fc2 = torch.nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.Dropout(p=0.5)(x)

        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.Dropout(p=0.5)(x)
        
        x = torch.nn.Flatten()(x)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.Dropout(p=0.5)(x)
        logits = self.fc2(x)

        out = logits
        return out

    def embed(self, x):
        self.eval()
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)

        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.Flatten()(x)
        emb = self.fc1(x)

        # 1024 dimensional embedding
        return emb

    def predict(self, x):
        self.eval()
        out = self.forward(x)
        return torch.softmax(out, dim=1)

    def output_predict(self, x):
        res = self.predict(x)
        return res.detach().numpy()

    

def transform_mnist_dataset(dataset):
    """
    Transform the MNIST like datasets, scaling the values and expanding dimensions
    (includes random Affine transformation).

    Parameters:
    dataset (list of (numpy.ndarray, int)): The dataset to transform

    Returns:
    - ret_dataset (list of (numpy.ndarray, int)): The transformed dataset
    """
    # Set up transform pipeline
    dataset_transforms = torch.nn.Sequential(
        torchvision.transforms.Normalize([0.], [255.]),
        torchvision.transforms.RandomAffine(degrees=30, translate=(0.4, 0.4), scale=(0.3, 1.7)),
    )

    # Make it a JIT script!
    transform_script = torch.jit.script(dataset_transforms)

    prepared_dataset = []
    for image, label in dataset:
        image = torchvision.transforms.functional.pil_to_tensor(image).type(torch.float32)
        prepared_dataset.append((transform_script(image), label)) 
    
    return prepared_dataset


input_transform = torchvision.transforms.Compose([
    v2.PILToTensor(),
    v2.Resize(size=(28, 28)),
    v2.ToDtype(torch.float32, scale=True), # Scales image to 0,1
])

def bytes_to_tensor(bytes):
    image = Image.open(BytesIO(bytes)).convert("L")
    image = input_transform(image)
    return image




########################################### Run the code!
def train_model():
    print("Training model!")
    batch_size = 64
    epochs = 150
    cuda = True

    # Load MNIST
    train_mnist = torchvision.datasets.MNIST('/tmp', train=True, download=True)
    test_mnist = torchvision.datasets.MNIST('/tmp', train=False, download=True)

    train_dataset_full = transform_mnist_dataset(train_mnist)
    val_ratio = 0.2
    train_dataset = train_dataset_full[:int(len(train_dataset_full)-len(train_dataset_full)*val_ratio)]
    validation_dataset = train_dataset_full[int(len(train_dataset_full)-len(train_dataset_full)*val_ratio):]
    test_dataset = transform_mnist_dataset(test_mnist)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    standard_model = MNISTModel()
    standard_model.train_on_data(train_loader, val_loader, epochs=epochs, lr=1e-4, verbose=False, logger=None, cuda=cuda)
    print("\nEvaluation")
    standard_model.eval()
    standard_model.evaluate(test_loader, logger=None, cuda=cuda)

    standard_model.save_model()

def test_model():
    print("Testing model!")
    batch_size = 64
    cuda = True

    # Load MNIST
    test_mnist = torchvision.datasets.MNIST('/tmp', train=False, download=True)
    test_dataset = transform_mnist_dataset(test_mnist)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    standard_model = MNISTModel()
    standard_model.load_model()
    print("\nEvaluation")
    standard_model.eval()
    standard_model.evaluate(test_loader, logger=None, cuda=cuda)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Specifies training of the model (default False)")
    args = parser.parse_args()

    if args.train:
        train_model()
    else:
        test_model()
