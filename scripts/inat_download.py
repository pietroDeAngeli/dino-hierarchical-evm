from torchvision.datasets import INaturalist

dataset = INaturalist(
    root="./data",
    version="2021_train_mini",
    download=True
)