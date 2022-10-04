from datasets import load_dataset, Dataset

## Load the dataset from disk. The dataset is stored in the `data` directory, split into train, test and val sets.
## The inputs for the model are the images (jpg), and the labels are numpy files of tensor of shape (27, 6).
## The labels are the one-hot encoded representation of the cube state (colors).

dataset = load_dataset("image_folder", 
                        data_files={"train": "data/train",
                                    "test": "data/test", 
                                    "val": "data/val"},
                        split=["train", "test", "val"],
                        label_type="numpy",
                        image_ext=".jpg")

print(dataset)

