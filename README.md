# DeepLearningProject3
### UNM CS429
# Authors
### Benjamin Liu, Danyelle Loffredo, and Jared Bock


# About

* This project uses Support Vector Machines, Neural Networks, and Convolutional Neural networks to mp3 files into one of 6 genres. This is done by first processing the train mp3 datasets with SVD, MFCC and a Spectrograms. The processed data is then used to train each of the tree classifiers. The twsting datasets are also processed the same way as the training datasets and are used to predict the genres of the mp3 files. 
# Installation

To install this project, you can clone the git repo to your local machine or, in GitHub, under the ```Code``` tab, click on the Green Code button, and the choose ```Download Zip```. 

The testing files and training files must be downloaded from: https://www.kaggle.com/competitions/cs529-project-3-audio/data. 
Files are too large to include in our repo but must be placed in the same working directory as the rest of the project. The folder must be extracted in such a way that the train and test folders are in the same working directory as the rest of the code.
![](Resources/codebutton.png)

# Usage 
For running SVM from the command line in a Linux terminal:

* cd into the directory where "Svm.py", "train.csv", "train" directory, "test" directory, and "test_idx.csv" are located (these should all be in the same directory with the exact names listed).

* Once in the correct directory, simply type 
```python Svm.py``` into the terminal.

* For Windows, the process is the same, however to run, you will need to enter ```py Svm.py``` and follow the prompts outlined as above.

* A second option for running the code on a Linux/UNIX system is, once located in the correct directory, type ``` make run``` into the terminal.

For running NeuralNetworks.py from the command line in a Linux terminal:

* cd into the directory where NeuralNetwork.py, "train.csv", "train" directory, "test" directory, and "test_idx.csv" are located (these should all be in the same directory with the exact names listed).

* Once in the correct directory, simply type 
```python NeuralNetwork.py``` into the terminal.

* For Windows, the process is the same, however to run, you will need to enter ```py NeuralNetwork.py``` and follow the prompts outlined as above.

* A second option for running the code on a Linux/UNIX system is, once located in the correct directory, type ``` make run``` into the terminal.

For running CNN.py from the command line in a Linux terminal:

* cd into the directory where LogReg.py, "train.csv", "train" directory, "test" directory, and "test_idx.csv" are located (these should all be in the same directory with the exact names listed).

* Once in the correct directory, simply type 
```python CNN.py``` into the terminal.

* For Windows, the process is the same, however to run, you will need to enter ```py CNN.py``` and follow the prompts outlined as above.

* A second option for running the code on a Linux/UNIX system is, once located in the correct directory, type ``` make run``` into the terminal.

For running Representation.py from the command line in a Linux terminal:
*You will be prompted to give paths the you mp3 files and train/test csvs. Verify these paths and that they have appropriate data.

* cd into the directory where Representation.py, "train.csv", "train" directory, "test" directory, and "test_idx.csv" are located (these should all be in the same directory with the exact names listed).

* Once in the correct directory, simply type 
```python LogReg.py``` into the terminal and respond to prompts.

* For Windows, the process is the same, however to run, you will need to enter ```py Representation.py``` and follow the prompts outlined as above.

* A second option for running the code on a Linux/UNIX system is, once located in the correct directory, type ``` make run``` into the terminal.


# Our environment
```bash
python --version
3.8.3
pandas.__version__
1.0.5
scipy.__version__
1.8.0
matplotlib.__version__
3.5.1
Tensorflow.__version__
2.4.1
librosa.__version__
0.9.1