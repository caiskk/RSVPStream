# RSVPStream

RSVPStream is a real-time EEG classification system that processes brain signals to detect responses to Rapid Serial Visual Presentation (RSVP) stimuli. This repository provides the necessary code and tools to deploy, train, and test the RSVP-based brain-computer interface on EEG data.

## Features
- Real-time RSVP signal processing.
- EEG data preprocessing and classification.
- Support for various machine learning models.
- Modular and scalable architecture.

## Prerequisites
Before getting started, ensure the following dependencies are installed:

- Python 3.8 or higher
- NumPy
- SciPy
- scikit-learn
- MNE (for EEG data processing)
- TensorFlow or PyTorch (depending on the model used)

You can install the required dependencies with the following command:

```bash
pip install -r requirements.txt
```


Installation
Clone the repository:

```bash
git clone --recurse-submodules https://github.com/KylinGR/RSVPStream.git
```

Navigate to the project directory:


```bash
cd RSVPStream
```

If you cloned the repository directly, you can use the following command to update the submodules:


```bash
git submodule update --init --recursive

```

Install the required Python packages:


```bash
pip install -r requirements.txt
```