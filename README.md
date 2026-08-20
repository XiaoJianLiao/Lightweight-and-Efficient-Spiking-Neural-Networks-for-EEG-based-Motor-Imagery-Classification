# SNN for MI-EEG Classification
Forked from original https://doi.org/10.1016/j.bspc.2024.107000 "Constructing Lightweight and Efficient Spiking Neural Networks for EEG-based Motor Imagery Classification" by [@XiaoJianLiao](https://github.com/XiaoJianLiao)

## Requirements
spikingjelly 0.0.0.0.14
pytorch 1.12.1
scikit-learn 1.5.1
scipy 1.13.1

## File list
### Original Scripts
	Pipeline.py: An example for constructing lightweight and efficient spiking neural networks for EEG-based motor imagery classification

	Train_ANN.py: Training LENet under ANN structure

	Tset_SNN.py : Convert trained LENet to SNN

### DataSets
  [BCICIV_2a_gdf](https://www.bbci.de/competition/iv/download/index.html): The preprocessed dataset for testing the framework

### Jupyter Notebooks (for Google Colab)
  SNN_for_EEG_based_Motor_Imagery_Classification.ipynb: All the scripts in one file
