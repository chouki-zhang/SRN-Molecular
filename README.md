# SRN molecular design
<p align="center">
  <img width="700" src="https://github.com/chouki-zhang/SRN-Molecular/blob/master/introduction/intro.png" alt="SRN">
</p>

**SRN(Synthetic-routes-navigated) molecular design** project is a digital laboratory for molecular design. Here you can try chemical reaction, property analysis and so on. Everything will be done by AI, all you need to do is typing the ingredients for the reaction!


The following table list some core packages required.

| Package        | Version    |
| -------------- | ---------- |
| `PyTorch`      | 1.4.0      |
| `pymatgen`     | 2020.1.28  |
| `matminer`     | 0.6.2      |
| `mordred`      | 1.2.0      |
| `scipy`        | 1.3.1      |
| `scikit-learn` | 0.22.1     |
| `xgboost`      | 1.0.0      |
| `pandas`       | 1.0.0      |
| `rdkit`        | 2019.03.3  |
| `jupyter`      | 1.0.0      |
| `seaborn`      | 0.9.0      |
| `matplotlib`   | 3.1.2      |
| `plotly`       | 4.5.0      |

## Usage
The deeplearning model is too large to upload here, please download from the following url and place it in "./models/transformer_models/"

https://drive.google.com/file/d/1H45ltTmp0j4WMahybbZxZv4918HKGblc/view?usp=sharing

Launch web on your server by the following command:

```bash
python app.py
```

## Tutorial
1, Open the homepage, click "Try reaction online!".
<p align="center">
  <img width="700" src="https://github.com/chouki-zhang/SRN-Molecular/blob/master/introduction/1.png" alt="SRN">
</p>

2, Input reactants (SMILES type), click "Synthesis".
<p align="center">
  <img width="700" src="https://github.com/chouki-zhang/SRN-Molecular/blob/master/introduction/2.png" alt="SRN">
</p>

3, Possiable products will be shown automatically, click "Analysis" for more analysis.
<p align="center">
  <img width="700" src="https://github.com/chouki-zhang/SRN-Molecular/blob/master/introduction/3.png" alt="SRN">
</p>

4, The structure images will be shown.
<p align="center">
  <img width="700" src="https://github.com/chouki-zhang/SRN-Molecular/blob/master/introduction/4.png" alt="SRN">
</p>

5, Click product of interest for physical property information.
<p align="center">
  <img width="700" src="https://github.com/chouki-zhang/SRN-Molecular/blob/master/introduction/5.png" alt="SRN">
</p>

