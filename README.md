# causal-discovery

This project investigates and compares two causal inference methods for multivariate time series:
- PCMCI 
- Liang-Kleeman Information Flow Rate 

## Folder Structure

causal-discovery/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── notebooks/              
│   └── main.ipynb
├── src/                    
│   └── todo.py
├── external/               
│   ├── causal_comp/  # From ddocquier/Causal_Comp
│   │   └── model_2var.py         
│   │   └── model_6var.py         
│   │   └── function_liang_nvar.py  
│   └── tigramite/     
│       └──
├── data/             # Synthetic datasets
│   └── 2D_series.npy
│   └── 6D_series.npy
└── results/               
    └── plots/
    └── npy/
	   └── 2D_liang.npy
	   └── 6D_liang.npy