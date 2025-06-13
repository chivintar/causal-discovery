# causal-discovery

This project investigates and compares different causal inference methods for multivariate time series, including:
- PCMCI (from Tigramite)
- Liang-Kleeman Information Flow Rate (LIFR, from Causal_Comp)

## Folder Structure

causal-discovery/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── notebooks/              # experiments
│   └── main.ipynb
│   
│
├── src/                    # own scripts
│   └── example.py
│   └── ex.py
│   └── e.py
│
├── external/               
│   ├── causal_comp/        # From ddocquier/Causal_Comp
│   │   └── model_2var.py         
│   │   └── model_6var.py         
│   │   └── function_liang_nvar.py         
│   └── tigramite/          # From jakobrunge/tigramite
│       └── p
│
├── data/                   # Synthetic or real datasets (small or symbolic)
│   └── 
│
└── results/                # Outputs: plots, p-values, causal graphs
    └── plots/
    └── csv/
