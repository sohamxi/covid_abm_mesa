# COVID-19 Agent-Based Model (Mesa)

An agent-based model for simulating COVID-19 spread using the Mesa ABM framework. Features multi-layer contact networks (household, workplace, school, community), age-stratified disease parameters, vaccination, and economic impact tracking.

## Requirements

- **Python >= 3.10** (Mesa 3.3+ does not support Python 3.9 or earlier)

## Setup

```bash
# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick test
```bash
python3 -c "from model import InfectionModel; m = InfectionModel(N=200); [m.step() for _ in range(60)]; print(f'R0={m.R0:.2f}')"
```

### Interactive visualization (Mesa SolaraViz)
```bash
solara run server.py
```

### Plotly Dash dashboard
```bash
python3 dashboard.py
# Open http://localhost:8050
```

### Batch scenario comparison
```bash
python3 batch_runner.py
```

### Jupyter notebook
```bash
jupyter notebook Report_Covid_Modelling.ipynb
```

## Project Structure

| File | Description |
|---|---|
| `model.py` | `InfectionModel` - main Mesa model with SEIR dynamics and interventions |
| `agent.py` | `Human` agent with age-stratified disease, vaccination, and economic behavior |
| `disease_params.py` | Age-stratified COVID-19 parameters (IFR, hospitalization, vaccine efficacy) |
| `contact_network.py` | Multi-layer contact network (household, workplace, school, community) |
| `server.py` | Mesa SolaraViz interactive visualization |
| `dashboard.py` | Plotly Dash production dashboard |
| `batch_runner.py` | Parameter sweep and sensitivity analysis |
| `data_loader.py` | Real-world OWID COVID-19 data loader for comparison |
| `Report_Covid_Modelling.ipynb` | Background and methodology notebook |
