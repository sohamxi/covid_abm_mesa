# COVID-19 Agent-Based Model (Mesa)

Simulate COVID-19 spread and **compare intervention strategies** (lockdown, masks, vaccination, etc.) to evaluate their health AND economic tradeoffs. Built to help decision-makers choose policies by running side-by-side scenario comparisons.

Features: multi-layer contact networks (household, workplace, school, community), age-stratified disease parameters, vaccination with waning immunity, and wealth impact tracking across social strata.

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

### Dashboard (scenario comparison)
```bash
python3 dashboard.py
# Open http://localhost:8050
```

Select intervention scenarios, configure shared parameters (population, transmission probability, simulation days), and click **Run Comparison** to see side-by-side health vs economic outcomes.

### Batch scenario comparison (headless)
```bash
python3 batch_runner.py
```

Runs predefined scenarios in the terminal and prints summary statistics.

### Quick test
```bash
python3 -c "from model import InfectionModel; m = InfectionModel(N=200); [m.step() for _ in range(60)]; print(f'R0={m.R0:.2f}')"
```

### Jupyter notebook
```bash
jupyter notebook Report_Covid_Modelling.ipynb
```

## Project Structure

| File | Description |
|---|---|
| `model.py` | `InfectionModel` — Mesa model with SEIR dynamics, interventions, configurable `max_steps` |
| `agent.py` | `Human` agent with age-stratified disease, vaccination, and economic behavior |
| `disease_params.py` | Age-stratified COVID-19 parameters (IFR, hospitalization, vaccine efficacy) |
| `contact_network.py` | Multi-layer contact network (household, workplace, school, community) |
| `dashboard.py` | Plotly Dash dashboard for intervention scenario comparison |
| `batch_runner.py` | Parameter sweep and sensitivity analysis (headless) |
| `data_loader.py` | Real-world OWID COVID-19 data loader for calibration |
| `Report_Covid_Modelling.ipynb` | Background and methodology notebook |
