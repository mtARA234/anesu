
import streamlit as st
import numpy as np
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import joblib # Import joblib for loading models

# ===============================
# SETTINGS (MUST MATCH TRAINING)
# ===============================
FP_SIZE = 2048
RADIUS = 2

morgan = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=FP_SIZE)

def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(morgan.GetFingerprint(mol))

def combine_fp(drug, excipient):
    fp1 = smiles_to_fp(drug)
    fp2 = smiles_to_fp(excipient)
    if fp1 is None or fp2 is None:
        return None
    return np.concatenate([fp1, fp2])  # 4096

# ===============================
# LOAD MODELS (JSON SAFE)
# ===============================
targets = ["SERT","DAT","D2","D3","D4","5HT1A","5HT6","5HT7"]

clf_models = {}
reg_models = {}

for t in targets:
    # Load classifier models using joblib
    clf = joblib.load(f"{t}_clf.json")
    clf_models[t] = clf

    # Load regressor models using joblib
    reg = joblib.load(f"{t}_reg.json")
    reg_models[t] = reg

# Load toxicity model using joblib
tox_model = joblib.load("tox_model.json")

# Load compatibility model using joblib
compat_model = joblib.load("compat_model.json")

# ===============================
# EXCIPIENT DATABASE (EDITABLE) (No change here)
# ===============================
EXCIPIENTS = {
    "Lactose": "C(C1C(C(C(C(O1)O)O)O)O)O",
    "Microcrystalline Cellulose": "OCC1OC(O)C(O)C(O)C1O",
    "Magnesium Stearate": "CCCCCCCCCCCCCCCC(=O)O[Mg]OC(=O)CCCCCCCCCCCCCCCC",
    "Starch": "OCC1OC(O)C(O)C(O)C1O",
    "Povidone (PVP)": "C=CC(=O)N1CCCC1=O",
    "HPMC": "OCC1OC(O)C(O)C(O)C1O",
    "PEG": "OCCO",
    "Talc": "Mg3Si4O10(OH)2",
    "Sucrose": "C(C1C(C(C(C(O1)O)O)O)O)O",
    "Gelatin": "NCC(=O)O"
}

# ===============================
# UI (No change here)
# ===============================
st.set_page_config(page_title="AI Drug Discovery", layout="wide")

st.title("🧪 AI Drug Discovery Platform")

smiles = st.text_input("Enter Drug SMILES")

selected_targets = st.multiselect(
    "Select Targets",
    targets,
    default=["SERT","DAT"]
)

run_tox = st.checkbox("Predict Toxicity")

run_comp = st.checkbox("Check Drug–Excipient Compatibility")

selected_excipient = None
excipient_smiles = ""

if run_comp:
    selected_excipient = st.selectbox("Select Excipient", list(EXCIPIENTS.keys()))
    excipient_smiles = EXCIPIENTS[selected_excipient]

# ===============================
# PREDICTION (No change here)
# ===============================
if st.button("Run Prediction"):

    fp = smiles_to_fp(smiles)

    if fp is None:
        st.error("❌ Invalid SMILES")
        st.stop()

    st.subheader("📊 IC50 Predictions")

    for t in selected_targets:
        clf = clf_models[t]
        reg = reg_models[t]

        try:
            prob = clf.predict_proba(fp.reshape(1,-1))[0][1]
        except:
            st.error(f"{t} model issue → retrain required")
            continue

        if prob > 0.5:
            pic50 = reg.predict(fp.reshape(1,-1))[0]
            ic50 = 10**(-pic50) * 1e9

            st.success(
                f"{t}: Active | pIC50={pic50:.2f} | IC50={ic50:.2f} nM | Confidence={prob:.2f}"
            )
        else:
            st.warning(f"{t}: Inactive | Confidence={prob:.2f}")

    # ===============================
    # TOXICITY (No change here)
    # ===============================
    if run_tox:
        st.subheader("🧬 Toxicity")
        try:
            pred = tox_model.predict(fp.reshape(1,-1))[0]
            st.success("Toxic" if pred==1 else "Non-toxic")
        except:
            st.error("Toxicity model issue")

    # ===============================
    # COMPATIBILITY (No change here)
    # ===============================
    if run_comp:
        st.subheader("⚗️ Compatibility")

        comp_fp = combine_fp(smiles, excipient_smiles)

        if comp_fp is None:
            st.error("Invalid SMILES or Excipient")
        else:
            try:
                pred = compat_model.predict(comp_fp.reshape(1,-1))[0]
                st.success(
                    f"{selected_excipient}: {'Compatible' if pred==1 else 'Incompatible'}"
                )
            except:
                st.error("Compatibility model issue")
                