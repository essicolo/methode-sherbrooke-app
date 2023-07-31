import datetime
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import altair as alt
from scipy.optimize import curve_fit

# Title
st.write("**CONFIDENTIEL**. Pour utilisation par le Ministère des Transports et de la Mobilité durable du Québec et par l'Universoté de Sherbrooke seulement. L'application sera désactivée le 1ier septembre 2023.")
st.title("Méthode Sherbrooke")

# Functions
def create_df(f, x_range, n=100, scale='linear', *args):
    """
    Crée un data frame selon un range en x et une fonction
    """
    if scale == 'log':
        x = np.logspace(x_range[0], x_range[1], n)
    else:
        x = np.linspace(x_range[0], x_range[1], n)

    df = pd.DataFrame({'x': x})
    df['y'] = f(df['x'], *args)
    return(df)

def WLR_to_VWC(x):
    """
    Transforme un log-ratio de x et de sa valeur complémentaire une proportion
    """
    return np.exp(x) / (1+np.exp(x))

def ρd_f(θ, ρw, Gs, Sr):
    """
    Calcul de la densité sèche en fonction d'indices de phase et de densité
    """
    porosity = θ/Sr
    ρs = Gs*ρw
    ρd = (1-porosity)*ρs
    return(ρd)

def w_f(θ, ρw, ρd):
    """
    Calcul de la teneur en eau gravimétrique
    """
    w = θ * ρw / ρd
    return(w)


## PSD functions
### would be easier to render the rosin-rammler equation from d85 and cu than use converters
def g1_f(cu, d85):
    x = 0.6
    y = 0.1
    z = 0.85
    g1 = -np.log(1 - z)/d85**(np.log(np.log(1 - x)/np.log(1 - y))/np.log(cu))
    return(g1)

def g2_f(cu):
    x = 0.6
    y = 0.1
    g2 = np.log(np.log(1 - x)/np.log(1 - y))/np.log(cu)
    return(g2)

def d85_f(g1, g2):
    z = 0.85
    d85 = (- np.log(1 - z) / g1) ** (1/g2)
    return(d85)

def cu_f(g2):
    x = 0.6
    y = 0.1
    return(np.exp(np.log(np.log(1 - x)/np.log(1 - y))/g2))

def psd_rosin_rammler(d, g1, g2):
    prop_passing = 1 - np.exp(-g1 * d**g2)
    return(prop_passing)


# Mise à l'échelle
## modèle c
Xm_c = np.array([20.97505497, 18.49365115, 2.76432237, 2277.61077303])
Xsd_c = np.array([3.75021039e+01, 1.38905689e+01, 5.41425509e-02, 2.49298347e+02])
Ym_c = -1.8885546039017012
Ysd_c = 0.503274059593005

## modèle sr
Xm_sr = np.array([76.25952122, 16.52049029,  2.73000215, -2.35141919])
Xsd_sr = np.array([7.42381273e+01, 9.74213339e+00, 4.79454043e-02, 5.15905434e-01])
Ym_sr = 0.8915630270359038
Ysd_sr = 0.5868257858692756

# Load model
class ExpoAssoTransformation(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(ExpoAssoTransformation, self).__init__()
    def call(self, inputs):   
        y_hat = (inputs[2]-inputs[1]) * (1 - tf.math.exp(-inputs[3] * inputs[0])) + inputs[1]   
        return(y_hat)


@st.cache_resource()
def load_models():
    custom_objects = {"ExpoAssoTransformation": ExpoAssoTransformation}
    cmodel = tf.keras.models.load_model('models/cmodel_nn.h5', custom_objects=custom_objects)
    srmodel = tf.keras.models.load_model('models/sr_nn.h5')
    return [cmodel, srmodel]

with st.spinner("Loading Model...."):
    models = load_models()
    cmodel = models[0]
    config = cmodel.get_config()
    srmodel = models[1]

# Inputs
with st.sidebar:
    st.header("Sol")
    psd_csv = st.file_uploader(label='Fichier granulo (csv)', type='csv')

if psd_csv is not None:
    psd_csv_df = pd.read_csv(psd_csv)
    psd_csv_df.columns = ['x', 'y']

    with st.sidebar:
        psd_optimized = st.button('Optimiser selon les données')
    if psd_optimized:
        # Fit the function to the data
        popt, pcov = curve_fit(psd_rosin_rammler, psd_csv_df['x'].values, psd_csv_df['y'].values)

        # Print the optimal parameters
        d85 = d85_f(popt[0], popt[1])
        cu = cu_f(popt[1])

with st.sidebar:
    if psd_csv is not None and psd_optimized:
        cu = st.slider("Coefficient d'uniformité", min_value=1.0, max_value=100.0, value=cu.item(), step=0.1)
        d85 = 10**st.slider("log10(D85)", min_value=-3.0, max_value=3.0, value=np.log10(d85).item(), step=0.01)
    else:
        cu = st.slider("Coefficient d'uniformité", min_value=1, max_value=100, value=30)
        d85 = 10**st.slider("log10(D85)", min_value=-3.0, max_value=3.0, value=0.0, step=0.01)
    gs = st.slider("Densité spécifique des grains (g/cm³)", min_value=2.5, max_value=2.9, value=2.7)
    ρw = st.slider("ρw (g/cm³)", min_value=900, max_value=1100, value=1000)
    st.header("Sonde")
    pr1 = st.slider("Mesure de la sonde **avant** inondation", min_value=1900, max_value=3500, value=2200)
    pr2 = st.slider("Mesure de la sonde **après** inondation", min_value=1900, max_value=3500, value=2600)


g1 = g1_f(cu, d85)
g2 = g2_f(cu)
psd_df = create_df(psd_rosin_rammler, [-3, 3], 100, 'log', g1, g2)

X_c1 = np.array([[cu, d85, gs, pr1]])
X_c2 = np.array([[cu, d85, gs, pr2]])
Xsc_c1 = (X_c1 - Xm_c) / Xsd_c
Xsc_c2 = (X_c2 - Xm_c) / Xsd_c
Ysc_c1 = cmodel.predict([Xsc_c1[:, 3], Xsc_c1[:, :3]])
Ysc_c2 = cmodel.predict([Xsc_c2[:, 3], Xsc_c2[:, :3]])
Y_c1 = Ysc_c1 * Ysd_c + Ym_c
Y_c2 = Ysc_c2 * Ysd_c + Ym_c
θ_R1 = WLR_to_VWC(Y_c1)[0][0]
θ_R2 = WLR_to_VWC(Y_c2)[0][0]

ae_df = pd.DataFrame({'x': np.arange(1800, 3000, 10)})
ae_df['y'] = cmodel.predict(
    [
        (ae_df['x'].values - Xm_c[3]) / Xsd_c[3],
        np.tile(Xsc_c1[:, :3], (ae_df.shape[0], 1))
    ],
    verbose=0
)  * Ysd_c + Ym_c
ae_df['y'] = WLR_to_VWC(ae_df['y'])

# Sr model
srmodel_input = np.array([[cu, d85, gs, Y_c1.item()]])
sr_opt = WLR_to_VWC(srmodel.predict(
    (srmodel_input - Xm_sr) / Xsd_sr,
    verbose=0
)  * Ysd_sr + Ym_sr)

dry_density = ρd_f(θ_R2, ρw, gs, sr_opt)
grav_wc = w_f(θ_R1, ρw, dry_density)

# limite de temps
now = datetime.datetime.now()
limit = datetime.datetime(2023, 9, 1)
if now < limit:
    st.write(
        "Masse volumique sèche du sol ($ρ_d$): **{}** kg/m³.".format(int(dry_density))
    )
    st.write(
        "Teneur en eau gravimétrique du sol ($w$): **{}** %.".format(np.round(grav_wc.item() * 100, 1))
    )
else:
    st.write("Le logiciel ne retourne plus la masse volumique sèche et la teneur en eau gravimétrique plus à partir du 2023-09-01.")

psd_plot = alt.Chart(psd_df).mark_line(color='#ff4b4b').encode(
    x=alt.X('x', scale=alt.Scale(type='log'), title='Diamètre (mm)'), y=alt.Y('y', title="Proportion passante")
).properties(width=300, height=300)

ae_plot = (
    alt.Chart(ae_df).mark_line(color='#ff4b4b').encode(
        x=alt.X('x', scale=alt.Scale(domain=[ae_df['x'].min(), ae_df['x'].max()]), title='Valeur de la sonde'),
        y=alt.Y('y', title="Teneur en eau volumétrique")
    ).properties(width=300, height=300)
    # id 1st measurement
    + alt.Chart(
        pd.DataFrame({'x': [ae_df['x'].min(), pr1], 'y': [θ_R1, θ_R1]})).mark_line(color='#ff4b4b', strokeWidth = 0.75).encode(
        x = 'x', y = 'y'
    )
    + alt.Chart(pd.DataFrame({'x': [pr1, pr1], 'y': [0, θ_R1]})).mark_line(color='#ff4b4b', strokeWidth = 0.75).encode(
        x = 'x', y = 'y'
    )
    + alt.Chart(pd.DataFrame({'x': [pr1], 'y': [θ_R1]})).mark_point(color='#ff4b4b', size=50).encode(
        x = 'x', y = 'y'
    )
    + alt.Chart(pd.DataFrame({'x': ae_df['x'].min(), 'y': [θ_R1]})).mark_text(
        text=str(np.round(θ_R1, 4)),
        color = '#ff4b4b',
        align="left",
        baseline="bottom"
    ).encode(
        x='x', y = 'y'
    )
    + alt.Chart(pd.DataFrame({'x': [pr1], 'y': [0]})).mark_text(
        text=str(pr1),
        color = '#ff4b4b',
        align="right",
        baseline="bottom"
    ).encode(
        x='x', y = 'y'
    )
    # id 2nd meadurement
    + alt.Chart(
        pd.DataFrame({'x': [ae_df['x'].min(), pr2], 'y': [θ_R2, θ_R2]})).mark_line(color='#ff4b4b', strokeWidth = 0.75).encode(
        x = 'x', y = 'y'
    )
    + alt.Chart(pd.DataFrame({'x': [pr2, pr2], 'y': [0, θ_R2]})).mark_line(color='#ff4b4b', strokeWidth = 0.75).encode(
        x = 'x', y = 'y'
    )
    + alt.Chart(pd.DataFrame({'x': [pr2], 'y': [θ_R2]})).mark_point(color='#ff4b4b', size=50).encode(
        x = 'x', y = 'y'
    )
    + alt.Chart(pd.DataFrame({'x': ae_df['x'].min(), 'y': [θ_R2]})).mark_text(
        text=str(np.round(θ_R2, 4)),
        color = '#ff4b4b',
        align="left",
        baseline="bottom"
    ).encode(
        x='x', y = 'y'
    )
    + alt.Chart(pd.DataFrame({'x': [pr2], 'y': [0]})).mark_text(
        text=str(pr2),
        color = '#ff4b4b',
        align="right",
        baseline="bottom"
    ).encode(
        x='x', y = 'y'
    )
)

if psd_csv is not None:
    psd_csv_plot = alt.Chart(psd_csv_df).mark_point(color='#ff4b4b').encode(
        x=alt.X('x', scale=alt.Scale(type='log', domain = [psd_df['x'].min(), psd_df['x'].max()])), y='y'
    ).properties(width=300, height=300)
    psd_plot = psd_plot + psd_csv_plot


st.altair_chart(psd_plot | ae_plot, use_container_width=True)

# Notes
st.write("*Version pre-alpha*. Pour des raisons de confidentialité de la méthode Sherbrooke, les réseaux de neurones utilisés sont correctes, mais volontairement sous-optimisés.")
st.write("Essi Parent, ing., Ph.D.")
