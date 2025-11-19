import streamlit as st
import numpy as np
from alambre import campo_alambre
from espira import campo_espira
from visualizacion_plotly import crear_grafico_2d_plotly, crear_grafico_3d_plotly

# Configuración de la página
st.set_page_config(
    page_title="Biot-Savart Visualizer",
    page_icon="🧲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para look "Google-style"
st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        font-family: 'Google Sans', 'Roboto', sans-serif;
        font-weight: 500;
    }
    .stRadio > div {
        flex-direction: row;
        gap: 20px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Título principal minimalista
st.title("Visualizador de Campo Magnético")
st.caption("Exploración interactiva de la Ley de Biot-Savart")

# ============================================================================
# SIDEBAR: CONTROLES FÍSICOS (Minimalista)
# ============================================================================
st.sidebar.header("Parámetros")

with st.sidebar.expander("🔴 Alambre Recto", expanded=True):
    I_alambre = st.slider("Corriente (A)", 0.0, 20.0, 10.0, 0.5, key="I_alambre")
    L_alambre = st.slider("Longitud (m)", 0.5, 5.0, 2.0, 0.1, key="L_alambre")
    z_offset_alambre = st.slider("Posición Z (m)", -2.0, 2.0, 0.0, 0.1, key="z_alambre")

with st.sidebar.expander("🔵 Espira Circular", expanded=True):
    I_espira = st.slider("Corriente (A)", 0.0, 20.0, 5.0, 0.5, key="I_espira")
    a_espira = st.slider("Radio (m)", 0.1, 2.0, 0.5, 0.05, key="a_espira")
    z_offset_espira = st.slider("Posición Z (m)", -2.0, 2.0, 0.0, 0.1, key="z_espira")

# Configuración Avanzada (Oculta por defecto)
with st.sidebar.expander("⚙️ Configuración Avanzada", expanded=False):
    resolucion_2d = st.slider("Res. 2D", 10, 30, 20, 2)
    resolucion_3d = st.slider("Res. 3D", 4, 12, 8, 1)
    N_elementos = st.slider("Elementos N", 500, 3000, 1000, 100)

# ============================================================================
# BARRA DE HERRAMIENTAS SUPERIOR (Vistas)
# ============================================================================
# Contenedor para controles de vista globales
with st.container(border=True):
    col_view1, col_view2 = st.columns([1, 2])
    
    with col_view1:
        st.markdown("**👁️ Plano de Visualización**")
        plano_vista = st.radio(
            "Seleccione plano",
            ["XY (Planta)", "XZ (Perfil)", "YZ (Lateral)"],
            index=1,
            label_visibility="collapsed",
            horizontal=True
        )
    
    with col_view2:
        # Lógica de corte según plano
        if "XY" in plano_vista:
            coord_fija_label, u_label, v_label = "z", "x (m)", "y (m)"
            idx_u, idx_v = 0, 1
            min_c, max_c = -2.0, 2.0
        elif "XZ" in plano_vista:
            coord_fija_label, u_label, v_label = "y", "x (m)", "z (m)"
            idx_u, idx_v = 0, 2
            min_c, max_c = -2.0, 2.0
        else: # YZ
            coord_fija_label, u_label, v_label = "x", "y (m)", "z (m)"
            idx_u, idx_v = 1, 2
            min_c, max_c = -2.0, 2.0
            
        coord_fija_val = st.slider(
            f"Corte en {coord_fija_label} (m)", 
            min_c, max_c, 0.0, 0.1,
            help=f"Ajusta la posición del plano de corte a lo largo del eje {coord_fija_label}"
        )

# ============================================================================
# CÁLCULO DE CAMPOS (Optimizado)
# ============================================================================

# Generación de mallas
if "XY" in plano_vista:
    u = np.linspace(-1.5, 1.5, resolucion_2d)
    v = np.linspace(-1.5, 1.5, resolucion_2d)
    uu, vv = np.meshgrid(u, v)
    r_2d = np.c_[uu.ravel(), vv.ravel(), np.full_like(uu, coord_fija_val).ravel()]
elif "XZ" in plano_vista:
    u = np.linspace(-1.5, 1.5, resolucion_2d)
    v = np.linspace(-1.5, 1.5, resolucion_2d)
    uu, vv = np.meshgrid(u, v)
    r_2d = np.c_[uu.ravel(), np.full_like(uu, coord_fija_val).ravel(), vv.ravel()]
else: # YZ
    u = np.linspace(-1.5, 1.5, resolucion_2d)
    v = np.linspace(-1.5, 1.5, resolucion_2d)
    uu, vv = np.meshgrid(u, v)
    r_2d = np.c_[np.full_like(uu, coord_fija_val).ravel(), uu.ravel(), vv.ravel()]

# Malla 3D
x_3d = np.linspace(-1.5, 1.5, resolucion_3d)
y_3d = np.linspace(-1.5, 1.5, resolucion_3d)
z_3d = np.linspace(-1.5, 1.5, resolucion_3d)
xx_3d, yy_3d, zz_3d = np.meshgrid(x_3d, y_3d, z_3d)
r_3d = np.c_[xx_3d.ravel(), yy_3d.ravel(), zz_3d.ravel()]

# Funciones cacheadas
@st.cache_data
def calcular_campo_alambre(I, L, z_off, N, r_shape):
    r_flat = r_shape.reshape(-1, 3)
    return campo_alambre(I, L, N, r_flat, z_offset=z_off)

@st.cache_data
def calcular_campo_espira(I, a, z_off, N, r_shape):
    r_flat = r_shape.reshape(-1, 3)
    return campo_espira(I, a, N, r_flat, z_offset=z_off)

# Cálculos
B_alambre_2d = calcular_campo_alambre(I_alambre, L_alambre, z_offset_alambre, N_elementos, r_2d)
B_espira_2d = calcular_campo_espira(I_espira, a_espira, z_offset_espira, N_elementos, r_2d)
B_total_2d = B_alambre_2d + B_espira_2d

B_alambre_3d = calcular_campo_alambre(I_alambre, L_alambre, z_offset_alambre, N_elementos, r_3d)
B_espira_3d = calcular_campo_espira(I_espira, a_espira, z_offset_espira, N_elementos, r_3d)
B_total_3d = B_alambre_3d + B_espira_3d

# ============================================================================
# VISUALIZACIÓN (Tabs Minimalistas)
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["Alambre", "Espira", "Superposición", "Info"])

def render_view(title, B_2d, B_3d, geo, key_suffix):
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.markdown("#### Vista 2D")
            Bu = B_2d[:, idx_u].reshape(uu.shape)
            Bv = B_2d[:, idx_v].reshape(vv.shape)
            fig = crear_grafico_2d_plotly(
                uu, vv, Bu, Bv, 
                titulo="", geometria=geo, 
                xlabel=u_label, ylabel=v_label, plano=plano_vista
            )
            st.plotly_chart(fig, use_container_width=True, key=f"2d_{key_suffix}_{plano_vista}")
            
    with col2:
        with st.container(border=True):
            st.markdown("#### Vista 3D")
            fig3 = crear_grafico_3d_plotly(
                xx_3d.ravel(), yy_3d.ravel(), zz_3d.ravel(),
                B_3d[:,0], B_3d[:,1], B_3d[:,2],
                titulo="", geometria=geo
            )
            st.plotly_chart(fig3, use_container_width=True, key=f"3d_{key_suffix}")

with tab1:
    render_view("Alambre", B_alambre_2d, B_alambre_3d, 
                {'tipo': 'alambre', 'L': L_alambre, 'z_offset_alambre': z_offset_alambre}, "alambre")

with tab2:
    render_view("Espira", B_espira_2d, B_espira_3d, 
                {'tipo': 'espira', 'a': a_espira, 'z_offset_espira': z_offset_espira}, "espira")

with tab3:
    render_view("Total", B_total_2d, B_total_3d, 
                {'tipo': 'ambos', 'L': L_alambre, 'a': a_espira, 
                 'z_offset_alambre': z_offset_alambre, 'z_offset_espira': z_offset_espira}, "total")

with tab4:
    st.markdown("### 📚 Teoría y Uso")
    st.info("Utiliza el selector de planos arriba para cambiar entre vistas de planta (XY) y perfil (XZ, YZ).")
    st.markdown("""
    **Ley de Biot-Savart**:
    $$ \\vec{B}(\\vec{r}) = \\frac{\\mu_0}{4\\pi} \\int \\frac{I \\, d\\vec{l}' \\times (\\vec{r} - \\vec{r}')}{|\\vec{r} - \\vec{r}'|^3} $$
    """)
    
    # Punto de prueba simplificado
    with st.expander("📍 Calculadora de Campo en Punto"):
        c1, c2, c3 = st.columns(3)
        xt = c1.number_input("x", 0.3)
        yt = c2.number_input("y", 0.0)
        zt = c3.number_input("z", 0.2)
        
        pt = np.array([[xt, yt, zt]])
        Bt = campo_alambre(I_alambre, L_alambre, N_elementos, pt, z_offset_alambre) + \
             campo_espira(I_espira, a_espira, N_elementos, pt, z_offset_espira)
        
        st.metric("Magnitud Total", f"{np.linalg.norm(Bt):.6e} T")
        st.caption(f"Vector: {Bt[0]}")
