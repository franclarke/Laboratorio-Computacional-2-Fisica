import streamlit as st
import numpy as np
from alambre import campo_alambre
from espira import campo_espira
from visualizacion_plotly import crear_grafico_2d_plotly, crear_grafico_3d_plotly

# Configuración de la página
st.set_page_config(
    page_title="Visualizador de Campo Magnético - Ley de Biot-Savart",
    page_icon="🧲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🧲 Visualizador de Campo Magnético")
st.markdown("### Ley de Biot-Savart: Alambre Recto y Espira Circular")
st.markdown("---")

# ============================================================================
# SIDEBAR: CONTROLES
# ============================================================================
st.sidebar.title("⚙️ Parámetros de Simulación")

st.sidebar.markdown("### 🔴 Alambre Recto")
I_alambre = st.sidebar.slider(
    "Corriente del alambre (A)",
    min_value=0.0, max_value=20.0, value=10.0, step=0.5,
    help="Corriente eléctrica que circula por el alambre"
)
L_alambre = st.sidebar.slider(
    "Longitud del alambre (m)",
    min_value=0.5, max_value=5.0, value=2.0, step=0.1,
    help="Longitud total del alambre"
)
z_offset_alambre = st.sidebar.slider(
    "Posición Z del alambre (m)",
    min_value=-2.0, max_value=2.0, value=0.0, step=0.1,
    help="Desplazamiento del alambre a lo largo del eje Z"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔵 Espira Circular")
I_espira = st.sidebar.slider(
    "Corriente de la espira (A)",
    min_value=0.0, max_value=20.0, value=5.0, step=0.5,
    help="Corriente eléctrica que circula por la espira"
)
a_espira = st.sidebar.slider(
    "Radio de la espira (m)",
    min_value=0.1, max_value=2.0, value=0.5, step=0.05,
    help="Radio de la espira circular"
)
z_offset_espira = st.sidebar.slider(
    "Posición Z de la espira (m)",
    min_value=-2.0, max_value=2.0, value=0.0, step=0.1,
    help="Desplazamiento de la espira a lo largo del eje Z"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Calidad de Visualización")
resolucion_2d = st.sidebar.slider(
    "Resolución malla 2D",
    min_value=10, max_value=30, value=20, step=2,
    help="Número de puntos en cada dirección para el gráfico 2D"
)
resolucion_3d = st.sidebar.slider(
    "Resolución malla 3D",
    min_value=4, max_value=12, value=8, step=1,
    help="Número de puntos en cada dirección para el gráfico 3D"
)
N_elementos = st.sidebar.slider(
    "Elementos de corriente (N)",
    min_value=500, max_value=3000, value=1000, step=100,
    help="Número de segmentos para integración numérica (mayor = más preciso pero más lento)"
)

# ============================================================================
# CÁLCULO DE CAMPOS
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### 👁️ Vistas 2D")
plano_vista = st.sidebar.radio(
    "Plano de visualización 2D",
    ["XY (Planta)", "XZ (Perfil)", "YZ (Perfil Lateral)"],
    index=1, # Default a XZ para ver mejor la espira
    help="Seleccione el plano de corte para los gráficos 2D"
)

# Configuración dinámica según el plano
if plano_vista == "XY (Planta)":
    coord_fija_label = "z"
    coord_fija_val = st.sidebar.slider("Posición Z del plano (m)", -2.0, 2.0, 0.0, 0.1)
    
    u_label, v_label = "x (m)", "y (m)"
    
    # Malla 2D
    u = np.linspace(-1.5, 1.5, resolucion_2d)
    v = np.linspace(-1.5, 1.5, resolucion_2d)
    uu, vv = np.meshgrid(u, v)
    
    # Puntos 3D: (x, y, z_fijo)
    r_2d = np.c_[uu.ravel(), vv.ravel(), np.full_like(uu, coord_fija_val).ravel()]
    
    idx_u, idx_v = 0, 1 # Índices de componentes vectoriales (Bx, By)

elif plano_vista == "XZ (Perfil)":
    coord_fija_label = "y"
    coord_fija_val = st.sidebar.slider("Posición Y del plano (m)", -2.0, 2.0, 0.0, 0.1)
    
    u_label, v_label = "x (m)", "z (m)"
    
    # Malla 2D
    u = np.linspace(-1.5, 1.5, resolucion_2d)
    v = np.linspace(-1.5, 1.5, resolucion_2d)
    uu, vv = np.meshgrid(u, v)
    
    # Puntos 3D: (x, y_fijo, z)
    r_2d = np.c_[uu.ravel(), np.full_like(uu, coord_fija_val).ravel(), vv.ravel()]
    
    idx_u, idx_v = 0, 2 # Índices de componentes vectoriales (Bx, Bz)

else: # YZ
    coord_fija_label = "x"
    coord_fija_val = st.sidebar.slider("Posición X del plano (m)", -2.0, 2.0, 0.0, 0.1)
    
    u_label, v_label = "y (m)", "z (m)"
    
    # Malla 2D
    u = np.linspace(-1.5, 1.5, resolucion_2d)
    v = np.linspace(-1.5, 1.5, resolucion_2d)
    uu, vv = np.meshgrid(u, v)
    
    # Puntos 3D: (x_fijo, y, z)
    r_2d = np.c_[np.full_like(uu, coord_fija_val).ravel(), uu.ravel(), vv.ravel()]
    
    idx_u, idx_v = 1, 2 # Índices de componentes vectoriales (By, Bz)


# Malla 3D (siempre igual)
x_3d = np.linspace(-1.5, 1.5, resolucion_3d)
y_3d = np.linspace(-1.5, 1.5, resolucion_3d)
z_3d = np.linspace(-1.5, 1.5, resolucion_3d)
xx_3d, yy_3d, zz_3d = np.meshgrid(x_3d, y_3d, z_3d)
r_3d = np.c_[xx_3d.ravel(), yy_3d.ravel(), zz_3d.ravel()]

# Calcular campos con caché para optimización
@st.cache_data
def calcular_campo_alambre(I, L, z_off, N, r_shape):
    r_flat = r_shape.reshape(-1, 3)
    return campo_alambre(I, L, N, r_flat, z_offset=z_off)

@st.cache_data
def calcular_campo_espira(I, a, z_off, N, r_shape):
    r_flat = r_shape.reshape(-1, 3)
    return campo_espira(I, a, N, r_flat, z_offset=z_off)

with st.spinner('Calculando campos magnéticos...'):
    # Campos 2D
    B_alambre_2d_full = calcular_campo_alambre(I_alambre, L_alambre, z_offset_alambre, N_elementos, r_2d)
    B_espira_2d_full = calcular_campo_espira(I_espira, a_espira, z_offset_espira, N_elementos, r_2d)
    B_total_2d_full = B_alambre_2d_full + B_espira_2d_full
    
    # Campos 3D
    B_alambre_3d = calcular_campo_alambre(I_alambre, L_alambre, z_offset_alambre, N_elementos, r_3d)
    B_espira_3d = calcular_campo_espira(I_espira, a_espira, z_offset_espira, N_elementos, r_3d)
    B_total_3d = B_alambre_3d + B_espira_3d

# ============================================================================
# TABS DE VISUALIZACIÓN
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔴 Alambre Recto",
    "🔵 Espira Circular",
    "🟣 Superposición",
    "📍 Punto de Prueba",
    "📚 Información"
])

# --- TAB 1: ALAMBRE ---
with tab1:
    st.header("Campo Magnético del Alambre Recto")
    st.markdown(f"**Parámetros**: I = {I_alambre} A, L = {L_alambre} m, z_offset = {z_offset_alambre} m")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Vista 2D: Plano {plano_vista.split()[0]}")
        st.caption(f"Corte en {coord_fija_label} = {coord_fija_val} m")
        
        Bu_2d = B_alambre_2d_full[:, idx_u].reshape(uu.shape)
        Bv_2d = B_alambre_2d_full[:, idx_v].reshape(vv.shape)
        
        fig_2d_alambre = crear_grafico_2d_plotly(
            uu, vv, Bu_2d, Bv_2d,
            titulo="",
            geometria={'tipo': 'alambre', 'L': L_alambre, 'z_offset_alambre': z_offset_alambre},
            xlabel=u_label, ylabel=v_label
        )
        st.plotly_chart(fig_2d_alambre, use_container_width=True, key="chart_2d_alambre")
    
    with col2:
        st.subheader("Vista 3D")
        fig_3d_alambre = crear_grafico_3d_plotly(
            xx_3d.ravel(), yy_3d.ravel(), zz_3d.ravel(),
            B_alambre_3d[:, 0], B_alambre_3d[:, 1], B_alambre_3d[:, 2],
            titulo="",
            geometria={'tipo': 'alambre', 'L': L_alambre, 'z_offset_alambre': z_offset_alambre}
        )
        st.plotly_chart(fig_3d_alambre, use_container_width=True, key="chart_3d_alambre")

# --- TAB 2: ESPIRA ---
with tab2:
    st.header("Campo Magnético de la Espira Circular")
    st.markdown(f"**Parámetros**: I = {I_espira} A, a = {a_espira} m, z_offset = {z_offset_espira} m")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Vista 2D: Plano {plano_vista.split()[0]}")
        st.caption(f"Corte en {coord_fija_label} = {coord_fija_val} m")
        
        Bu_espira_2d = B_espira_2d_full[:, idx_u].reshape(uu.shape)
        Bv_espira_2d = B_espira_2d_full[:, idx_v].reshape(vv.shape)
        
        fig_2d_espira = crear_grafico_2d_plotly(
            uu, vv, Bu_espira_2d, Bv_espira_2d,
            titulo="",
            geometria={'tipo': 'espira', 'a': a_espira, 'z_offset_espira': z_offset_espira},
            xlabel=u_label, ylabel=v_label
        )
        st.plotly_chart(fig_2d_espira, use_container_width=True, key="chart_2d_espira")
    
    with col2:
        st.subheader("Vista 3D")
        fig_3d_espira = crear_grafico_3d_plotly(
            xx_3d.ravel(), yy_3d.ravel(), zz_3d.ravel(),
            B_espira_3d[:, 0], B_espira_3d[:, 1], B_espira_3d[:, 2],
            titulo="",
            geometria={'tipo': 'espira', 'a': a_espira, 'z_offset_espira': z_offset_espira}
        )
        st.plotly_chart(fig_3d_espira, use_container_width=True, key="chart_3d_espira")

# --- TAB 3: SUPERPOSICIÓN ---
with tab3:
    st.header("Superposición: Alambre + Espira")
    st.markdown(f"**Alambre**: I = {I_alambre} A, L = {L_alambre} m, z = {z_offset_alambre} m")
    st.markdown(f"**Espira**: I = {I_espira} A, a = {a_espira} m, z = {z_offset_espira} m")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Vista 2D: Plano {plano_vista.split()[0]}")
        st.caption(f"Corte en {coord_fija_label} = {coord_fija_val} m")
        
        Bu_total_2d = B_total_2d_full[:, idx_u].reshape(uu.shape)
        Bv_total_2d = B_total_2d_full[:, idx_v].reshape(vv.shape)
        
        fig_2d_total = crear_grafico_2d_plotly(
            uu, vv, Bu_total_2d, Bv_total_2d,
            titulo="",
            geometria={
                'tipo': 'ambos',
                'L': L_alambre, 'a': a_espira,
                'z_offset_alambre': z_offset_alambre,
                'z_offset_espira': z_offset_espira
            },
            xlabel=u_label, ylabel=v_label
        )
        st.plotly_chart(fig_2d_total, use_container_width=True, key="chart_2d_total")
    
    with col2:
        st.subheader("Vista 3D")
        fig_3d_total = crear_grafico_3d_plotly(
            xx_3d.ravel(), yy_3d.ravel(), zz_3d.ravel(),
            B_total_3d[:, 0], B_total_3d[:, 1], B_total_3d[:, 2],
            titulo="",
            geometria={
                'tipo': 'ambos',
                'L': L_alambre, 'a': a_espira,
                'z_offset_alambre': z_offset_alambre,
                'z_offset_espira': z_offset_espira
            }
        )
        st.plotly_chart(fig_3d_total, use_container_width=True, key="chart_3d_total")

# --- TAB 4: PUNTO DE PRUEBA ---
with tab4:
    st.header("Cálculo en Punto Específico")
    st.markdown("Ingrese las coordenadas de un punto para calcular el campo magnético en esa ubicación.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        x_test = st.number_input("Coordenada x (m)", value=0.3, step=0.1, format="%.2f")
    with col2:
        y_test = st.number_input("Coordenada y (m)", value=0.0, step=0.1, format="%.2f")
    with col3:
        z_test = st.number_input("Coordenada z (m)", value=0.2, step=0.1, format="%.2f")
    
    punto_test = np.array([[x_test, y_test, z_test]])
    
    B_alambre_punto = campo_alambre(I_alambre, L_alambre, N_elementos, punto_test, z_offset_alambre)
    B_espira_punto = campo_espira(I_espira, a_espira, N_elementos, punto_test, z_offset_espira)
    B_total_punto = B_alambre_punto + B_espira_punto
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🔴 Alambre", f"{np.linalg.norm(B_alambre_punto):.6e} T")
        st.code(f"Bx = {B_alambre_punto[0,0]:.6e} T\nBy = {B_alambre_punto[0,1]:.6e} T\nBz = {B_alambre_punto[0,2]:.6e} T")
    
    with col2:
        st.metric("🔵 Espira", f"{np.linalg.norm(B_espira_punto):.6e} T")
        st.code(f"Bx = {B_espira_punto[0,0]:.6e} T\nBy = {B_espira_punto[0,1]:.6e} T\nBz = {B_espira_punto[0,2]:.6e} T")
    
    with col3:
        st.metric("🟣 Total", f"{np.linalg.norm(B_total_punto):.6e} T")
        st.code(f"Bx = {B_total_punto[0,0]:.6e} T\nBy = {B_total_punto[0,1]:.6e} T\nBz = {B_total_punto[0,2]:.6e} T")

# --- TAB 5: INFORMACIÓN ---
with tab5:
    st.header("📚 Ley de Biot-Savart")
    
    st.markdown("""
    ### Teoría
    
    La **Ley de Biot-Savart** describe el campo magnético generado por una corriente estacionaria en un conductor:
    
    $$
    \\vec{B}(\\vec{r}) = \\frac{\\mu_0}{4\\pi} \\int \\frac{I \\, d\\vec{l}' \\times (\\vec{r} - \\vec{r}')}{|\\vec{r} - \\vec{r}'|^3}
    $$
    
    **Donde:**
    - $\\vec{B}(\\vec{r})$: campo magnético en la posición $\\vec{r}$
    - $\\mu_0 = 4\\pi \\times 10^{-7}$ T·m/A: permeabilidad del vacío
    - $I$: corriente eléctrica
    - $d\\vec{l}'$: elemento diferencial de corriente
    - $\\vec{r}'$: posición del elemento de corriente
    
    ### Configuraciones Implementadas
    
    #### 🔴 Alambre Recto
    - Alambre recto de longitud $L$ con corriente $I$
    - El alambre se ubica a lo largo del eje Z
    - El campo circula alrededor del alambre (regla de la mano derecha)
    
    #### 🔵 Espira Circular
    - Espira circular de radio $a$ con corriente $I$
    - La espira se ubica en el plano XY
    - El campo es más intenso en el centro de la espira
    
    #### 🟣 Superposición
    - El campo total es la suma vectorial de los campos individuales
    - Demuestra el **principio de superposición** del electromagnetismo
    
    ### Controles Interactivos
    
    Use el panel lateral para ajustar:
    - **Corrientes**: Intensidad del campo magnético
    - **Dimensiones**: Geometría de las fuentes
    - **Posiciones**: Ubicación en el espacio 3D
    - **Resolución**: Calidad vs rendimiento
    
    ### Interpretación de los Gráficos
    
    - **Mapas de calor**: Intensidad del campo magnético
    - **Vectores/Conos**: Dirección del campo
    - **Colores**: Magnitud (azul = débil, amarillo = fuerte)
    """)
    
    st.info("💡 **Tip**: Inténtee colocar el alambre en el eje de la espira (ambos z_offset = 0) para ver una configuración simétrica interesante.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
    Laboratorio Computacional 2 - Física II | Ley de Biot-Savart<br>
    Visualizador interactivo desarrollado con Streamlit & Plotly
    </div>
    """,
    unsafe_allow_html=True
)
