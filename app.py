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

with st.sidebar.expander("🟠 Bobinas de Helmholtz", expanded=False):
    I_helmholtz = st.slider("Corriente (A)", 0.0, 20.0, 5.0, 0.5, key="I_helmholtz")
    R_helmholtz = st.slider("Radio R (m)", 0.1, 2.0, 0.5, 0.05, key="R_helmholtz")
    st.caption(f"Separación d = R = {R_helmholtz} m")

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

# Cálculos Básicos
B_alambre_2d = calcular_campo_alambre(I_alambre, L_alambre, z_offset_alambre, N_elementos, r_2d)
B_espira_2d = calcular_campo_espira(I_espira, a_espira, z_offset_espira, N_elementos, r_2d)
B_total_2d = B_alambre_2d + B_espira_2d

B_alambre_3d = calcular_campo_alambre(I_alambre, L_alambre, z_offset_alambre, N_elementos, r_3d)
B_espira_3d = calcular_campo_espira(I_espira, a_espira, z_offset_espira, N_elementos, r_3d)
B_total_3d = B_alambre_3d + B_espira_3d

# Cálculos Helmholtz
# Bobina 1 en z = -R/2, Bobina 2 en z = +R/2
B_h1_2d = calcular_campo_espira(I_helmholtz, R_helmholtz, -R_helmholtz/2, N_elementos, r_2d)
B_h2_2d = calcular_campo_espira(I_helmholtz, R_helmholtz, R_helmholtz/2, N_elementos, r_2d)
B_helmholtz_2d = B_h1_2d + B_h2_2d

B_h1_3d = calcular_campo_espira(I_helmholtz, R_helmholtz, -R_helmholtz/2, N_elementos, r_3d)
B_h2_3d = calcular_campo_espira(I_helmholtz, R_helmholtz, R_helmholtz/2, N_elementos, r_3d)
B_helmholtz_3d = B_h1_3d + B_h2_3d

# ============================================================================
# VISUALIZACIÓN (Tabs Minimalistas)
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔴 Alambre Recto",
    "🔵 Espira Circular",
    "🟣 Superposición",
    "🟠 Bobinas de Helmholtz",
    "📚 Información"
])

def render_point_calculator(key_suffix, calc_func, *args, **kwargs):
    """Helper para renderizar la calculadora en un punto dentro de una tab"""
    with st.expander(f"📍 Calculadora de Campo en Punto ({key_suffix.capitalize()})"):
        c1, c2, c3 = st.columns(3)
        xt = c1.number_input("x (m)", value=0.0, step=0.1, format="%.2f", key=f"calc_x_{key_suffix}")
        yt = c2.number_input("y (m)", value=0.0, step=0.1, format="%.2f", key=f"calc_y_{key_suffix}")
        zt = c3.number_input("z (m)", value=0.0, step=0.1, format="%.2f", key=f"calc_z_{key_suffix}")
        
        if st.button("Calcular", key=f"calc_btn_{key_suffix}"):
            pt = np.array([[xt, yt, zt]])
            B_res = calc_func(*args, pt, **kwargs)
            B_norm = np.linalg.norm(B_res)
            
            st.markdown(f"**Magnitud |B|:** `{B_norm:.6e} T`")
            st.markdown(f"**Vector B:** `[{B_res[0,0]:.2e}, {B_res[0,1]:.2e}, {B_res[0,2]:.2e}] T`")

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
    
    # Calculadora específica para Alambre
    render_point_calculator("alambre", campo_alambre, I_alambre, L_alambre, N_elementos, z_offset=z_offset_alambre)

with tab2:
    render_view("Espira", B_espira_2d, B_espira_3d, 
                {'tipo': 'espira', 'a': a_espira, 'z_offset_espira': z_offset_espira}, "espira")
    
    # Calculadora específica para Espira
    render_point_calculator("espira", campo_espira, I_espira, a_espira, N_elementos, z_offset=z_offset_espira)

with tab3:
    render_view("Total", B_total_2d, B_total_3d, 
                {'tipo': 'ambos', 'L': L_alambre, 'a': a_espira, 
                 'z_offset_alambre': z_offset_alambre, 'z_offset_espira': z_offset_espira}, "total")
    
    # Calculadora para Superposición
    def calc_total(I_a, L_a, z_a, I_e, a_e, z_e, N, pt):
        Ba = campo_alambre(I_a, L_a, N, pt, z_offset=z_a)
        Be = campo_espira(I_e, a_e, N, pt, z_offset=z_e)
        return Ba + Be
        
    render_point_calculator("total", calc_total, I_alambre, L_alambre, z_offset_alambre, I_espira, a_espira, z_offset_espira, N_elementos)

with tab4:
    st.header("🟠 Bobinas de Helmholtz")
    st.markdown(f"**Configuración**: Dos espiras de radio $R={R_helmholtz}$ m separadas por una distancia $d=R$. Corriente $I={I_helmholtz}$ A.")
    
    render_view("Helmholtz", B_helmholtz_2d, B_helmholtz_3d,
                {'tipo': 'helmholtz', 'R': R_helmholtz}, "helmholtz")
    
    # Calculadora Helmholtz
    def calc_helmholtz(I, R, N, pt):
        B1 = campo_espira(I, R, N, pt, z_offset=-R/2)
        B2 = campo_espira(I, R, N, pt, z_offset=R/2)
        return B1 + B2
    
    render_point_calculator("helmholtz", calc_helmholtz, I_helmholtz, R_helmholtz, N_elementos)
    
    st.markdown("---")
    st.markdown("""
    ### Análisis de Bobinas de Helmholtz
    
    **¿Qué son?**
    Son un par de bobinas circulares idénticas, colocadas paralelamente y separadas por una distancia igual a su radio ($d = R$). Por ambas circula la misma corriente en el mismo sentido.
    
    **¿Para qué se utilizan?**
    Se utilizan para generar una región de **campo magnético casi uniforme** en el espacio entre ellas. Esto es fundamental en experimentos de física (como la medición de la relación carga-masa del electrón) y para cancelar campos magnéticos externos (como el campo terrestre).
    
    **Análisis de Resultados**
    *   **Uniformidad**: Observa en la vista 2D (corte XZ o YZ) que las líneas de campo en la región central (entre las bobinas) son casi paralelas y equidistantes. Esto indica que el gradiente del campo es mínimo ($\partial B / \partial z \approx 0$ y $\partial^2 B / \partial z^2 \approx 0$ en el centro).
    *   **Intensidad**: El campo en el centro es la suma constructiva de los campos de ambas bobinas.
    """)

with tab5:
    st.header("📚 Análisis y Teoría")
    
    st.markdown("""
    ### Preguntas de Análisis
    
    #### 1. ¿Cómo varía la magnitud del campo magnético con la distancia al conductor?
    *   **Alambre Recto**: Para un alambre infinito, la magnitud decae inversamente con la distancia radial ($B \\propto 1/r$). Cerca de los extremos de un alambre finito, este comportamiento se modifica.
    *   **Espira Circular**: En el eje de la espira, el campo decae rápidamente con la distancia $z$. Lejos de la espira ($z \\gg a$), se comporta como un dipolo magnético, decayendo como $1/z^3$.
    
    #### 2. ¿Son las líneas de campo magnético las esperadas?
    *   **Sí**. Las visualizaciones muestran claramente:
        *   Líneas concéntricas alrededor del alambre (regla de la mano derecha).
        *   Líneas que pasan por el centro de la espira y se curvan alrededor de ella, cerrándose sobre sí mismas.
        *   Esto confirma la naturaleza solenoidal del campo magnético ($\\nabla \\cdot \\vec{B} = 0$), es decir, no tiene monopolos; las líneas siempre se cierran.
    
    #### 3. ¿Qué puede decir de la validez de los resultados obtenidos a partir de la Ley de Biot-Savart?
    *   **Validez**: La Ley de Biot-Savart es fundamental para la magnetostática (corrientes estacionarias). Los resultados numéricos obtenidos mediante la discretización del conductor (suma de Riemann de $I d\\vec{l} \\times \\vec{r} / r^3$) convergen a la solución teórica exacta a medida que aumentamos el número de elementos $N$.
    *   **Limitaciones Numéricas**: Muy cerca de los conductores ($r \\to 0$), el término $1/r^2$ diverge, lo que puede causar inestabilidades numéricas (singularidades) si el punto de evaluación coincide exactamente con un segmento de corriente. En la simulación, esto se maneja evitando evaluar exactamente en el conductor o usando un mallado fino.
    
    ---
    ### Teoría Fundamental
    
    La **Ley de Biot-Savart** describe el campo magnético generado por una corriente estacionaria:
    
    $$
    \\vec{B}(\\vec{r}) = \\frac{\\mu_0}{4\\pi} \\int \\frac{I \\, d\\vec{l}' \\times (\\vec{r} - \\vec{r}')}{|\\vec{r} - \\vec{r}'|^3}
    $$
    """)
    
    st.info("💡 **Tip**: Usa la calculadora en cada pestaña para verificar numéricamente cómo disminuye el campo al aumentar la distancia (x, y o z).")

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
