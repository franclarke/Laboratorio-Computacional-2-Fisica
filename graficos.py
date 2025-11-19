import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def graficar_2d(x, y, Bx, By, titulo="Campo Magnético", geometria=None):
    """
    Grafica campo magnético en 2D con vectores.
    """
    plt.figure(figsize=(8, 8))
    
    # Calcular magnitud
    B_mag = np.sqrt(Bx**2 + By**2)
    
    # --- NORMALIZACIÓN ROBUSTA ---
    # Clipear la magnitud para visualización para evitar que singularidades dominen
    B_visual_max = np.percentile(B_mag, 90) if B_mag.size > 0 else 1.0
    if B_visual_max == 0: B_visual_max = 1.0
    
    # Normalizar vectores
    # Evitar división por cero
    B_mag_safe = np.where(B_mag == 0, 1e-9, B_mag)
    Bx_unit = Bx / B_mag_safe
    By_unit = By / B_mag_safe
    
    # Escalar vectores: longitud base * factor de intensidad atenuado
    # Esto asegura que se vea la dirección en todas partes
    scale_factor = np.minimum(B_mag / B_visual_max, 1.0)
    # Mezcla: 50% longitud fija, 50% variable
    lengths = 0.5 + 0.5 * scale_factor
    
    Bx_vis = Bx_unit * lengths
    By_vis = By_unit * lengths
    
    # Colorear por magnitud (logarítmico o clipeado)
    # Usamos clipeado para consistencia
    colors = np.minimum(B_mag, np.percentile(B_mag, 95))
    
    plt.quiver(x, y, Bx_vis, By_vis, colors, cmap='viridis', scale=25, width=0.005)
    plt.colorbar(label='|B| (T) (clipeado)')
    
    # Dibujar geometría de la fuente
    if geometria:
        if geometria['tipo'] == 'alambre' or geometria['tipo'] == 'ambos':
            L = geometria.get('L', 2)
            plt.plot([0, 0], [-L/2, L/2], 'r-', linewidth=3, label='Alambre')
            plt.plot([0, 0], [-L/2, L/2], 'ro', markersize=8)
        
        if geometria['tipo'] == 'espira' or geometria['tipo'] == 'ambos':
            a = geometria.get('a', 0.5)
            theta = np.linspace(0, 2*np.pi, 100)
            plt.plot(a*np.cos(theta), a*np.sin(theta), 'b-', linewidth=3, label='Espira')
    
    plt.title(titulo)
    plt.axis('equal')
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.grid(True, alpha=0.3)
    if geometria:
        plt.legend()
    plt.show()

def graficar_3d(x, y, z, Bx, By, Bz, titulo="Campo Magnético 3D", geometria=None):
    """
    Grafica campo magnético en 3D con vectores.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calcular magnitud
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
    
    # Normalizar vectores (longitud fija para 3D suele ser mejor para ver estructura)
    B_mag_safe = np.where(B_mag == 0, 1e-9, B_mag)
    Bx_unit = Bx / B_mag_safe
    By_unit = By / B_mag_safe
    Bz_unit = Bz / B_mag_safe
    
    # Colorear por magnitud
    # Matplotlib quiver 3D no soporta colormap fácilmente en versiones antiguas,
    # pero intentaremos pasar colores si es posible, o usar longitud fija.
    # Para robustez en mpl 3d, usamos longitud fija y color sólido o variable si se puede.
    # Simplificación: Longitud fija proporcional a una escala pequeña
    
    length = 0.1
    
    # Quiver plot
    # Nota: mpl 3d quiver no soporta array de colores fácilmente en todas las versiones
    # Usaremos color por magnitud normalizada si es posible, sino un color sólido
    
    q = ax.quiver(x, y, z, Bx_unit, By_unit, Bz_unit, 
                  length=length, normalize=True, 
                  cmap='viridis', linewidth=1.5)
    
    # Dibujar geometría
    if geometria:
        if geometria['tipo'] == 'alambre' or geometria['tipo'] == 'ambos':
            L = geometria.get('L', 2)
            zs = np.linspace(-L/2, L/2, 50)
            ax.plot([0]*len(zs), [0]*len(zs), zs, 'r-', linewidth=3, label='Alambre')
        
        if geometria['tipo'] == 'espira' or geometria['tipo'] == 'ambos':
            a = geometria.get('a', 0.5)
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(a*np.cos(theta), a*np.sin(theta), [0]*len(theta), 
                   'b-', linewidth=3, label='Espira')
    
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title(titulo)
    if geometria:
        ax.legend()
    plt.show()
