import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def crear_grafico_2d_plotly(xx, yy, Bx, By, titulo="Campo Magnético 2D", geometria=None, xlabel='x (m)', ylabel='y (m)', plano='XY'):
    """
    Crea un gráfico 2D interactivo del campo magnético usando Plotly.
    """
    # Calcular magnitud del campo
    B_mag = np.sqrt(Bx**2 + By**2)
    
    # Crear figura
    fig = go.Figure()
    
    # Añadir heatmap de magnitud
    fig.add_trace(go.Heatmap(
        x=xx[0, :],
        y=yy[:, 0],
        z=B_mag,
        colorscale='Viridis',
        colorbar=dict(title='|B| (T)'),
        name='Magnitud',
        hovertemplate=f'{xlabel.split()[0]}: %{{x:.3f}}<br>{ylabel.split()[0]}: %{{y:.3f}}<br>|B|: %{{z:.3e}} T<extra></extra>'
    ))
    
    # Añadir vectores del campo (subsample para no saturar)
    step = max(1, len(xx) // 15)
    xx_sub = xx[::step, ::step]
    yy_sub = yy[::step, ::step]
    Bx_sub = Bx[::step, ::step]
    By_sub = By[::step, ::step]
    B_mag_sub = B_mag[::step, ::step]
    
    # --- NORMALIZACIÓN ROBUSTA ---
    B_visual_max = np.percentile(B_mag_sub, 90) if len(B_mag_sub.flat) > 0 else 1.0
    if B_visual_max == 0: B_visual_max = 1.0
    
    scale = 0.15 * (xx.max() - xx.min()) / (xx_sub.shape[0]) * 5 
    
    for i in range(xx_sub.shape[0]):
        for j in range(xx_sub.shape[1]):
            x0, y0 = xx_sub[i, j], yy_sub[i, j]
            bx, by = Bx_sub[i, j], By_sub[i, j]
            b_mag = B_mag_sub[i, j]
            
            if b_mag == 0: continue
            
            ux, uy = bx / b_mag, by / b_mag
            arrow_len = scale * (0.5 + 0.5 * np.minimum(b_mag / B_visual_max, 1.0))
            dx, dy = ux * arrow_len, uy * arrow_len
            
            fig.add_trace(go.Scatter(
                x=[x0, x0 + dx], y=[y0, y0 + dy], mode='lines',
                line=dict(color='white', width=1.5), showlegend=False, hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=[x0 + dx], y=[y0 + dy], mode='markers',
                marker=dict(symbol='arrow', size=6, color='white', angle=np.degrees(np.arctan2(dy, dx)), angleref='previous'),
                showlegend=False, hoverinfo='skip'
            ))
    
    # Dibujar geometría según el plano
    if geometria:
        plano_key = plano.split()[0] # 'XY', 'XZ', 'YZ'
        
        # --- ALAMBRE ---
        if geometria['tipo'] in ['alambre', 'ambos']:
            L = geometria.get('L', 2)
            z_off = geometria.get('z_offset_alambre', 0)
            I_val = geometria.get('I', geometria.get('I_alambre', 1.0))
            signo = 1 if I_val >= 0 else -1
            
            if plano_key == 'XY':
                # Alambre es un punto en el origen
                # Si I > 0 (hacia +z): Saliente (punto)
                # Si I < 0 (hacia -z): Entrante (cruz)
                symbol = 'circle-dot' if signo > 0 else 'circle-x'
                label = 'Saliente (+z)' if signo > 0 else 'Entrante (-z)'
                
                fig.add_trace(go.Scatter(
                    x=[0], y=[0], mode='markers',
                    marker=dict(size=18, color='red', symbol=symbol, line=dict(width=2, color='black')),
                    name=f'Alambre ({label})',
                    hoverinfo='name'
                ))
            elif plano_key in ['XZ', 'YZ']:
                # Alambre es una línea vertical
                fig.add_trace(go.Scatter(
                    x=[0, 0], y=[z_off - L/2, z_off + L/2], mode='lines',
                    line=dict(color='red', width=5),
                    name='Alambre'
                ))
                # Flecha de corriente
                # Si I > 0: arrow-up, Si I < 0: arrow-down
                arrow_sym = 'arrow-up' if signo > 0 else 'arrow-down'
                fig.add_trace(go.Scatter(
                    x=[0], y=[z_off], mode='markers',
                    marker=dict(symbol=arrow_sym, size=20, color='red', line=dict(width=1, color='black')),
                    name=f'I ({"Hacia arriba" if signo > 0 else "Hacia abajo"})',
                    hoverinfo='name'
                ))

        # --- ESPIRA ---
        if geometria['tipo'] in ['espira', 'ambos']:
            a = geometria.get('a', 0.5)
            z_off = geometria.get('z_offset_espira', 0)
            I_val = geometria.get('I', geometria.get('I_espira', 1.0))
            signo = 1 if I_val >= 0 else -1
            
            if plano_key == 'XY':
                # Espira es un círculo
                theta = np.linspace(0, 2*np.pi, 100)
                fig.add_trace(go.Scatter(
                    x=a*np.cos(theta), y=a*np.sin(theta), mode='lines',
                    line=dict(color='cyan', width=4),
                    name='Espira'
                ))
                # Flechas de corriente
                # Antihorario (I>0) vs Horario (I<0)
                theta_arrow = [0, np.pi/2, np.pi, 3*np.pi/2]
                for th in theta_arrow:
                    # Tangente: (-sin, cos) para antihorario
                    # Si signo < 0, invertimos vector tangente -> (sin, -cos)
                    vec_x = -np.sin(th) * signo
                    vec_y = np.cos(th) * signo
                    angle = np.degrees(np.arctan2(vec_y, vec_x))
                    
                    fig.add_trace(go.Scatter(
                        x=[a*np.cos(th)], y=[a*np.sin(th)], mode='markers',
                        marker=dict(symbol='arrow', size=15, color='cyan', angle=angle, line=dict(width=1, color='black')),
                        showlegend=False, hoverinfo='skip'
                    ))
                    
            elif plano_key in ['XZ', 'YZ']:
                # Espira son dos puntos
                # Izquierda (-a): 
                #   Si Antihorario (I>0): en -a la corriente SALE (punto)
                #   Si Horario (I<0): en -a la corriente ENTRA (cruz)
                sym_left = 'circle-dot' if signo > 0 else 'circle-x'
                sym_right = 'circle-x' if signo > 0 else 'circle-dot'
                
                fig.add_trace(go.Scatter(
                    x=[-a], y=[z_off], mode='markers',
                    marker=dict(size=15, color='cyan', symbol=sym_left, line=dict(width=2, color='black')),
                    name='I Izq',
                    hoverinfo='name'
                ))
                fig.add_trace(go.Scatter(
                    x=[a], y=[z_off], mode='markers',
                    marker=dict(size=15, color='cyan', symbol=sym_right, line=dict(width=2, color='black')),
                    name='I Der',
                    hoverinfo='name'
                ))

        # --- BOBINAS DE HELMHOLTZ ---
        if geometria['tipo'] == 'helmholtz':
            R = geometria.get('R', 0.5)
            I_val = geometria.get('I', 1.0)
            signo = 1 if I_val >= 0 else -1
            
            if plano_key == 'XY':
                theta = np.linspace(0, 2*np.pi, 100)
                fig.add_trace(go.Scatter(
                    x=R*np.cos(theta), y=R*np.sin(theta), mode='lines',
                    line=dict(color='orange', width=4),
                    name='Bobinas Helmholtz'
                ))
                # Flechas
                theta_arrow = [0, np.pi/2, np.pi, 3*np.pi/2]
                for th in theta_arrow:
                    vec_x = -np.sin(th) * signo
                    vec_y = np.cos(th) * signo
                    angle = np.degrees(np.arctan2(vec_y, vec_x))
                    
                    fig.add_trace(go.Scatter(
                        x=[R*np.cos(th)], y=[R*np.sin(th)], mode='markers',
                        marker=dict(symbol='arrow', size=15, color='orange', angle=angle, line=dict(width=1, color='black')),
                        showlegend=False, hoverinfo='skip'
                    ))
            elif plano_key in ['XZ', 'YZ']:
                # Bobina 1 (-R/2) y Bobina 2 (+R/2)
                sym_left = 'circle-dot' if signo > 0 else 'circle-x'
                sym_right = 'circle-x' if signo > 0 else 'circle-dot'
                
                for z_c in [-R/2, R/2]:
                    fig.add_trace(go.Scatter(
                        x=[-R], y=[z_c], mode='markers',
                        marker=dict(size=15, color='orange', symbol=sym_left, line=dict(width=2, color='black')),
                        showlegend=False, hoverinfo='name', name='I Izq'
                    ))
                    fig.add_trace(go.Scatter(
                        x=[R], y=[z_c], mode='markers',
                        marker=dict(size=15, color='orange', symbol=sym_right, line=dict(width=2, color='black')),
                        showlegend=False, hoverinfo='name', name='I Der'
                    ))
    
    fig.update_layout(
        title=titulo,
        xaxis=dict(title=xlabel, scaleanchor='y', scaleratio=1),
        yaxis=dict(title=ylabel),
        autosize=True,
        showlegend=True,
        hovermode='closest',
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def crear_grafico_3d_plotly(x, y, z, Bx, By, Bz, titulo="Campo Magnético 3D", geometria=None):
    """
    Crea un gráfico 3D interactivo del campo magnético usando Plotly.
    """
    # Calcular magnitud
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
    
    # --- NORMALIZACIÓN ROBUSTA PARA CONOS ---
    # Usamos vectores unitarios para la dirección y dejamos que el color indique la magnitud.
    # Plotly Cone usa (u,v,w) para tamaño Y dirección.
    # Si normalizamos (u,v,w), todos los conos tendrán el mismo tamaño (controlado por sizeref).
    # Esto es ideal para visualizar la estructura del campo sin que las singularidades oculten todo.
    
    # Evitar división por cero
    B_mag_safe = np.where(B_mag == 0, 1e-9, B_mag)
    
    Bx_unit = Bx / B_mag_safe
    By_unit = By / B_mag_safe
    Bz_unit = Bz / B_mag_safe
    
    # Crear figura
    fig = go.Figure()
    
    # Añadir vectores usando Cone plot
    # Usamos vectores unitarios -> tamaño uniforme
    # Coloreamos por la magnitud real (B_mag)
    
    fig.add_trace(go.Cone(
        x=x, y=y, z=z,
        u=Bx_unit, v=By_unit, w=Bz_unit,
        colorscale='Viridis',
        sizemode='absolute',
        sizeref=0.5, # Tamaño fijo de los conos
        cmin=np.min(B_mag),
        cmax=np.percentile(B_mag, 95), # Saturar el color al 95% para ver contraste
        colorbar=dict(title='|B| (T)'),
        hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<br>|B|: %{customdata:.3e} T<extra></extra>',
        customdata=B_mag, # Pasar magnitud real para el tooltip
        showscale=True
    ))
    
    # Dibujar geometría
    if geometria:
        if geometria['tipo'] in ['alambre', 'ambos']:
            L = geometria.get('L', 2)
            z_offset = geometria.get('z_offset_alambre', 0)
            I_val = geometria.get('I', geometria.get('I_alambre', 1.0))
            signo = 1 if I_val >= 0 else -1
            
            zs = np.linspace(z_offset - L/2, z_offset + L/2, 50)
            fig.add_trace(go.Scatter3d(
                x=np.zeros_like(zs),
                y=np.zeros_like(zs),
                z=zs,
                mode='lines',
                line=dict(color='red', width=8),
                name='Alambre',
                hovertemplate=f'Alambre<br>L={L} m<extra></extra>'
            ))
            # Cono indicador dirección corriente
            fig.add_trace(go.Cone(
                x=[0], y=[0], z=[z_offset],
                u=[0], v=[0], w=[signo],
                sizemode="absolute", sizeref=0.5, anchor="tail",
                showscale=False, colorscale=[[0, 'red'], [1, 'red']]
            ))
        
        if geometria['tipo'] in ['espira', 'ambos']:
            a = geometria.get('a', 0.5)
            z_offset = geometria.get('z_offset_espira', 0)
            I_val = geometria.get('I', geometria.get('I_espira', 1.0))
            signo = 1 if I_val >= 0 else -1
            
            theta = np.linspace(0, 2*np.pi, 100)
            fig.add_trace(go.Scatter3d(
                x=a*np.cos(theta),
                y=a*np.sin(theta),
                z=np.full_like(theta, z_offset),
                mode='lines',
                line=dict(color='cyan', width=8),
                name='Espira',
                hovertemplate=f'Espira<br>Radio={a} m<extra></extra>'
            ))
            # Conos indicadores dirección corriente
            th_arrows = [0, np.pi/2, np.pi, 3*np.pi/2]
            x_arr = a * np.cos(th_arrows)
            y_arr = a * np.sin(th_arrows)
            z_arr = np.full_like(x_arr, z_offset)
            # Tangentes: (-sin, cos, 0) * signo
            u_arr = -np.sin(th_arrows) * signo
            v_arr = np.cos(th_arrows) * signo
            w_arr = np.zeros_like(u_arr)
            
            fig.add_trace(go.Cone(
                x=x_arr, y=y_arr, z=z_arr,
                u=u_arr, v=v_arr, w=w_arr,
                sizemode="absolute", sizeref=0.3, anchor="center",
                showscale=False, colorscale=[[0, 'cyan'], [1, 'cyan']]
            ))

        if geometria['tipo'] == 'helmholtz':
            R = geometria.get('R', 0.5)
            I_val = geometria.get('I', 1.0)
            signo = 1 if I_val >= 0 else -1
            theta = np.linspace(0, 2*np.pi, 100)
            
            # Flechas para ambas bobinas
            th_arrows = [0, np.pi/2, np.pi, 3*np.pi/2]
            x_arr = R * np.cos(th_arrows)
            y_arr = R * np.sin(th_arrows)
            u_arr = -np.sin(th_arrows) * signo
            v_arr = np.cos(th_arrows) * signo
            w_arr = np.zeros_like(u_arr)
            
            for z_c, name in [(-R/2, 'Bobina 1'), (R/2, 'Bobina 2')]:
                fig.add_trace(go.Scatter3d(
                    x=R*np.cos(theta),
                    y=R*np.sin(theta),
                    z=np.full_like(theta, z_c),
                    mode='lines',
                    line=dict(color='orange', width=8),
                    name=name,
                    hovertemplate=f'{name}<br>z={z_c} m<extra></extra>'
                ))
                
                fig.add_trace(go.Cone(
                    x=x_arr, y=y_arr, z=np.full_like(x_arr, z_c),
                    u=u_arr, v=v_arr, w=w_arr,
                    sizemode="absolute", sizeref=0.3, anchor="center",
                    showscale=False, colorscale=[[0, 'orange'], [1, 'orange']]
                ))
    
    # Configurar layout
    fig.update_layout(
        title=titulo,
        scene=dict(
            xaxis_title='x (m)',
            yaxis_title='y (m)',
            zaxis_title='z (m)',
            aspectmode='data' # Mantiene proporción real
        ),
        autosize=True,
        showlegend=True,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.9,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig
