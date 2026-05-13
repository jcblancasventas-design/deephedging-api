# ============================================
# DEEP HEDGING API - FLASK VERSION
# ============================================
# AUTOR: Juan Carlos Blancas Garcia
# FECHA: 3 de Mayo 2026
# ============================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os

app = Flask(__name__)
CORS(app)  # Permite llamadas desde cualquier origen

# ============================================
# CONFIGURACIÓN DE SEGURIDAD
# ============================================
import os
API_KEY = os.environ.get("API_KEY")  # se carga desde variable de entorno

def verificar_api_key():
    """Verifica que la solicitud tenga una API Key válida"""
    api_key = request.headers.get('X-API-Key')
    return api_key == API_KEY

print("="*50)
print("DEEP HEDGING API INICIADA")
print("="*50)

# ============================================
# DISPOSITIVO
# ============================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n Dispositivo de cómputo: {device}")

# ============================================
# 1. MODELO ACTOR-CRÍTICO
# ============================================
print("\n CONSTRUYENDO MODELO ACTOR-CRÍTICO")
print("-"*40)

class RobustActor(nn.Module):
    def __init__(self, state_dim=15, action_dim=3, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, action_dim), nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)

class RobustCritic(nn.Module):
    def __init__(self, state_dim=15, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x)

# Instanciar modelo
robust_actor = RobustActor().to(device)
robust_actor.eval()

print(f"Modelo cargado: {sum(p.numel() for p in robust_actor.parameters())} parámetros")

# ============================================
# 2. FUNCIÓN DE ESTADO
# ============================================
def get_state(trajectory, step, n_assets):
    start = max(0, step - 4)
    state = trajectory[start:step+1].flatten()
    expected_len = n_assets * 5
    if len(state) < expected_len:
        state = np.pad(state, (0, expected_len - len(state)))
    return torch.FloatTensor(state[:expected_len]).to(device)

print("Función de estado definida")

# ============================================
# 3. BLOCK BOOTSTRAP
# ============================================
def block_bootstrap(data_df, block_size=20, n_blocks=13, seq_length=252):
    n = len(data_df)
    
    if n < block_size + 1:
        raise ValueError(f"Se requieren al menos {block_size + 1} filas. El archivo tiene solo {n} filas.")
    
    if n < seq_length:
        seq_length = n - 1
        print(f"Datos insuficientes. Se usará seq_length={seq_length}")
    
    indices = []
    for _ in range(n_blocks):
        start = np.random.randint(0, n - block_size)
        indices.extend(list(range(start, start + block_size)))
    indices = indices[:seq_length]
    return data_df.iloc[indices].values

print("Block bootstrap definido")

# ============================================
# 4. FUNCIÓN PRINCIPAL DE ANÁLISIS
# ============================================
def analizar_cobertura(prices_df):
    returns_df = prices_df.pct_change().dropna()
    
    np.random.seed(42)
    n_trajectories = 64
    seq_length = 252
    n_assets = returns_df.shape[1]
    
    trajectories = np.zeros((n_trajectories, seq_length, n_assets))
    for i in range(n_trajectories):
        trajectories[i] = block_bootstrap(returns_df)
    
    robust_actor.eval()
    lista_acciones = []
    
    for idx in range(min(20, len(trajectories))):
        trayectoria = trajectories[idx]
        for paso in range(seq_length - 1):
            estado = get_state(trayectoria, paso, n_assets)
            with torch.no_grad():
                accion = robust_actor(estado).cpu().numpy()
            lista_acciones.append(accion)
    
    acciones = np.array(lista_acciones)
    
    pos_promedio_abs = np.mean(np.abs(acciones))
    prop_mayor_01 = np.mean(np.abs(acciones) > 0.1)
    pos_max = np.max(np.abs(acciones))
    pos_min = np.min(acciones)
    
    if pos_promedio_abs < 0.1 and prop_mayor_01 < 0.01:
        conclusion = "INACCION"
        recomendacion = "No implementar Deep Hedging. Usar forwards USD/PEN y futuros de cobre LME."
    else:
        conclusion = "ACTIVA"
        recomendacion = "El agente aprendió una política activa. Se requiere análisis adicional."
    
    return {
        'posicion_promedio_abs': round(float(pos_promedio_abs), 4),
        'proporcion_mayor_01': round(float(prop_mayor_01), 3),
        'posicion_maxima': round(float(pos_max), 6),
        'posicion_minima': round(float(pos_min), 4),
        'conclusion': conclusion,
        'recomendacion': recomendacion
    }

# ============================================
# 5. ENDPOINTS
# ============================================
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'nombre': 'Deep Hedging API - Sector Minero Peruano',
        'autor': 'Juan Carlos Blancas Garcia',
        'version': '1.0',
        'endpoints': {
            '/simular': 'POST - Enviar archivo CSV con precios',
            '/health': 'GET - Verificar estado del servicio'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'API funcionando correctamente'})

@app.route('/simular', methods=['POST'])
def simular():
    # Verificar autenticación
    if not verificar_api_key():
        return jsonify({'error': 'No autorizado. API Key inválida o faltante'}), 401
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No se envió ningún archivo'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Nombre de archivo vacío'}), 400
        
        # Leer CSV
        df = pd.read_csv(file)
        
        # Verificar columnas
        required = ['SCCO', 'MINSUR', 'COPPER']
        missing = [col for col in required if col not in df.columns]
        if missing:
            return jsonify({'error': f'Faltan columnas: {missing}'}), 400
        
        # Ajustar formato si hay columna fecha
        if 'date' in df.columns or 'fecha' in df.columns:
            date_col = 'date' if 'date' in df.columns else 'fecha'
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        
        prices_df = df[required].dropna()
        
        if len(prices_df) < 20:
            return jsonify({'error': f'Se requieren al menos 20 filas de datos. El archivo tiene {len(prices_df)} filas.'}), 400
        
        # Ejecutar análisis
        print(f"Analizando datos con {len(prices_df)} registros...")
        resultado = analizar_cobertura(prices_df)
        
        resultado['registros_analizados'] = len(prices_df)
        resultado['activos'] = list(prices_df.columns)
        
        print(f"Resultado: {resultado['conclusion']}")
        
        return jsonify(resultado)
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

# ============================================
# 6. EJECUTAR API
# ============================================
if __name__ == '__main__':
    print("\n" + "="*50)
    print("API DISPONIBLE EN: http://localhost:10000")
    print("Endpoint de simulación: POST http://localhost:10000/simular")
    print("="*50)
    app.run(debug=True, host='0.0.0.0', port=10000)
