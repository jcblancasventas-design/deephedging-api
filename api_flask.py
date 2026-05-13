# ============================================
# DEEP HEDGING API - FLASK VERSION
# ============================================
# AUTOR: Juan Carlos Blancas Garcia
# FECHA: 3 de Mayo 2026
# VERSIÓN: 2.0 (con modelo A2C entrenado)
# ============================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import math

app = Flask(__name__)
CORS(app)

# ============================================
# CONFIGURACIÓN DE SEGURIDAD
# ============================================
API_KEY = os.environ.get("API_KEY")

def verificar_api_key():
    """Verifica que la solicitud tenga una API Key válida."""
    api_key = request.headers.get('X-API-Key')
    return api_key == API_KEY

print("=" * 50)
print("DEEP HEDGING API v2.0 INICIADA")
print("=" * 50)

# ============================================
# DISPOSITIVO
# ============================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDispositivo de cómputo: {device}")

# ============================================
# PARÁMETROS (deben coincidir con el entrenamiento)
# ============================================
N_ASSETS    = 3
WINDOW      = 5
STATE_DIM   = N_ASSETS * WINDOW + N_ASSETS   # 15 + 3 = 18
ACTION_DIM  = N_ASSETS
HIDDEN      = 128
TRANSACTION_COST = 0.001
RISK_AVERSION    = 2.0

# ============================================
# MODELO ACTOR (igual que en el entrenamiento)
# ============================================
class Actor(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden),   nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, action_dim), nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)

# ============================================
# CARGAR MODELO ENTRENADO
# ============================================
MODEL_PATH = "actor_entrenado.pth"

actor = Actor(state_dim=STATE_DIM).to(device)

if os.path.exists(MODEL_PATH):
    actor.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    actor.eval()
    print(f"\nModelo A2C entrenado cargado desde: {MODEL_PATH}")
    print(f"Parámetros: {sum(p.numel() for p in actor.parameters()):,}")
else:
    print(f"\nADVERTENCIA: No se encontró {MODEL_PATH}")
    print("La API usará pesos aleatorios — suba el archivo del modelo entrenado.")

# ============================================
# FUNCIÓN DE ESTADO
# ============================================
def get_state(trajectory, step, prev_action):
    """Estado: retornos recientes + posición anterior."""
    start = max(0, step - WINDOW + 1)
    ret_window = trajectory[start:step + 1].flatten()
    expected = N_ASSETS * WINDOW
    if len(ret_window) < expected:
        ret_window = np.pad(ret_window, (0, expected - len(ret_window)))
    state = np.concatenate([ret_window[:expected], prev_action])
    return torch.FloatTensor(state).to(device)

# ============================================
# BLOCK BOOTSTRAP
# ============================================
def block_bootstrap(data_df, block_size=20, seq_length=252):
    n = len(data_df)
    if n < block_size + 1:
        raise ValueError(f"Se requieren al menos {block_size + 1} filas. El archivo tiene {n}.")
    seq_length = min(seq_length, n - 1)
    n_blocks = math.ceil(seq_length / block_size)
    indices = []
    for _ in range(n_blocks):
        start = np.random.randint(0, n - block_size)
        indices.extend(range(start, start + block_size))
    return data_df.iloc[indices[:seq_length]].values

# ============================================
# FUNCIÓN PRINCIPAL DE ANÁLISIS
# ============================================
def analizar_cobertura(prices_df):
    returns_df = prices_df.pct_change().dropna()

    np.random.seed(42)
    n_trajectories = 64
    seq_length     = 252
    n_assets       = returns_df.shape[1]

    trajectories = np.array([
        block_bootstrap(returns_df, seq_length=seq_length)
        for _ in range(n_trajectories)
    ])

    actor.eval()
    lista_acciones, lista_rewards = [], []

    for traj in trajectories[:20]:
        prev_action = np.zeros(n_assets)
        traj_reward = 0.0
        for step in range(len(traj) - 1):
            state_t = get_state(traj, step, prev_action)
            with torch.no_grad():
                action = actor(state_t).cpu().numpy()

            next_ret  = traj[step + 1]
            pnl       = np.dot(action, next_ret)
            delta     = np.abs(action - prev_action)
            tc        = TRANSACTION_COST * delta.sum()
            risk_pen  = RISK_AVERSION * (pnl ** 2)
            reward    = pnl - tc - risk_pen

            lista_acciones.append(action)
            traj_reward += reward
            prev_action  = action
        lista_rewards.append(traj_reward)

    acciones = np.array(lista_acciones)
    pos_promedio_abs = float(np.mean(np.abs(acciones)))
    prop_mayor_01    = float(np.mean(np.abs(acciones) > 0.1))
    pos_max          = float(np.max(np.abs(acciones)))
    pos_min          = float(np.min(acciones))
    reward_prom      = float(np.mean(lista_rewards))
    reward_std       = float(np.std(lista_rewards))

    if pos_promedio_abs > 0.1 and prop_mayor_01 > 0.5:
        conclusion    = "ACTIVA"
        recomendacion = ("El agente A2C aprendió una política de cobertura activa. "
                         "Calibrar con costos reales de la BVL antes de implementar. "
                         "Usar futuros de cobre LME para Southern Copper y "
                         "forwards USD/PEN para Minsur.")
    else:
        conclusion    = "INACCION"
        recomendacion = ("La cobertura activa no agrega valor neto. "
                         "Usar forwards USD/PEN y futuros de cobre LME como cobertura estática.")

    return {
        'version_modelo':        '2.0 - A2C entrenado',
        'posicion_promedio_abs': round(pos_promedio_abs, 4),
        'proporcion_mayor_01':   round(prop_mayor_01, 3),
        'posicion_maxima':       round(pos_max, 4),
        'posicion_minima':       round(pos_min, 4),
        'reward_promedio':       round(reward_prom, 4),
        'reward_std':            round(reward_std, 4),
        'conclusion':            conclusion,
        'recomendacion':         recomendacion,
    }

# ============================================
# ENDPOINTS
# ============================================
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'nombre':   'Deep Hedging API - Sector Minero Peruano',
        'autor':    'Juan Carlos Blancas Garcia',
        'version':  '2.0 - A2C entrenado',
        'modelo':   'Actor-Crítico con aprendizaje por refuerzo (A2C) + Block Bootstrap',
        'endpoints': {
            '/simular': 'POST - Enviar archivo CSV con precios',
            '/health':  'GET  - Verificar estado del servicio'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    modelo_cargado = os.path.exists(MODEL_PATH)
    return jsonify({
        'status':         'ok',
        'message':        'API funcionando correctamente',
        'version':        '2.0',
        'modelo_cargado': modelo_cargado
    })

@app.route('/simular', methods=['POST'])
def simular():
    if not verificar_api_key():
        return jsonify({'error': 'No autorizado. API Key inválida o faltante.'}), 401

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No se envió ningún archivo.'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Nombre de archivo vacío.'}), 400

        df = pd.read_csv(file)

        # Columna de fecha opcional
        for col in ['date', 'fecha']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df.set_index(col, inplace=True)
                break

        required = ['SCCO', 'MINSUR', 'COPPER']
        missing  = [c for c in required if c not in df.columns]
        if missing:
            return jsonify({'error': f'Faltan columnas: {missing}'}), 400

        prices_df = df[required].dropna()

        if len(prices_df) < 22:
            return jsonify({
                'error': f'Se requieren al menos 22 filas. El archivo tiene {len(prices_df)}.'
            }), 400

        print(f"Analizando {len(prices_df)} registros...")
        resultado = analizar_cobertura(prices_df)
        resultado['registros_analizados'] = len(prices_df)
        resultado['activos'] = required

        print(f"Conclusión: {resultado['conclusion']}")
        return jsonify(resultado)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

# ============================================
# EJECUTAR API
# ============================================
if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("API DISPONIBLE EN: http://localhost:10000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=10000)
