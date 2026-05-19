import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Configuração da página e visual
st.set_page_config(page_title="Estação de Solo", layout="wide", page_icon="🛰️")

# Dicionário de cores
COLOR_DICT = {
    0: [60, 16, 152],   # Prédios (Roxo)
    1: [132, 41, 246],  # Terra (Lilás)
    2: [110, 193, 228], # Estradas (Azul claro)
    3: [254, 221, 58],  # Vegetação (Amarelo)
    4: [226, 169, 41],  # Água (Dourado/Laranja)
    5: [155, 155, 155]  # Sem rótulo (Cinza)
}

# Carrega o Modelo com cache para não recarregar a cada clique
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('modelo_unet.keras')

# Inferência
def preprocessar_imagem(image_pil):
    # Força a imagem para 256x256 e extrai matrizes RGB normais
    img = image_pil.resize((512, 512)) # lembrar de manter igual ao que foi usado no treinamento
    img_array = np.array(img).astype(np.float32) / 255.0
    
    R = img_array[:, :, 0]
    G = img_array[:, :, 1]
    B = img_array[:, :, 2]
    
    # Recalcula as features matemáticas em tempo real
    exg = 2.0 * G - R - B
    cive = 0.441 * R - 0.811 * G + 0.385 * B + 18.787
    
    exg = np.expand_dims(exg, axis=-1)
    cive = np.expand_dims(cive, axis=-1)
    
    # Monta os 5 canais e adiciona a dimensão do lote (Batch)
    img_5_channels = np.concatenate([img_array, exg, cive], axis=-1)
    return np.expand_dims(img_5_channels, axis=0)

def decodificar_mascara(predicao):
    # A predição sai em probabilidades, pegamos a classe com maior chance
    mask_indices = np.argmax(predicao[0], axis=-1)
    
    # Criamos uma tela em branco RGB e pintamos conforme o dicionário
    rgb_mask = np.zeros((512, 512, 3), dtype=np.uint8)
    for class_idx, color in COLOR_DICT.items():
        rgb_mask[mask_indices == class_idx] = color
        
    return rgb_mask

# FRONT-END STREAMLIT
st.title("🛰️ Estação de Solo: Segmentação Semântica Multiespectral")
st.markdown("""
Este painel simula a inferência do Computador de Bordo. 
Ao enviar uma imagem RGB, o sistema calcula os índices **ExG** e **CIVE** em tempo real antes de submeter os 5 canais à rede convolucional U-Net.
""")

try:
    model = load_model()
except Exception as e:
    st.error("⚠️ Modelo não encontrado! Treine e salve o `modelo_unet.keras` primeiro.")
    st.stop()

uploaded_file = st.file_uploader("Suba uma imagem aérea (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Mostra a interface em duas colunas lado a lado
    image = Image.open(uploaded_file).convert('RGB')
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagem do Sensor Ótico (RGB)")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("Análise U-Net Predita")
        with st.spinner('Calculando matrizes e executando rede neural...'):
            # Roda a inteligência do sistema
            input_tensor = preprocessar_imagem(image)
            predicao_bruta = model.predict(input_tensor)
            mascara_colorida = decodificar_mascara(predicao_bruta)
            
        st.image(mascara_colorida, use_container_width=True)
        
        # Legenda técnica para o avaliador
        # todo: Depois procurar por melhorar esses emoticons coloridos
        st.markdown("**Classes Detectadas:**")
        st.markdown("🟣 **Construções** | 🟪 **Terra** | 🟦 **Estradas** | 🟨 **Vegetação** | 🟧 **Água**")