import cv2
import numpy as np
import os
import re

# Pasta onde ficam guardadas as imagens de referência das cartas
PASTA_REFERENCIAS = "referencias"

def deslocar_imagem(img, dx, dy):
    """Empurra o desenho um pouco para os lados, para cima ou para baixo, preenchendo as bordas de preto."""
    linhas, colunas = img.shape
    matriz = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, matriz, (colunas, linhas), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def rotacionar_leve(img, angulo):
    """Gira a imagem bem de leve mantendo o mesmo tamanho original e fundo preto."""
    linhas, colunas = img.shape
    centro = (colunas / 2.0, linhas / 2.0)
    matriz = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    return cv2.warpAffine(img, matriz, (colunas, linhas), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def salvar_nova_variacao(nome_carta, img_bin):
    """Garante que a imagem seja preto e branco puro e salva na pasta sem substituir arquivos antigos."""
    _, img_bin = cv2.threshold(img_bin, 127, 255, cv2.THRESH_BINARY)
    contador = 2
    while True:
        caminho = os.path.join(PASTA_REFERENCIAS, f"{nome_carta}-{contador}.png")
        if not os.path.exists(caminho):
            cv2.imwrite(caminho, img_bin)
            return
        contador += 1

def main():
    # Verifica se a pasta de referências realmente existe antes de começar
    if not os.path.exists(PASTA_REFERENCIAS):
        print(f"Pasta '{PASTA_REFERENCIAS}' não encontrada.")
        return

    # Busca apenas as fotos originais principais para evitar multiplicar cópias repetidas
    arquivos_originais = [
        f for f in os.listdir(PASTA_REFERENCIAS)
        if f.lower().endswith("-1.png")
    ]

    # Caso não encontre arquivos terminados com o sufixo padrão, lê todas as imagens PNG da pasta
    if not arquivos_originais:
        arquivos_originais = [
            f for f in os.listdir(PASTA_REFERENCIAS)
            if f.lower().endswith(".png")
        ]

    # Dicionários para separar e guardar os desenhos de números/letras e os desenhos dos naipes
    ranks_por_valor = {}
    naipes_por_tipo = {"Copas": [], "Espadas": [], "Ouros": [], "Paus": []}
    originais_carregadas = {}

    for arq in arquivos_originais:
        # Limpa o nome do arquivo para descobrir qual é o valor e qual é o naipe
        nome_base = re.sub(r'[_\-\s]*\d+$', '', os.path.splitext(arq)[0])
        partes = nome_base.split("_")
        if len(partes) != 2:
            continue

        valor, naipe = partes[0], partes[1]
        img = cv2.imread(os.path.join(PASTA_REFERENCIAS, arq), cv2.IMREAD_GRAYSCALE)
        
        # Ignora arquivos corrompidos ou que estejam fora do tamanho padrão do recorte
        if img is None or img.shape != (80, 30):
            continue

        # Separa a metade de cima (número/letra) e a metade de baixo (naipe)
        originais_carregadas[nome_base] = img
        topo = img[0:41, :]
        base = img[41:80, :]

        ranks_por_valor.setdefault(valor, []).append(topo)
        if naipe in naipes_por_tipo:
            naipes_por_tipo[naipe].append(base)

    total_geradas = 0

    # Percorre cada carta original para fabricar suas novas variações
    for nome_base, img_orig in originais_carregadas.items():
        valor, naipe = nome_base.split("_")
        variacoes = []

        # Mistura o número/letra desta carta com outros recortes reais do mesmo naipe
        if naipe in naipes_por_tipo and len(naipes_por_tipo[naipe]) > 1:
            topo_atual = img_orig[0:41, :]
            for base_extra in naipes_por_tipo[naipe][:2]:
                combinada = np.vstack([topo_atual, base_extra])
                variacoes.append(combinada)

        # Cria versões levemente deslocadas para simular pequenas mudanças de posição na câmera
        deslocamentos = [
            (-1, 0), (1, 0),   # Esquerda e Direita
            (0, -1), (0, 1),   # Cima e Baixo
            (-1, -1), (1, 1)   # Diagonais
        ]
        for dx, dy in deslocamentos:
            variacoes.append(deslocar_imagem(img_orig, dx, dy))

        # Cria versões levemente inclinadas para ajudar quando a carta estiver torta na mesa
        for angulo in [-2.5, 2.5]:
            variacoes.append(rotacionar_leve(img_orig, angulo))

        # Grava todas as novas imagens criadas na pasta de referências
        for var in variacoes:
            salvar_nova_variacao(nome_base, var)
            total_geradas += 1

    print(f"Sucesso! {len(originais_carregadas)} cartas originais processadas.")
    print(f"Foram geradas {total_geradas} novas variações limpas na pasta '{PASTA_REFERENCIAS}'.")

if __name__ == "__main__":
    main()