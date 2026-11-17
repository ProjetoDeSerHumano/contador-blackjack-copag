import cv2
import numpy as np
import os
import re

# Configurações de dimensões para normalizar a carta (proporção padrão Poker/Bridge)
LARGURA_CARTA = 200
ALTURA_CARTA = 300

# Dimensões da área do canto (índice: número + naipe) da Copag 157
CANTO_LARGURA = 35
CANTO_ALTURA = 85

PASTA_REFERENCIAS = "referencias"
if not os.path.exists(PASTA_REFERENCIAS):
    os.makedirs(PASTA_REFERENCIAS)

def ordenar_pontos(pts):
    """Ordena os 4 cantos da carta: topo-esq, topo-dir, baixo-dir, baixo-esq."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def achatar_carta(imagem, pontos):
    """Corrige a perspectiva da carta inclinada para uma visão perfeitamente reta."""
    rect = ordenar_pontos(pontos)
    dst = np.array([
        [0, 0],
        [LARGURA_CARTA - 1, 0],
        [LARGURA_CARTA - 1, ALTURA_CARTA - 1],
        [0, ALTURA_CARTA - 1]
    ], dtype="float32")
    
    matriz = cv2.getPerspectiveTransform(rect, dst)
    carta_reta = cv2.warpPerspective(imagem, matriz, (LARGURA_CARTA, ALTURA_CARTA))
    return carta_reta

def processar_canto(carta_reta):
    """Extrai e binariza (preto e branco puro) o canto superior esquerdo."""
    canto = carta_reta[5:CANTO_ALTURA, 5:CANTO_LARGURA]
    cinza = cv2.cvtColor(canto, cv2.COLOR_BGR2GRAY)
    _, binarizada = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binarizada


def carregar_referencias():
    """Carrega múltiplas referências agrupadas pelo nome limpo da carta."""
    refs = {}
    total_arquivos = 0
    for arquivo in os.listdir(PASTA_REFERENCIAS):
        if arquivo.lower().endswith(".png"):
            nome_sem_ext = os.path.splitext(arquivo)[0]
            
            # Remove qualquer numeração no final do nome (ex: _1, -2, 3, etc.)
            nome_carta = re.sub(r'[_\-\s]*\d+$', '', nome_sem_ext)
            
            # Caso o arquivo seja só um número (ex: "7.png"), mantém o nome original
            if not nome_carta:
                nome_carta = nome_sem_ext

            img = cv2.imread(os.path.join(PASTA_REFERENCIAS, arquivo), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                if nome_carta not in refs:
                    refs[nome_carta] = []
                refs[nome_carta].append(img)
                total_arquivos += 1
    return refs, total_arquivos

def salvar_referencia(nome_carta, imagem_canto):
    """Salva uma nova variação da carta sem sobrescrever as existentes."""
    # Limpa caso o usuário digite o número sem querer
    nome_limpo = re.sub(r'[_\-\s]*\d+$', '', nome_carta) or nome_carta
    
    contador = 1
    while True:
        caminho = os.path.join(PASTA_REFERENCIAS, f"{nome_limpo}-{contador}.png")
        if not os.path.exists(caminho):
            cv2.imwrite(caminho, imagem_canto)
            return caminho
        contador += 1 

def identificar_carta(carta_reta, referencias):
    """Testa a carta normal e rotacionada em 180° contra todas as variações salvas."""
    if not referencias:
        return "Sem Referencia", 0

    # Extrai o canto normal e o canto oposto (caso a carta esteja de cabeça para baixo)
    canto_normal = processar_canto(carta_reta)
    carta_180 = cv2.rotate(carta_reta, cv2.ROTATE_180)
    canto_invertido = processar_canto(carta_180)
    
    candidatos = [canto_normal, canto_invertido]

    melhor_nome = "Desconhecida"
    menor_diferenca = float("inf")
    
    # Limiar máximo de pixels diferentes aceitável
    LIMIAR_CONFIANCA = 850 

    for canto_atual in candidatos:
        for nome, lista_imgs in referencias.items():
            for img_ref in lista_imgs:
                if canto_atual.shape != img_ref.shape:
                    continue
                    
                diff = cv2.absdiff(canto_atual, img_ref)
                pontuacao = np.sum(diff) / 255
                
                if pontuacao < menor_diferenca:
                    menor_diferenca = pontuacao
                    melhor_nome = nome

    if menor_diferenca == float("inf"):
        return "Desconhecida", 0

    if menor_diferenca < LIMIAR_CONFIANCA:
        return melhor_nome, int(menor_diferenca)
    return "Desconhecida", int(menor_diferenca)

def main():
    cap = cv2.VideoCapture(0)
    referencias, total_fotos = carregar_referencias()
    print(f"Sistema iniciado. {len(referencias)} cartas ({total_fotos} fotos de referencia) carregadas.")
    print("Pressione 'C' para cadastrar uma variação da carta | 'Q' para sair.")

    ultima_carta_registrada = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(cinza, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 140, 255, cv2.THRESH_BINARY)

        contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        canto_para_salvar = None

        for cnt in contornos:
            area = cv2.contourArea(cnt)
            if area > 5000:
                perimetro = cv2.arcLength(cnt, True)
                aprox = cv2.approxPolyDP(cnt, 0.02 * perimetro, True)

                if len(aprox) == 4:
                    pontos = aprox.reshape(4, 2)
                    carta_reta = achatar_carta(frame, pontos)
                    canto_para_salvar = processar_canto(carta_reta)

                    nome_carta, erro = identificar_carta(carta_reta, referencias)

                    # Cor verde se reconheceu, amarela se ainda é desconhecida
                    cor = (0, 255, 0) if nome_carta not in ["Desconhecida", "Sem Referencia"] else (0, 255, 255)
                    
                    cv2.drawContours(frame, [aprox], -1, cor, 3)
                    x, y = pontos[0]
                    texto = f"{nome_carta} (Erro: {erro})"
                    cv2.putText(frame, texto, (int(x), int(y) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)
                    
                    cv2.imshow("Canto Processado", cv2.resize(canto_para_salvar, (105, 255)))

        cv2.imshow("Detector Copag 157", frame)
        tecla = cv2.waitKey(1) & 0xFF

        # Salvar nova variação da carta
        if tecla == ord('c') and canto_para_salvar is not None:
            sugestao = f" [Enter para repetir '{ultima_carta_registrada}']" if ultima_carta_registrada else ""
            entrada = input(f"\nNome da carta (ex: A_Espadas){sugestao}: ").strip()
            
            nome_final = entrada if entrada else ultima_carta_registrada

            if nome_final:
                caminho_salvo = salvar_referencia(nome_final, canto_para_salvar)
                ultima_carta_registrada = nome_final
                referencias, total_fotos = carregar_referencias()
                qtd_var = len(referencias[nome_final])
                print(f"Salvo em '{caminho_salvo}'! Total de variações para {nome_final}: {qtd_var}")

        elif tecla == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()